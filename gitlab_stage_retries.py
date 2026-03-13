#!/usr/bin/env python3
"""
gitlab_stage_retries.py

Анализирует число retry для конкретного stage в GitLab-проекте.

Логика подсчета:
- берем pipelines проекта
- для каждого pipeline запрашиваем jobs с include_retried=true
- фильтруем jobs по нужному stage
- группируем jobs по (pipeline_id, job_name)
- retries для job = attempts - 1

Примеры:
  export GITLAB_TOKEN="glpat-xxxxx"

  python3 gitlab_stage_retries.py \
    --gitlab-url "https://gitlab.example.com" \
    --project "group/subgroup/repo" \
    --stage "deploy" \
    --ref "main" \
    --created-after "2026-03-01T00:00:00Z"

  python3 gitlab_stage_retries.py \
    --gitlab-url "https://gitlab.example.com" \
    --project 1234 \
    --stage test \
    --max-pipelines 200 \
    --output-json report.json

Зависимости:
  pip install requests
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote

import requests


@dataclass
class JobSummary:
    pipeline_id: int
    pipeline_web_url: str
    pipeline_ref: str
    pipeline_status: str
    job_name: str
    stage: str
    attempts: int
    retries: int
    latest_job_id: int
    latest_job_status: str
    latest_job_web_url: str


class GitLabClient:
    def __init__(self, base_url: str, token: str, timeout: int = 30, verify_ssl: bool = True) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api/v4"
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "PRIVATE-TOKEN": token,
            "Accept": "application/json",
        })
        self.verify_ssl = verify_ssl

    def _request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None) -> requests.Response:
        url = f"{self.api_url}{path}"
        last_exc: Optional[Exception] = None

        for attempt in range(1, 5):
            try:
                resp = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                )

                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    sleep_s = int(retry_after) if retry_after and retry_after.isdigit() else min(2 ** attempt, 10)
                    time.sleep(sleep_s)
                    continue

                if 500 <= resp.status_code < 600:
                    time.sleep(min(2 ** attempt, 10))
                    continue

                resp.raise_for_status()
                return resp

            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(min(2 ** attempt, 10))

        raise RuntimeError(f"GitLab API request failed: {method} {url}: {last_exc}")

    @staticmethod
    def _next_page(resp: requests.Response) -> Optional[int]:
        value = resp.headers.get("X-Next-Page", "").strip()
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    @staticmethod
    def _project_ref(project: str) -> str:
        # Если это integer id, оставляем как есть.
        if project.isdigit():
            return project
        return quote(project, safe="")

    def iter_pipelines(
        self,
        project: str,
        ref: Optional[str] = None,
        status: Optional[str] = None,
        created_after: Optional[str] = None,
        updated_after: Optional[str] = None,
        max_pipelines: int = 100,
        per_page: int = 100,
    ) -> Iterable[Dict[str, Any]]:
        project_ref = self._project_ref(project)
        page = 1
        yielded = 0

        while True:
            params: Dict[str, Any] = {
                "page": page,
                "per_page": per_page,
                "sort": "desc",
                "order_by": "id",
            }
            if ref:
                params["ref"] = ref
            if status:
                params["status"] = status
            if created_after:
                params["created_after"] = created_after
            if updated_after:
                params["updated_after"] = updated_after

            resp = self._request("GET", f"/projects/{project_ref}/pipelines", params=params)
            items = resp.json()

            if not items:
                break

            for item in items:
                yield item
                yielded += 1
                if yielded >= max_pipelines:
                    return

            next_page = self._next_page(resp)
            if not next_page:
                break
            page = next_page

    def iter_pipeline_jobs(
        self,
        project: str,
        pipeline_id: int,
        include_retried: bool = True,
        per_page: int = 100,
    ) -> Iterable[Dict[str, Any]]:
        project_ref = self._project_ref(project)
        page = 1

        while True:
            params: Dict[str, Any] = {
                "page": page,
                "per_page": per_page,
                "include_retried": str(include_retried).lower(),
            }

            resp = self._request(
                "GET",
                f"/projects/{project_ref}/pipelines/{pipeline_id}/jobs",
                params=params,
            )
            items = resp.json()

            if not items:
                break

            for item in items:
                yield item

            next_page = self._next_page(resp)
            if not next_page:
                break
            page = next_page


def analyze_stage_retries(
    client: GitLabClient,
    project: str,
    stage: str,
    ref: Optional[str],
    status: Optional[str],
    created_after: Optional[str],
    updated_after: Optional[str],
    max_pipelines: int,
) -> Dict[str, Any]:
    summaries: List[JobSummary] = []
    retries_by_job: Counter[str] = Counter()
    retries_by_pipeline: Counter[int] = Counter()

    pipelines_scanned = 0
    pipelines_with_stage = 0
    raw_stage_job_records = 0

    for pipeline in client.iter_pipelines(
        project=project,
        ref=ref,
        status=status,
        created_after=created_after,
        updated_after=updated_after,
        max_pipelines=max_pipelines,
    ):
        pipelines_scanned += 1
        pipeline_id = int(pipeline["id"])
        stage_jobs: List[Dict[str, Any]] = []

        for job in client.iter_pipeline_jobs(project=project, pipeline_id=pipeline_id, include_retried=True):
            if job.get("stage") == stage:
                stage_jobs.append(job)

        if not stage_jobs:
            continue

        pipelines_with_stage += 1
        raw_stage_job_records += len(stage_jobs)

        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for job in stage_jobs:
            grouped[str(job.get("name", "<unknown>"))].append(job)

        for job_name, attempts_list in grouped.items():
            attempts_list.sort(key=lambda x: int(x.get("id", 0)))
            attempts = len(attempts_list)
            retries = max(0, attempts - 1)
            latest = attempts_list[-1]

            summary = JobSummary(
                pipeline_id=pipeline_id,
                pipeline_web_url=str(pipeline.get("web_url", "")),
                pipeline_ref=str(pipeline.get("ref", "")),
                pipeline_status=str(pipeline.get("status", "")),
                job_name=job_name,
                stage=stage,
                attempts=attempts,
                retries=retries,
                latest_job_id=int(latest.get("id", 0)),
                latest_job_status=str(latest.get("status", "")),
                latest_job_web_url=str(latest.get("web_url", "")),
            )
            summaries.append(summary)

            if retries > 0:
                retries_by_job[job_name] += retries
                retries_by_pipeline[pipeline_id] += retries

    total_retries = sum(item.retries for item in summaries)
    total_unique_jobs_in_stage = len(summaries)

    return {
        "project": project,
        "stage": stage,
        "filters": {
            "ref": ref,
            "status": status,
            "created_after": created_after,
            "updated_after": updated_after,
            "max_pipelines": max_pipelines,
        },
        "summary": {
            "pipelines_scanned": pipelines_scanned,
            "pipelines_with_stage": pipelines_with_stage,
            "raw_stage_job_records": raw_stage_job_records,
            "unique_jobs_in_stage": total_unique_jobs_in_stage,
            "total_retries": total_retries,
        },
        "top_jobs_by_retries": retries_by_job.most_common(20),
        "top_pipelines_by_retries": retries_by_pipeline.most_common(20),
        "jobs": [asdict(item) for item in summaries],
    }


def print_report(report: Dict[str, Any]) -> None:
    summary = report["summary"]
    print("=" * 80)
    print(f"Project:          {report['project']}")
    print(f"Stage:            {report['stage']}")
    print(f"Pipelines scanned:{summary['pipelines_scanned']}")
    print(f"Pipelines w/stage:{summary['pipelines_with_stage']}")
    print(f"Stage job records:{summary['raw_stage_job_records']}")
    print(f"Unique jobs:      {summary['unique_jobs_in_stage']}")
    print(f"Total retries:    {summary['total_retries']}")
    print("=" * 80)

    if report["filters"]:
        print("Filters:")
        for key, value in report["filters"].items():
            if value is not None:
                print(f"  - {key}: {value}")
        print("=" * 80)

    print("Top jobs by retries:")
    if report["top_jobs_by_retries"]:
        for job_name, retries in report["top_jobs_by_retries"]:
            print(f"  - {job_name}: {retries}")
    else:
        print("  - no retries found")

    print("=" * 80)
    print("Top pipelines by retries:")
    if report["top_pipelines_by_retries"]:
        for pipeline_id, retries in report["top_pipelines_by_retries"]:
            print(f"  - pipeline {pipeline_id}: {retries}")
    else:
        print("  - no retries found")

    print("=" * 80)
    print("Detailed jobs:")
    for item in sorted(report["jobs"], key=lambda x: (-x["retries"], -x["attempts"], -x["pipeline_id"], x["job_name"])):
        print(
            f"- pipeline={item['pipeline_id']} ref={item['pipeline_ref']} "
            f"job={item['job_name']} attempts={item['attempts']} retries={item['retries']} "
            f"latest_status={item['latest_job_status']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze retry count for a specific GitLab stage in a specific project."
    )
    parser.add_argument("--gitlab-url", required=True, help="Base GitLab URL, e.g. https://gitlab.example.com")
    parser.add_argument("--project", required=True, help="Project numeric ID or path like group/subgroup/repo")
    parser.add_argument("--stage", required=True, help="Stage name to analyze, e.g. deploy")
    parser.add_argument("--token", default=os.getenv("GITLAB_TOKEN"), help="GitLab token or env GITLAB_TOKEN")
    parser.add_argument("--ref", help="Optional ref filter, e.g. main")
    parser.add_argument("--status", help="Optional pipeline status filter, e.g. failed, success, running")
    parser.add_argument("--created-after", help="ISO8601, e.g. 2026-03-01T00:00:00Z")
    parser.add_argument("--updated-after", help="ISO8601, e.g. 2026-03-01T00:00:00Z")
    parser.add_argument("--max-pipelines", type=int, default=100, help="Max number of pipelines to scan")
    parser.add_argument("--output-json", help="Write full JSON report to file")
    parser.add_argument("--fail-if-total-retries-above", type=int, help="Exit 2 if total retries exceeds threshold")
    parser.add_argument("--insecure", action="store_true", help="Disable SSL verification")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.token:
        print("ERROR: provide --token or set GITLAB_TOKEN", file=sys.stderr)
        return 1

    client = GitLabClient(
        base_url=args.gitlab_url,
        token=args.token,
        verify_ssl=not args.insecure,
    )

    try:
        report = analyze_stage_retries(
            client=client,
            project=args.project,
            stage=args.stage,
            ref=args.ref,
            status=args.status,
            created_after=args.created_after,
            updated_after=args.updated_after,
            max_pipelines=args.max_pipelines,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print_report(report)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nJSON report written to: {args.output_json}")

    threshold = args.fail_if_total_retries_above
    total_retries = report["summary"]["total_retries"]
    if threshold is not None and total_retries > threshold:
        print(
            f"\nThreshold exceeded: total_retries={total_retries} > {threshold}",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
