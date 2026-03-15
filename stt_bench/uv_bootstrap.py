from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


_STAMP_FILE = ".uv-requirements.sha256"


def ensure_uv_venv(entrypoint_file: str) -> None:
    repo_root = _find_repo_root(Path(entrypoint_file).resolve().parent)
    if repo_root is None:
        return

    venv_dir = repo_root / ".venv"
    venv_python = _venv_python(venv_dir)
    requirements = repo_root / "requirements.txt"
    uv = shutil.which("uv")

    if _is_running_in_venv(venv_dir, venv_python):
        return

    if not venv_python.exists():
        if uv is None:
            return
        _run([uv, "venv", str(venv_dir)], cwd=repo_root, message=f"Creating virtualenv in {venv_dir}")

    if requirements.exists() and uv is not None and _requirements_changed(venv_dir, requirements):
        _run(
            [uv, "pip", "install", "--python", str(venv_python), "-r", str(requirements)],
            cwd=repo_root,
            message="Installing Python dependencies with uv",
        )
        _write_requirements_stamp(venv_dir, requirements)

    if venv_python.exists():
        _exec_into_venv(venv_dir, venv_python)


def _find_repo_root(start_dir: Path) -> Path | None:
    for candidate in (start_dir, *start_dir.parents):
        if (candidate / "requirements.txt").exists():
            return candidate
    return None


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _is_running_in_venv(venv_dir: Path, venv_python: Path) -> bool:
    active_venv = os.environ.get("VIRTUAL_ENV")
    if active_venv and Path(active_venv).resolve() == venv_dir.resolve():
        return True
    try:
        return venv_python.exists() and Path(sys.executable).resolve() == venv_python.resolve()
    except OSError:
        return False


def _requirements_changed(venv_dir: Path, requirements: Path) -> bool:
    stamp_path = venv_dir / _STAMP_FILE
    expected = _requirements_hash(requirements)
    if not stamp_path.exists():
        return True
    return stamp_path.read_text(encoding="utf-8").strip() != expected


def _requirements_hash(requirements: Path) -> str:
    return hashlib.sha256(requirements.read_bytes()).hexdigest()


def _write_requirements_stamp(venv_dir: Path, requirements: Path) -> None:
    stamp_path = venv_dir / _STAMP_FILE
    stamp_path.write_text(_requirements_hash(requirements), encoding="utf-8")


def _run(cmd: list[str], *, cwd: Path, message: str) -> None:
    print(message, file=sys.stderr)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _exec_into_venv(venv_dir: Path, venv_python: Path) -> None:
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv_dir)
    env["PATH"] = str(venv_python.parent) + os.pathsep + env.get("PATH", "")
    os.execve(str(venv_python), [str(venv_python), *sys.argv], env)
