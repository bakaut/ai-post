#!/usr/bin/env python3
from __future__ import annotations

# https://github.com/kubernetes-sigs/descheduler/tree/v0.30.1

import sys
from pathlib import Path
from typing import Any, Iterable

import yaml
from jsonschema import Draft7Validator


EXTENSION_POINTS = (
    "presort",
    "sort",
    "deschedule",
    "balance",
    "filter",
    "preevictionfilter",
)

VALID_PLUGINS: dict[str, set[str]] = {
    "DefaultEvictor": {"filter", "preevictionfilter"},
    "RemoveDuplicates": {"balance"},
    "LowNodeUtilization": {"balance"},
    "HighNodeUtilization": {"balance"},
    "RemovePodsViolatingInterPodAntiAffinity": {"deschedule"},
    "RemovePodsViolatingNodeAffinity": {"deschedule"},
    "RemovePodsViolatingNodeTaints": {"deschedule"},
    "RemovePodsViolatingTopologySpreadConstraint": {"balance"},
    "RemovePodsHavingTooManyRestarts": {"deschedule"},
    "PodLifeTime": {"deschedule"},
    "RemoveFailedPods": {"deschedule"},
}

PLUGINS_REQUIRING_CONFIG = {
    "LowNodeUtilization",
    "HighNodeUtilization",
    "RemovePodsHavingTooManyRestarts",
    "PodLifeTime",
    "RemovePodsViolatingNodeAffinity",
}

LOW_NODE_UTILIZATION_ALLOWED_ARGS = {
    "useDeviationThresholds",
    "thresholds",
    "targetThresholds",
    "numberOfNodes",
    "evictableNamespaces",
}

HIGH_NODE_UTILIZATION_ALLOWED_ARGS = {
    "thresholds",
    "numberOfNodes",
    "evictableNamespaces",
}

DEFAULT_EVICTOR_ALLOWED_ARGS = {
    "nodeSelector",
    "evictLocalStoragePods",
    "evictDaemonSetPods",
    "evictSystemCriticalPods",
    "ignorePvcPods",
    "evictFailedBarePods",
    "labelSelector",
    "priorityThreshold",
    "nodeFit",
    "minReplicas",
}

POD_LIFETIME_ALLOWED_ARGS = {
    "namespaces",
    "labelSelector",
    "maxPodLifeTimeSeconds",
    "states",
}

TOO_MANY_RESTARTS_ALLOWED_ARGS = {
    "namespaces",
    "labelSelector",
    "podRestartThreshold",
    "includingInitContainers",
    "states",
}

NODE_AFFINITY_ALLOWED_ARGS = {
    "namespaces",
    "labelSelector",
    "nodeAffinityType",
}

NODE_TAINTS_ALLOWED_ARGS = {
    "namespaces",
    "labelSelector",
    "includePreferNoSchedule",
    "excludedTaints",
    "includedTaints",
}

TOPOLOGY_SPREAD_ALLOWED_ARGS = {
    "namespaces",
    "labelSelector",
    "constraints",
    "topologyBalanceNodeFit",
}

REMOVE_FAILED_PODS_ALLOWED_ARGS = {
    "namespaces",
    "labelSelector",
    "excludeOwnerKinds",
    "minPodLifetimeSeconds",
    "reasons",
    "exitCodes",
    "includingInitContainers",
}

REMOVE_DUPLICATES_ALLOWED_ARGS = {
    "namespaces",
    "excludeOwnerKinds",
}

INTER_POD_ANTI_AFFINITY_ALLOWED_ARGS = {
    "namespaces",
    "labelSelector",
}

VALID_NODE_AFFINITY_TYPES = {
    "requiredDuringSchedulingIgnoredDuringExecution",
    "preferredDuringSchedulingIgnoredDuringExecution",
}

VALID_TOPOLOGY_CONSTRAINTS = {
    "DoNotSchedule",
    "ScheduleAnyway",
}

DESCHEDULER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["apiVersion", "kind", "profiles"],
    "properties": {
        "apiVersion": {"type": "string", "enum": ["descheduler/v1alpha2"]},
        "kind": {"type": "string", "enum": ["DeschedulerPolicy"]},
        "nodeSelector": {"type": "string"},
        "maxNoOfPodsToEvictPerNode": {"type": "integer", "minimum": 0},
        "maxNoOfPodsToEvictPerNamespace": {"type": "integer", "minimum": 0},
        "profiles": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "plugins"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "pluginConfig": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["name"],
                            "properties": {
                                "name": {"type": "string", "minLength": 1},
                                "args": {"type": "object"},
                            },
                        },
                    },
                    "plugins": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            point: {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "enabled": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "disabled": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                            }
                            for point in EXTENSION_POINTS
                        },
                    },
                },
            },
        },
    },
}


def format_path(path: Iterable[Any]) -> str:
    parts = ["$"]
    for part in path:
        if isinstance(part, int):
            parts.append(f"[{part}]")
        else:
            parts.append(f".{part}")
    return "".join(parts)


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class LintReporter:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, path: str, message: str) -> None:
        self.errors.append(f"{path}: {message}")

    def warn(self, path: str, message: str) -> None:
        self.warnings.append(f"{path}: {message}")


class PluginRuleEngine:
    def __init__(self, reporter: LintReporter) -> None:
        self.r = reporter

    def run(self, profiles: list[dict[str, Any]]) -> None:
        for profile_index, profile in enumerate(profiles):
            self._check_profile(profile_index, profile)

    def _check_profile(self, profile_index: int, profile: dict[str, Any]) -> None:
        profile_path = f"$.profiles[{profile_index}]"
        plugins_obj = profile.get("plugins", {}) or {}
        plugin_configs = profile.get("pluginConfig", []) or []

        configured_plugins: dict[str, dict[str, Any]] = {}
        for config_index, plugin_cfg in enumerate(plugin_configs):
            cfg_path = f"{profile_path}.pluginConfig[{config_index}]"
            name = plugin_cfg.get("name")
            if name in configured_plugins:
                self.r.error(cfg_path, f"duplicate pluginConfig entry for plugin '{name}'")
                continue
            configured_plugins[name] = plugin_cfg

        enabled_by_extension_point: dict[str, set[str]] = {ep: set() for ep in EXTENSION_POINTS}
        enabled_anywhere: set[str] = set()

        for extension_point, plugin_set in plugins_obj.items():
            point_path = f"{profile_path}.plugins.{extension_point}"
            enabled = plugin_set.get("enabled", []) or []
            disabled = plugin_set.get("disabled", []) or []

            overlap = set(enabled) & set(disabled)
            if overlap:
                self.r.error(
                    point_path,
                    f"the same plugin cannot be both enabled and disabled: {sorted(overlap)}",
                )

            for plugin_name in enabled:
                enabled_by_extension_point[extension_point].add(plugin_name)
                enabled_anywhere.add(plugin_name)
                self._check_plugin_extension_point(point_path, extension_point, plugin_name)

            for plugin_name in disabled:
                self._check_plugin_extension_point(point_path, extension_point, plugin_name)

        if not enabled_anywhere:
            self.r.warn(profile_path, "profile has no enabled plugins")

        for plugin_name in enabled_anywhere:
            if plugin_name in PLUGINS_REQUIRING_CONFIG and plugin_name not in configured_plugins:
                self.r.error(
                    profile_path,
                    f"plugin '{plugin_name}' is enabled but has no pluginConfig entry",
                )

        for plugin_name, plugin_cfg in configured_plugins.items():
            cfg_path = f"{profile_path}.pluginConfig[{plugin_configs.index(plugin_cfg)}]"
            if plugin_name not in VALID_PLUGINS:
                self.r.error(cfg_path, f"unknown plugin '{plugin_name}' for descheduler v0.30.1")
                continue

            if plugin_name != "DefaultEvictor" and plugin_name not in enabled_anywhere:
                self.r.warn(
                    cfg_path,
                    f"pluginConfig for '{plugin_name}' exists but the plugin is not enabled in any extension point",
                )

            args = plugin_cfg.get("args", {}) or {}
            if not isinstance(args, dict):
                self.r.error(f"{cfg_path}.args", "args must be a YAML mapping/object")
                continue

            self._validate_plugin_args(plugin_name, args, f"{cfg_path}.args")

    def _check_plugin_extension_point(self, path: str, extension_point: str, plugin_name: str) -> None:
        if plugin_name not in VALID_PLUGINS:
            self.r.error(path, f"unknown plugin '{plugin_name}' for descheduler v0.30.1")
            return

        allowed_points = VALID_PLUGINS[plugin_name]
        if extension_point not in allowed_points:
            self.r.error(
                path,
                f"plugin '{plugin_name}' cannot be used in extension point "
                f"'{extension_point}', allowed: {sorted(allowed_points)}",
            )

    def _validate_plugin_args(self, plugin_name: str, args: dict[str, Any], path: str) -> None:
        if plugin_name == "DefaultEvictor":
            self._validate_default_evictor(args, path)
        elif plugin_name == "LowNodeUtilization":
            self._validate_low_node_utilization(args, path)
        elif plugin_name == "HighNodeUtilization":
            self._validate_high_node_utilization(args, path)
        elif plugin_name == "PodLifeTime":
            self._validate_pod_lifetime(args, path)
        elif plugin_name == "RemovePodsHavingTooManyRestarts":
            self._validate_too_many_restarts(args, path)
        elif plugin_name == "RemovePodsViolatingNodeAffinity":
            self._validate_node_affinity(args, path)
        elif plugin_name == "RemovePodsViolatingNodeTaints":
            self._validate_node_taints(args, path)
        elif plugin_name == "RemovePodsViolatingTopologySpreadConstraint":
            self._validate_topology_spread(args, path)
        elif plugin_name == "RemoveFailedPods":
            self._validate_remove_failed_pods(args, path)
        elif plugin_name == "RemoveDuplicates":
            self._validate_remove_duplicates(args, path)
        elif plugin_name == "RemovePodsViolatingInterPodAntiAffinity":
            self._validate_inter_pod_anti_affinity(args, path)

    def _validate_allowed_keys(self, args: dict[str, Any], allowed: set[str], path: str) -> None:
        unknown = sorted(set(args.keys()) - allowed)
        for key in unknown:
            self.r.error(path, f"unsupported arg '{key}' for descheduler v0.30.1")

    def _validate_namespaces(self, value: Any, path: str) -> None:
        if not isinstance(value, dict):
            self.r.error(path, "namespaces must be an object with optional include/exclude lists")
            return

        allowed_keys = {"include", "exclude"}
        unknown = sorted(set(value.keys()) - allowed_keys)
        for key in unknown:
            self.r.error(path, f"unsupported namespaces key '{key}'")

        include = value.get("include", [])
        exclude = value.get("exclude", [])

        if include is None:
            include = []
        if exclude is None:
            exclude = []

        if not isinstance(include, list) or not all(isinstance(x, str) for x in include):
            self.r.error(f"{path}.include", "include must be a list of strings")

        if not isinstance(exclude, list) or not all(isinstance(x, str) for x in exclude):
            self.r.error(f"{path}.exclude", "exclude must be a list of strings")

        if include and exclude:
            self.r.error(path, "include and exclude cannot be used together")

    def _validate_label_selector(self, value: Any, path: str) -> None:
        if not isinstance(value, dict):
            self.r.error(path, "labelSelector must be an object")

    def _validate_string_list(self, value: Any, path: str, non_empty: bool = False) -> None:
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            self.r.error(path, "must be a list of strings")
            return
        if non_empty and not value:
            self.r.error(path, "must not be empty")

    def _validate_int_list(self, value: Any, path: str) -> None:
        if not isinstance(value, list) or not all(is_int(x) for x in value):
            self.r.error(path, "must be a list of integers")

    def _validate_percentage_map(
        self,
        value: Any,
        path: str,
        required: bool,
    ) -> dict[str, float]:
        if value is None:
            if required:
                self.r.error(path, "is required")
            return {}

        if not isinstance(value, dict) or not value:
            self.r.error(path, "must be a non-empty object of resource -> percentage")
            return {}

        normalized: dict[str, float] = {}
        for resource_name, percentage in value.items():
            if not isinstance(resource_name, str):
                self.r.error(path, "resource names must be strings")
                continue
            if not is_number(percentage):
                self.r.error(f"{path}.{resource_name}", "must be a number in [0, 100]")
                continue
            numeric_percentage = float(percentage)
            if numeric_percentage < 0 or numeric_percentage > 100:
                self.r.error(f"{path}.{resource_name}", "must be in range [0, 100]")
                continue
            normalized[resource_name] = numeric_percentage

        return normalized

    def _validate_evictable_namespaces(self, value: Any, path: str) -> None:
        self._validate_namespaces(value, path)

    def _validate_default_evictor(self, args: dict[str, Any], path: str) -> None:
        self._validate_allowed_keys(args, DEFAULT_EVICTOR_ALLOWED_ARGS, path)

        if "nodeSelector" in args and not isinstance(args["nodeSelector"], str):
            self.r.error(f"{path}.nodeSelector", "must be a string")

        for bool_key in (
            "evictLocalStoragePods",
            "evictDaemonSetPods",
            "evictSystemCriticalPods",
            "ignorePvcPods",
            "evictFailedBarePods",
            "nodeFit",
        ):
            if bool_key in args and not isinstance(args[bool_key], bool):
                self.r.error(f"{path}.{bool_key}", "must be a boolean")

        if "labelSelector" in args:
            self._validate_label_selector(args["labelSelector"], f"{path}.labelSelector")

        if "minReplicas" in args:
            if not is_int(args["minReplicas"]) or args["minReplicas"] < 0:
                self.r.error(f"{path}.minReplicas", "must be a non-negative integer")

        if "priorityThreshold" in args:
            prio = args["priorityThreshold"]
            prio_path = f"{path}.priorityThreshold"
            if not isinstance(prio, dict):
                self.r.error(prio_path, "must be an object with either name or value")
            else:
                allowed_keys = {"name", "value"}
                unknown = sorted(set(prio.keys()) - allowed_keys)
                for key in unknown:
                    self.r.error(prio_path, f"unsupported priorityThreshold key '{key}'")

                has_name = isinstance(prio.get("name"), str) and bool(prio.get("name"))
                has_value = "value" in prio and is_int(prio.get("value"))

                if not has_name and not has_value:
                    self.r.error(prio_path, "must define either non-empty name or integer value")
                if has_name and has_value:
                    self.r.error(prio_path, "must define either name or value, not both")
                if "value" in prio and not is_int(prio.get("value")):
                    self.r.error(f"{prio_path}.value", "must be an integer")

        if args.get("evictSystemCriticalPods") is True and "priorityThreshold" in args:
            self.r.warn(
                path,
                "evictSystemCriticalPods=true disables priority filtering, so priorityThreshold is ineffective",
            )

    def _validate_low_node_utilization(self, args: dict[str, Any], path: str) -> None:
        self._validate_allowed_keys(args, LOW_NODE_UTILIZATION_ALLOWED_ARGS, path)

        thresholds = self._validate_percentage_map(
            args.get("thresholds"),
            f"{path}.thresholds",
            required=True,
        )
        target_thresholds = self._validate_percentage_map(
            args.get("targetThresholds"),
            f"{path}.targetThresholds",
            required=True,
        )

        if thresholds and target_thresholds:
            if set(thresholds.keys()) != set(target_thresholds.keys()):
                self.r.error(
                    path,
                    "thresholds and targetThresholds must define exactly the same resource keys",
                )

            for resource_name in sorted(set(thresholds.keys()) & set(target_thresholds.keys())):
                if thresholds[resource_name] > target_thresholds[resource_name]:
                    self.r.error(
                        path,
                        f"thresholds['{resource_name}'] must be <= targetThresholds['{resource_name}']",
                    )

        if "useDeviationThresholds" in args and not isinstance(args["useDeviationThresholds"], bool):
            self.r.error(f"{path}.useDeviationThresholds", "must be a boolean")

        if "numberOfNodes" in args:
            if not is_int(args["numberOfNodes"]) or args["numberOfNodes"] < 0:
                self.r.error(f"{path}.numberOfNodes", "must be a non-negative integer")

        if "evictableNamespaces" in args:
            self._validate_evictable_namespaces(
                args["evictableNamespaces"],
                f"{path}.evictableNamespaces",
            )

    def _validate_high_node_utilization(self, args: dict[str, Any], path: str) -> None:
        self._validate_allowed_keys(args, HIGH_NODE_UTILIZATION_ALLOWED_ARGS, path)

        self._validate_percentage_map(
            args.get("thresholds"),
            f"{path}.thresholds",
            required=True,
        )

        if "numberOfNodes" in args:
            if not is_int(args["numberOfNodes"]) or args["numberOfNodes"] < 0:
                self.r.error(f"{path}.numberOfNodes", "must be a non-negative integer")

        if "evictableNamespaces" in args:
            self._validate_evictable_namespaces(
                args["evictableNamespaces"],
                f"{path}.evictableNamespaces",
            )

    def _validate_pod_lifetime(self, args: dict[str, Any], path: str) -> None:
        self._validate_allowed_keys(args, POD_LIFETIME_ALLOWED_ARGS, path)

        if "namespaces" in args:
            self._validate_namespaces(args["namespaces"], f"{path}.namespaces")
        if "labelSelector" in args:
            self._validate_label_selector(args["labelSelector"], f"{path}.labelSelector")
        if "states" in args:
            self._validate_string_list(args["states"], f"{path}.states")

        if "maxPodLifeTimeSeconds" not in args:
            self.r.error(
                path,
                "maxPodLifeTimeSeconds is required in v0.30.1",
            )
            return

        max_age = args["maxPodLifeTimeSeconds"]
        if not is_int(max_age) or max_age < 0:
            self.r.error(f"{path}.maxPodLifeTimeSeconds", "must be a non-negative integer")
            return

        if max_age == 0:
            self.r.warn(f"{path}.maxPodLifeTimeSeconds", "0 means almost everything older than now becomes eligible")
        elif max_age < 60:
            self.r.warn(f"{path}.maxPodLifeTimeSeconds", "very low value; this is unusually aggressive")

    def _validate_too_many_restarts(self, args: dict[str, Any], path: str) -> None:
        self._validate_allowed_keys(args, TOO_MANY_RESTARTS_ALLOWED_ARGS, path)

        if "namespaces" in args:
            self._validate_namespaces(args["namespaces"], f"{path}.namespaces")
        if "labelSelector" in args:
            self._validate_label_selector(args["labelSelector"], f"{path}.labelSelector")
        if "states" in args:
            self._validate_string_list(args["states"], f"{path}.states")
        if "includingInitContainers" in args and not isinstance(args["includingInitContainers"], bool):
            self.r.error(f"{path}.includingInitContainers", "must be a boolean")

        if "podRestartThreshold" not in args:
            self.r.error(path, "podRestartThreshold is required")
            return

        threshold = args["podRestartThreshold"]
        if not is_int(threshold):
            self.r.error(f"{path}.podRestartThreshold", "must be an integer")
            return
        if threshold < 0:
            self.r.error(f"{path}.podRestartThreshold", "must be >= 0")
        elif threshold == 0:
            self.r.warn(f"{path}.podRestartThreshold", "0 is very aggressive and matches any restarted pod")

    def _validate_node_affinity(self, args: dict[str, Any], path: str) -> None:
        self._validate_allowed_keys(args, NODE_AFFINITY_ALLOWED_ARGS, path)

        if "namespaces" in args:
            self._validate_namespaces(args["namespaces"], f"{path}.namespaces")
        if "labelSelector" in args:
            self._validate_label_selector(args["labelSelector"], f"{path}.labelSelector")

        node_affinity_type = args.get("nodeAffinityType")
        if node_affinity_type is None:
            self.r.error(path, "nodeAffinityType is required")
            return

        self._validate_string_list(node_affinity_type, f"{path}.nodeAffinityType", non_empty=True)
        if isinstance(node_affinity_type, list):
            for item in node_affinity_type:
                if item not in VALID_NODE_AFFINITY_TYPES:
                    self.r.error(
                        f"{path}.nodeAffinityType",
                        f"unsupported value '{item}', allowed: {sorted(VALID_NODE_AFFINITY_TYPES)}",
                    )

    def _validate_node_taints(self, args: dict[str, Any], path: str) -> None:
        self._validate_allowed_keys(args, NODE_TAINTS_ALLOWED_ARGS, path)

        if "namespaces" in args:
            self._validate_namespaces(args["namespaces"], f"{path}.namespaces")
        if "labelSelector" in args:
            self._validate_label_selector(args["labelSelector"], f"{path}.labelSelector")
        if "includePreferNoSchedule" in args and not isinstance(args["includePreferNoSchedule"], bool):
            self.r.error(f"{path}.includePreferNoSchedule", "must be a boolean")
        if "excludedTaints" in args:
            self._validate_string_list(args["excludedTaints"], f"{path}.excludedTaints")
        if "includedTaints" in args:
            self._validate_string_list(args["includedTaints"], f"{path}.includedTaints")

    def _validate_topology_spread(self, args: dict[str, Any], path: str) -> None:
        self._validate_allowed_keys(args, TOPOLOGY_SPREAD_ALLOWED_ARGS, path)

        if "namespaces" in args:
            self._validate_namespaces(args["namespaces"], f"{path}.namespaces")
        if "labelSelector" in args:
            self._validate_label_selector(args["labelSelector"], f"{path}.labelSelector")
        if "topologyBalanceNodeFit" in args and not isinstance(args["topologyBalanceNodeFit"], bool):
            self.r.error(f"{path}.topologyBalanceNodeFit", "must be a boolean")

        if "constraints" in args:
            self._validate_string_list(args["constraints"], f"{path}.constraints", non_empty=True)
            constraints = args["constraints"]
            if isinstance(constraints, list):
                for item in constraints:
                    if item not in VALID_TOPOLOGY_CONSTRAINTS:
                        self.r.error(
                            f"{path}.constraints",
                            f"unsupported value '{item}', allowed: {sorted(VALID_TOPOLOGY_CONSTRAINTS)}",
                        )

    def _validate_remove_failed_pods(self, args: dict[str, Any], path: str) -> None:
        self._validate_allowed_keys(args, REMOVE_FAILED_PODS_ALLOWED_ARGS, path)

        if "namespaces" in args:
            self._validate_namespaces(args["namespaces"], f"{path}.namespaces")
        if "labelSelector" in args:
            self._validate_label_selector(args["labelSelector"], f"{path}.labelSelector")
        if "excludeOwnerKinds" in args:
            self._validate_string_list(args["excludeOwnerKinds"], f"{path}.excludeOwnerKinds")
        if "reasons" in args:
            self._validate_string_list(args["reasons"], f"{path}.reasons")
        if "exitCodes" in args:
            self._validate_int_list(args["exitCodes"], f"{path}.exitCodes")
        if "includingInitContainers" in args and not isinstance(args["includingInitContainers"], bool):
            self.r.error(f"{path}.includingInitContainers", "must be a boolean")

        if "minPodLifetimeSeconds" in args:
            value = args["minPodLifetimeSeconds"]
            if not is_int(value) or value < 0:
                self.r.error(f"{path}.minPodLifetimeSeconds", "must be a non-negative integer")

    def _validate_remove_duplicates(self, args: dict[str, Any], path: str) -> None:
        self._validate_allowed_keys(args, REMOVE_DUPLICATES_ALLOWED_ARGS, path)

        if "namespaces" in args:
            self._validate_namespaces(args["namespaces"], f"{path}.namespaces")
        if "excludeOwnerKinds" in args:
            self._validate_string_list(args["excludeOwnerKinds"], f"{path}.excludeOwnerKinds")

    def _validate_inter_pod_anti_affinity(self, args: dict[str, Any], path: str) -> None:
        self._validate_allowed_keys(args, INTER_POD_ANTI_AFFINITY_ALLOWED_ARGS, path)

        if "namespaces" in args:
            self._validate_namespaces(args["namespaces"], f"{path}.namespaces")
        if "labelSelector" in args:
            self._validate_label_selector(args["labelSelector"], f"{path}.labelSelector")


class DeschedulerPolicyLinter:
    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self.reporter = LintReporter()

    def validate_schema(self) -> None:
        validator = Draft7Validator(DESCHEDULER_SCHEMA)
        errors = sorted(validator.iter_errors(self.policy), key=lambda e: list(e.path))
        for error in errors:
            self.reporter.error(format_path(error.path), error.message)

    def validate_semantics(self) -> None:
        if not isinstance(self.policy, dict):
            return

        profiles = self.policy.get("profiles")
        if not isinstance(profiles, list):
            return

        engine = PluginRuleEngine(self.reporter)
        engine.run([p for p in profiles if isinstance(p, dict)])

    def lint(self) -> tuple[list[str], list[str]]:
        self.validate_schema()
        self.validate_semantics()
        return self.reporter.errors, self.reporter.warnings


def load_policy(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(2)
    except yaml.YAMLError as exc:
        print(f"YAML parse error: {exc}", file=sys.stderr)
        sys.exit(2)

    if data is None:
        print("YAML file is empty", file=sys.stderr)
        sys.exit(2)

    return data


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: descheduler_policy_linter.py <policy.yaml>")
        sys.exit(1)

    policy_path = Path(sys.argv[1])
    policy = load_policy(policy_path)

    linter = DeschedulerPolicyLinter(policy)
    errors, warnings = linter.lint()

    print("\nDescheduler Policy Lint Report (target: v0.30.1)\n")

    if errors:
        print("Errors:")
        for item in errors:
            print(f" - {item}")

    if warnings:
        print("\nWarnings:")
        for item in warnings:
            print(f" - {item}")

    if not errors and not warnings:
        print("Policy passed all checks 🎉")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
