"""This file loads YAML configs into an argparse-compatible namespace without changing runtime logic."""

from argparse import Namespace
from copy import deepcopy

import yaml


def _to_namespace(value):
    """Recursively convert dictionaries into namespaces for attribute access."""
    if isinstance(value, dict):
        return Namespace(**{key: _to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def load_config(config_path):
    """Load a YAML config file and return an argparse-style namespace."""
    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return _to_namespace(data)


def namespace_to_dict(namespace):
    """Convert nested namespaces back into plain dictionaries for logging."""
    if isinstance(namespace, Namespace):
        return {key: namespace_to_dict(value) for key, value in vars(namespace).items()}
    if isinstance(namespace, list):
        return [namespace_to_dict(item) for item in namespace]
    return deepcopy(namespace)

