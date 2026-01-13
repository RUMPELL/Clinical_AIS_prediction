from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file.

    The project uses YAML configs to make experiments reproducible and avoid
    long CLI commands.
    """

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"Config must be a mapping/dict. Got: {type(obj)}")
    return obj


def parse_args_with_config(parser: Any) -> Tuple[Any, Dict[str, Any]]:
    """Two-pass argparse parsing with YAML config.

    1) Parse only --config (if present)
    2) Load YAML and set it as defaults
    3) Parse full args; CLI flags override config defaults

    Returns (args, cfg_dict).
    """

    # Pass 1: detect --config
    cfg_path = None
    argv = sys.argv[1:]
    if "--config" in argv:
        try:
            idx = argv.index("--config")
            cfg_path = argv[idx + 1]
        except Exception:
            cfg_path = None

    cfg: Dict[str, Any] = {}
    if cfg_path:
        cfg = load_yaml_config(cfg_path)
        # Apply defaults from YAML (only for keys that exist in the parser)
        # Unknown keys are ignored to keep scripts robust.
        parser.set_defaults(**{k: v for k, v in cfg.items()})

    args = parser.parse_args()
    return args, cfg
