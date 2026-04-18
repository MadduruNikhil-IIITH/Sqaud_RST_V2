from __future__ import annotations

import importlib
import json
import platform
import sys
from pathlib import Path


def import_status(module_name: str) -> dict[str, str]:
    try:
        importlib.import_module(module_name)
        return {"module": module_name, "status": "ok"}
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"module": module_name, "status": "fail", "error": str(exc)}


def main() -> None:
    report: dict[str, object] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "imports": [],
    }

    modules = [
        "torch",
        "transformers",
        "pandas",
        "sklearn",
        "spacy",
        "stanza",
        "isanlp_rst",
    ]
    report["imports"] = [import_status(m) for m in modules]

    cuda_available = False
    cuda_device_count = 0
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_device_count = int(torch.cuda.device_count())
    except Exception:
        pass

    report["cuda_available"] = cuda_available
    report["cuda_device_count"] = cuda_device_count
    report["isanlp_rst_gate"] = any(
        item.get("module") == "isanlp_rst" and item.get("status") == "ok" for item in report["imports"]
    )

    out = Path("data/interim/env_gate_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    if not report["isanlp_rst_gate"]:
        raise SystemExit("isanlp_rst gate failed. Switch to py311 or py310 environment.")


if __name__ == "__main__":
    main()
