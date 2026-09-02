"""Print public metadata and verify that a checkpoint matches SARC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sarc import load_sarc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    model, checkpoint = load_sarc(args.checkpoint)
    print(json.dumps(checkpoint.get("manifest", {}), indent=2))
    print(f"parameter_count: {model.parameter_count()}")


if __name__ == "__main__":
    main()

