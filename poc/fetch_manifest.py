#!/usr/bin/env python3
"""Fetch real tournament data from Gradients API and write poc/manifest.json."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poc.manifest import refresh_manifest


async def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh POC manifest from Gradients API")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "manifest.json",
        help="Output manifest path",
    )
    args = parser.parse_args()

    manifest = await refresh_manifest(args.output)
    print(f"Wrote manifest for tournament {manifest.tournament_id}")
    print(f"  Boss battery tasks: {len(manifest.boss_battery)}")
    print(f"  Regression tasks:   {len(manifest.regression_suite)}")
    print(f"  Smoke task:         {manifest.smoke['task_id']}")
    print(f"  Path:               {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
