#!/usr/bin/env python3
"""Validate LLaVA-ReCap-CC3M parquet shards and write a training manifest."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ImportError as exc:
    raise SystemExit("pyarrow required: pip install pyarrow") from exc


def validate_shard(path: Path) -> dict:
    if path.stat().st_size == 0:
        raise ValueError("empty file")
    pf = pq.ParquetFile(path)
    cols = {f.name for f in pf.schema_arrow}
    required = {"id", "image", "conversations"}
    missing = required - cols
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "num_rows": int(pf.metadata.num_rows),
        "num_row_groups": int(pf.metadata.num_row_groups),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        default="playground/Datasets/LLaVA-ReCap-CC3M/data",
        help="Directory containing train-*-of-*.parquet shards",
    )
    ap.add_argument(
        "--out",
        default="logs/setup/cc3m_manifest.json",
        help="Manifest path (writable; dataset dir on ftp may be read-only)",
    )
    ap.add_argument(
        "--ready-marker",
        default="logs/setup/.cc3m_ready",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"data dir missing: {data_dir}")

    shards = sorted(data_dir.glob("train-*-of-*.parquet"))
    if not shards:
        raise SystemExit(f"no parquet shards under {data_dir}")

    # Parse expected count from filename pattern train-00000-of-00281.parquet
    expected = None
    for shard in shards:
        parts = shard.stem.split("-")
        if len(parts) >= 4 and parts[-2] == "of":
            expected = int(parts[-1]) + 0  # of-00281 means indices 0..280 => 281 files
            break

    indices = []
    for shard in shards:
        parts = shard.stem.split("-")
        if len(parts) >= 4 and parts[1].isdigit():
            indices.append(int(parts[1]))
    missing = []
    if expected is not None:
        missing = [i for i in range(expected) if i not in set(indices)]

    t0 = time.time()
    entries = []
    errors = []
    total_rows = 0
    for shard in shards:
        try:
            info = validate_shard(shard)
            entries.append(info)
            total_rows += info["num_rows"]
        except Exception as exc:
            errors.append({"file": shard.name, "error": str(exc)})

    manifest = {
        "dataset": "lmms-lab/LLaVA-ReCap-CC3M",
        "data_dir": str(data_dir.resolve()),
        "num_shards": len(shards),
        "expected_shards": expected,
        "missing_indices": missing,
        "total_rows": total_rows,
        "total_bytes": sum(e["bytes"] for e in entries),
        "validated_shards": len(entries),
        "errors": errors,
        "ready_for_training": len(errors) == 0 and not missing,
        "vlm_data_path": str(data_dir.resolve()),
        "elapsed_sec": round(time.time() - t0, 1),
        "shards": entries,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))

    ready_path = Path(args.ready_marker)
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest["ready_for_training"]:
        ready_path.write_text(json.dumps({"total_rows": total_rows, "num_shards": len(shards)}, indent=2))
    elif ready_path.exists():
        ready_path.unlink()

    print(
        f"[CC3M] shards={len(shards)}/{expected} rows={total_rows} "
        f"bytes={manifest['total_bytes']/2**30:.2f}GiB errors={len(errors)} missing={len(missing)} "
        f"ready={manifest['ready_for_training']}"
    )
    print(f"[CC3M] manifest -> {out_path}")
    if errors:
        for e in errors[:5]:
            print(f"  ERROR {e['file']}: {e['error']}", file=sys.stderr)
        raise SystemExit(1)
    if missing:
        print(f"  missing indices (first 10): {missing[:10]}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
