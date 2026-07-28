"""Memory-bounded append of JSONL embedding checkpoints."""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from collections.abc import Callable

import numpy as np

from .extend import extend_corpus


def checkpoint_done_ids(path: str | Path) -> set[str]:
    """Read only checkpoint IDs, never the large embedding payloads."""
    done: set[str] = set()
    path = Path(path)
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "ok" and row.get("id") is not None:
                done.add(row["id"])
    return done


def _item(row: dict) -> dict:
    return {
        "meta": row["meta"],
        "raw_vector": np.asarray(row["embedding"], dtype=np.float32),
        "merit_backbone": (
            np.asarray(row["merit_backbone"], dtype=np.float32)
            if row.get("merit_backbone") is not None else None
        ),
    }


def _append_batch(
    current: Path, out_dir: Path, batch: list[dict],
    dedupe_threshold: float, knn_k: int,
) -> tuple[Path, dict]:
    """Append one batch, returning the bundle the *next* batch should read.

    A batch whose rows are all dropped as duplicates makes ``extend_corpus``
    return early without persisting anything, so ``out_dir`` may still hold no
    bundle. Keep reading from ``current`` until a write has actually landed --
    otherwise the next batch loads an empty directory.
    """
    report = extend_corpus(
        current, batch, out_dir=out_dir,
        dedupe_threshold=dedupe_threshold, knn_k=knn_k,
    )
    batch.clear()
    gc.collect()
    if (out_dir / "manifest.json").exists():
        return out_dir, report
    return current, report


def _write_state(path: Path, checkpoint: Path, next_line: int) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({
        "checkpoint": str(checkpoint.resolve()),
        "next_line": next_line,
    }))
    os.replace(tmp, path)


def append_checkpoint_in_batches(
    bundle_dir: str | Path,
    checkpoint: str | Path,
    out_dir: str | Path,
    *,
    batch_size: int = 500,
    dedupe_threshold: float = 0.98,
    knn_k: int = 15,
) -> dict:
    """Append a JSONL checkpoint without materializing it in memory.

    The progress file is written only after a batch has been saved. If the
    process is killed after saving but before the state update, that batch is
    safely replayed and deduped on restart.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    checkpoint = Path(checkpoint)
    current = Path(bundle_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / ".checkpoint_append_state.json"
    start_line = 0
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
            if state.get("checkpoint") == str(checkpoint.resolve()):
                start_line = int(state.get("next_line", 0))
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    n_rows = n_added = n_dropped = 0
    batch: list[dict] = []
    with checkpoint.open() as f:
        for line_no, line in enumerate(f):
            if line_no < start_line or not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") != "ok":
                continue
            batch.append(_item(row))
            n_rows += 1
            if len(batch) < batch_size:
                continue

            current, report = _append_batch(
                current, out_dir, batch, dedupe_threshold, knn_k
            )
            n_added += report.get("n_added", 0)
            n_dropped += report.get("n_dropped_duplicate", 0)
            _write_state(state_path, checkpoint, line_no + 1)
            print(
                f"checkpoint line {line_no + 1}: batch={len(batch)} "
                f"added={report.get('n_added', 0)} "
                f"dropped={report.get('n_dropped_duplicate', 0)}",
                flush=True,
            )

    if batch:
        current, report = _append_batch(
            current, out_dir, batch, dedupe_threshold, knn_k
        )
        n_added += report.get("n_added", 0)
        n_dropped += report.get("n_dropped_duplicate", 0)
        _write_state(state_path, checkpoint, line_no + 1)
        print(
            f"checkpoint line {line_no + 1}: batch={len(batch)} "
            f"added={report.get('n_added', 0)} "
            f"dropped={report.get('n_dropped_duplicate', 0)}",
            flush=True,
        )

    return {
        "checkpoint_rows_processed": n_rows,
        "n_added": n_added,
        "n_dropped_duplicate": n_dropped,
        "out_dir": str(out_dir),
        "bundle_written": current == out_dir,
    }


def append_jsonl_in_batches(
    bundle_dir: str | Path,
    jsonl_path: str | Path,
    out_dir: str | Path,
    *,
    row_to_item: Callable[[dict], dict | None],
    batch_size: int = 500,
    dedupe_threshold: float = 0.98,
    knn_k: int = 15,
    state_name: str = ".jsonl_append_state.json",
) -> dict:
    """Generic streaming JSONL appender for source-specific checkpoints."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    jsonl_path = Path(jsonl_path)
    current = Path(bundle_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / state_name
    start_line = 0
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
            if state.get("checkpoint") == str(jsonl_path.resolve()):
                start_line = int(state.get("next_line", 0))
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    n_rows = n_added = n_dropped = 0
    batch: list[dict] = []
    last_line = start_line
    with jsonl_path.open() as f:
        for line_no, line in enumerate(f):
            last_line = line_no + 1
            if line_no < start_line or not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            item = row_to_item(row)
            if item is None:
                continue
            batch.append(item)
            n_rows += 1
            if len(batch) < batch_size:
                continue
            current, report = _append_batch(
                current, out_dir, batch, dedupe_threshold, knn_k
            )
            n_added += report.get("n_added", 0)
            n_dropped += report.get("n_dropped_duplicate", 0)
            _write_state(state_path, jsonl_path, line_no + 1)
            print(
                f"checkpoint line {line_no + 1}: "
                f"added={report.get('n_added', 0)} "
                f"dropped={report.get('n_dropped_duplicate', 0)}",
                flush=True,
            )

    if batch:
        current, report = _append_batch(
            current, out_dir, batch, dedupe_threshold, knn_k
        )
        n_added += report.get("n_added", 0)
        n_dropped += report.get("n_dropped_duplicate", 0)
        _write_state(state_path, jsonl_path, last_line)
        print(
            f"checkpoint line {last_line}: "
            f"added={report.get('n_added', 0)} "
            f"dropped={report.get('n_dropped_duplicate', 0)}",
            flush=True,
        )

    return {
        "checkpoint_rows_processed": n_rows,
        "n_added": n_added,
        "n_dropped_duplicate": n_dropped,
        "out_dir": str(out_dir),
        "bundle_written": current == out_dir,
    }
