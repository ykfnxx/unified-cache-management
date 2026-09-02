#!/usr/bin/env python3
#
# MIT License
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


METRICS = (
    "on_evict_eviction_paths_total",
    "on_evict_evicted_blocks_total",
    "on_evict_backend_write_requests_total",
    "on_evict_backend_write_bytes_total",
    "on_evict_discarded_blocks_total",
    "on_evict_discarded_bytes_total",
)


def block_id(value: int | str) -> bytes:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.blake2b(encoded, digest_size=16).digest()


def replay(args: argparse.Namespace) -> dict:
    from ucm.shared.metrics import ucmmetrics
    from ucm.store.pipeline import ucmpipelinestore

    ucmmetrics.set_up()
    for name in METRICS:
        ucmmetrics.create_stats(name, "counter")
    ucmmetrics.get_all_stats_and_clear()

    config = {
        "block_size": args.block_size_bytes,
        "on_evict_cache_capacity_gb": args.capacity_gb,
        "on_evict_cache_policy": "radix_lru",
        "on_evict_cache_dump_max_idle_s": args.dump_max_idle_s,
        "store_health": {"enabled": False},
    }
    store_dir = Path(ucmpipelinestore.__file__).resolve().parents[1]
    pipeline = ucmpipelinestore.PipelineStore()
    pipeline.Stack(
        "Fake",
        str(store_dir / "fake/libfakestore.so"),
        config | {"fake_on_evict_mode": True},
    )
    pipeline.Stack(
        "OnEvictCache",
        str(store_dir / "on_evict_cache/libonevictcachestore.so"),
        config,
    )

    unit_scale = {"s": 1_000_000_000, "ms": 1_000_000}[args.timestamp_unit]
    requests = 0
    empty_requests = 0
    block_occurrences = 0
    prefix_hits = 0
    admitted_blocks = 0
    first_timestamp = None
    previous_timestamp = None
    start = time.perf_counter()

    with args.trace_path.open(encoding="utf-8") as trace:
        for line_number, line in enumerate(trace, 1):
            record = json.loads(line)
            timestamp = record["timestamp"]
            hash_ids = record["hash_ids"]
            if timestamp < 0 or (
                previous_timestamp is not None and timestamp < previous_timestamp
            ):
                raise ValueError(f"invalid timestamp at line {line_number}")
            if any(type(value) not in (int, str) for value in hash_ids):
                raise ValueError(f"invalid hash_ids at line {line_number}")

            requests += 1
            block_occurrences += len(hash_ids)
            if first_timestamp is None:
                first_timestamp = timestamp
            previous_timestamp = timestamp
            if not hash_ids:
                empty_requests += 1
                continue

            logical_time_ns = int((timestamp - first_timestamp) * unit_scale)
            hits, admitted = pipeline.ReplayRequest(
                [block_id(value) for value in hash_ids], logical_time_ns
            )
            prefix_hits += hits
            admitted_blocks += admitted

    elapsed = time.perf_counter() - start
    counters = ucmmetrics.get_all_stats_and_clear()[0]
    native_metrics = {name: counters.get(name, 0.0) for name in METRICS}
    eviction_paths = native_metrics["on_evict_eviction_paths_total"]
    evicted_blocks = native_metrics["on_evict_evicted_blocks_total"]
    discarded_blocks = native_metrics["on_evict_discarded_blocks_total"]

    return {
        "trace_path": str(args.trace_path),
        "timestamp_unit": args.timestamp_unit,
        "capacity_gb": args.capacity_gb,
        "capacity_blocks": (args.capacity_gb << 30) // args.block_size_bytes,
        "block_size_bytes": args.block_size_bytes,
        "dump_max_idle_s": args.dump_max_idle_s,
        "block_id_mapping": "blake2b-128-json-v1",
        "requests_total": requests,
        "empty_requests_total": empty_requests,
        "block_occurrences_total": block_occurrences,
        "prefix_hit_blocks_total": prefix_hits,
        "admitted_blocks_total": admitted_blocks,
        "prefix_block_hit_ratio": (
            prefix_hits / block_occurrences if block_occurrences else 0.0
        ),
        "replay_wall_time_s": elapsed,
        "replay_requests_per_second": requests / elapsed if elapsed else 0.0,
        **native_metrics,
        "mean_blocks_per_eviction_path": (
            evicted_blocks / eviction_paths if eviction_paths else 0.0
        ),
        "discard_ratio": discarded_blocks / evicted_blocks if evicted_blocks else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay hash-id traces against OnEvictCache")
    parser.add_argument("--trace-path", type=Path, required=True)
    parser.add_argument("--timestamp-unit", choices=("s", "ms"), required=True)
    parser.add_argument("--capacity-gb", type=int, required=True)
    parser.add_argument("--block-size-bytes", type=int, required=True)
    parser.add_argument("--dump-max-idle-s", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = replay(args)
    print(
        f"requests={report['requests_total']} "
        f"blocks={report['block_occurrences_total']} "
        f"prefix_hit_ratio={report['prefix_block_hit_ratio']:.6f}"
    )
    print(
        f"eviction_paths={report['on_evict_eviction_paths_total']:.0f} "
        f"evicted_blocks={report['on_evict_evicted_blocks_total']:.0f} "
        f"backend_writes={report['on_evict_backend_write_requests_total']:.0f} "
        f"discarded_blocks={report['on_evict_discarded_blocks_total']:.0f}"
    )
    print(
        f"wall_time_s={report['replay_wall_time_s']:.6f} "
        f"requests_per_second={report['replay_requests_per_second']:.2f}"
    )
    if args.output:
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
