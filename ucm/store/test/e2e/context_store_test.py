"""CPU simu integration tests. Set UCM_CONTEXT_BUILD to the CMake build directory."""

import ast
import copy
import dataclasses
import importlib.util
import multiprocessing
import os
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


def native():
    build = os.environ.get("UCM_CONTEXT_BUILD")
    if not build:
        pytest.skip("UCM_CONTEXT_BUILD is required (simu runtime)")
    build = Path(build)
    extension = next((build / "ucm/store/pipeline").glob("ucmpipelinestore*.so"))
    spec = importlib.util.spec_from_file_location("ucmpipelinestore", extension)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, build / "ucm/store/context/libcontextstore.so"


def ids(*numbers):
    return np.frombuffer(
        b"".join(n.to_bytes(16, "little") for n in numbers), dtype=np.uint8
    )


def test_native_pipeline_on_evict_and_watcher():
    module, library = native()
    config = {
        "unique_id": "python_" + uuid.uuid4().hex,
        "device_id": 0,
        "block_size": 64,
        "shard_size": 64,
        "tensor_size": 64,
        "context_memory_capacity_bytes": 64,
        "context_simulated_ssd_capacity_bytes": 128,
        "context_max_eviction_blocks": 1,
        "cache_stream_number": 2,
    }
    watcher = module.PipelineStore()
    watcher.Stack("Context", str(library), {"unique_id": config["unique_id"]})
    assert watcher.Lookup(ids(1)) == b"\0"
    worker = module.PipelineStore()
    worker.Stack("Context", str(library), config)
    index = np.array([0], dtype=np.uint64)
    source = np.arange(64, dtype=np.uint8)
    address = np.array([[source.ctypes.data]], dtype=np.uint64)
    for n in (1, 2):
        worker.ObserveRequest(str(n), n, n, ids(n))
        task = worker.Dump(ids(n), index, address, 0)
        worker.Wait(task)
        assert watcher.Lookup(ids(n)) == b"\1"
        assert worker.ContextStats().get("ssd_write_blocks", 0) == n - 1
    target = np.zeros_like(source)
    task = worker.Load(ids(1), index, np.array([[target.ctypes.data]], dtype=np.uint64))
    worker.Wait(task)
    np.testing.assert_array_equal(source, target)
    assert worker.ContextStats()["ssd_read_bytes"] == 64
    # Retiring a context does not remove its stored payload.
    worker.ObserveRequest("1", 0, 0, ids())
    assert watcher.LookupOnPrefix(ids(1, 2, 3)) == 1
    del worker
    assert watcher.Lookup(ids(1, 2)) == b"\0\0"


def connector_functions():
    # Execute the actual connector methods with engine objects replaced by small
    # fixtures. No vLLM installation or accelerator is needed for metadata tests.
    tree = ast.parse(
        (ROOT / "ucm/integration/vllm/ucm_connector.py").read_text(encoding="utf-8-sig")
    )
    selected = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in {
            "RequestMeta",
            "RequestDispatchMeta",
        }:
            selected.append(node)
        if isinstance(node, ast.ClassDef) and node.name == "UCMDirectConnector":
            methods = [
                n
                for n in node.body
                if isinstance(n, ast.FunctionDef)
                and n.name
                in {
                    "_generate_dispatch_meta",
                    "_observe_context_requests",
                    "_store_block_ids",
                    "_create_store",
                }
            ]
            selected.append(
                ast.ClassDef(
                    name="Connector",
                    bases=[],
                    keywords=[],
                    body=methods,
                    decorator_list=[],
                )
            )
    namespace = {
        "dataclass": dataclasses.dataclass,
        "field": dataclasses.field,
        "time": time,
        "copy": copy,
        "Optional": Optional,
        "KVCacheLayout": object,
        "UcmKVStoreBaseV1": object,
        "KVConnectorRole": SimpleNamespace(WORKER="worker", SCHEDULER="scheduler"),
        "UcmConnectorFactoryV1": SimpleNamespace(
            create_connector=lambda name, config, path: config
        ),
        "logger": SimpleNamespace(info=lambda *args: None),
        "_get_store_io_sizes": lambda shard, block: (shard, block),
        "_get_store_gc_block_size": lambda pipeline, tensors, shard, block: block,
    }
    code = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(code, "connector_context_methods", "exec"), namespace)  # noqa: S102
    return namespace


def test_connector_transmits_full_prefix_even_for_suffix_io():
    ns = connector_functions()
    connector = ns["Connector"]()
    connector._context_store_enabled = True
    connector.cp_world_size = 1
    connector.block_size = 16
    blocks = [bytes([n]) * 16 for n in range(4)]
    request = ns["RequestMeta"](
        ucm_block_ids=blocks,
        hbm_hit_block_num=1,
        total_hit_block_num=2,
        num_token_ids=64,
        token_processed=32,
    )
    meta = connector._generate_dispatch_meta(request, 32, [10, 11, 12, 13])
    assert meta.load_block_ids[0] == blocks[1:2]
    assert meta.dump_block_ids[0] == blocks[2:]
    assert meta.context_block_ids == blocks
    calls = []
    connector.connector_configs = [
        {"ucm_connector_config": {"store_pipeline": "ContextStore"}}
    ]
    connector.store = SimpleNamespace(observe_request=lambda *args: calls.append(args))
    connector._observe_context_requests(SimpleNamespace(request_meta={"req": meta}))
    assert len(calls) == 1
    assert calls[0][0] == "req"
    assert calls[0][1] == meta.context_observation
    assert calls[0][3] == blocks
    connector._context_store_enabled = False
    connector._observe_context_requests(SimpleNamespace(request_meta={"req": meta}))
    assert len(calls) == 1


def rank_process(connection, config):
    module, library = native()
    worker = module.PipelineStore()
    worker.Stack("Context", str(library), config)
    connection.send("ready")
    try:
        while True:
            command, key, shard = connection.recv()
            if command == "close":
                break
            if command == "dump":
                worker.ObserveRequest(str(key), key, key * 100, ids(key))
                data = np.full(64, 30 * config["context_tp_rank"] + key, dtype=np.uint8)
                task = worker.Dump(
                    ids(key),
                    np.array([shard], dtype=np.uint64),
                    np.array([[data.ctypes.data]], dtype=np.uint64),
                    0,
                )
                worker.Wait(task)
                connection.send(worker.ContextStats())
            elif command == "load":
                data = np.zeros(64, dtype=np.uint8)
                task = worker.Load(
                    ids(key),
                    np.array([shard], dtype=np.uint64),
                    np.array([[data.ctypes.data]], dtype=np.uint64),
                )
                worker.Wait(task)
                connection.send(data.tobytes())
    finally:
        del worker
        connection.close()


def test_tp_processes_intersect_readiness_and_independent_evictions():
    module, library = native()
    name = "tp_" + uuid.uuid4().hex
    watcher = module.PipelineStore()
    watcher.Stack("Context", str(library), {"unique_id": name, "context_tp_size": 2})
    assert watcher.Lookup(ids(1, 2)) == b"\0\0"
    ctx = multiprocessing.get_context("spawn")
    processes, connections = [], []

    def call(rank, command, key=1, shard=0):
        pipe = connections[rank]
        pipe.send((command, key, shard))
        assert pipe.poll(30), "TP worker did not complete"
        return pipe.recv()

    try:
        for rank in range(2):
            parent, child = ctx.Pipe()
            process = ctx.Process(
                target=rank_process,
                args=(
                    child,
                    {
                        "unique_id": name,
                        "context_tp_size": 2,
                        "context_tp_rank": rank,
                        "device_id": rank,
                        "block_size": 128,
                        "shard_size": 64,
                        "tensor_size": 64,
                        "context_memory_capacity_bytes": 128,
                        "context_simulated_ssd_capacity_bytes": 1024,
                        "context_max_eviction_blocks": 1,
                        "context_retention_ns": -1 if rank == 0 else 0,
                    },
                ),
            )
            process.start()
            child.close()
            processes.append(process)
            connections.append(parent)
            assert parent.poll(30)
            assert parent.recv() == "ready"
        call(0, "dump", 1, 0)
        call(0, "dump", 1, 1)
        assert watcher.Lookup(ids(1)) == b"\0"
        call(1, "dump", 1, 0)
        assert watcher.Lookup(ids(1)) == b"\0"  # rank 1 is only partially ready
        call(1, "dump", 1, 1)
        assert watcher.Lookup(ids(1)) == b"\1"
        call(0, "dump", 2, 0)
        stats = call(0, "dump", 2, 1)
        assert stats["ssd_write_blocks"] == 1
        assert watcher.LookupOnPrefix(ids(1, 2)) == 0
        # Rank 0 reads SSD, rank 1 reads Memory, each with distinct actual bytes.
        assert call(0, "load") == bytes([1]) * 64
        assert call(1, "load") == bytes([31]) * 64
        call(1, "dump", 2, 0)
        stats = call(1, "dump", 2, 1)
        assert stats.get("ssd_write_blocks", 0) == 0
        assert watcher.Lookup(ids(1, 2)) == b"\0\1"
        assert watcher.LookupOnPrefix(ids(1, 2)) == -1
        assert watcher.LookupOnReverse(ids(1, 2)) == 1
        connections[1].send(("close", 0, 0))
        processes[1].join(30)
        assert processes[1].exitcode == 0
        assert watcher.Lookup(ids(2)) == b"\0"
    finally:
        for process, pipe in zip(processes, connections):
            if process.is_alive():
                pipe.send(("close", 0, 0))
                process.join(5)
            if process.is_alive():
                process.terminate()
                process.join()
            pipe.close()


@pytest.mark.parametrize("rank", [0, 1, 3])
def test_context_keys_match_io_on_every_rank(rank):
    connector = connector_functions()["Connector"]()
    connector._context_store_enabled = True
    connector.tp_rank = rank
    connector.tp_size = 4
    connector.is_mla = False
    connector.request_hasher = lambda key: bytes(x ^ 255 for x in key)
    blocks = [bytes([1]) * 16, bytes([2]) * 16]
    calls = []
    connector.store = SimpleNamespace(observe_request=lambda *args: calls.append(args))
    connector._observe_context_requests(
        SimpleNamespace(
            request_meta={
                "request": SimpleNamespace(
                    context_observation=1, context_block_ids=blocks
                )
            }
        )
    )
    assert connector._store_block_ids(blocks) == calls[0][3] == blocks
    # Existing stores retain their nonzero-rank hashing rule.
    connector._context_store_enabled = False
    expected = blocks if rank == 0 else [connector.request_hasher(k) for k in blocks]
    assert connector._store_block_ids(blocks) == expected


@pytest.mark.parametrize("role", ["scheduler", "worker"])
def test_connector_supplies_tp_config_and_rejects_unsupported_modes(role):
    connector = connector_functions()["Connector"]()
    connector.connector_configs = [
        {
            "ucm_connector_name": "UcmPipelineStore",
            "ucm_connector_config": {"store_pipeline": "ContextStore"},
        }
    ]
    parallel = SimpleNamespace(pipeline_parallel_size=1)
    connector._vllm_config = SimpleNamespace(parallel_config=parallel)
    connector.tp_size, connector.tp_rank = 4, 3
    connector.cp_world_size = 1
    connector.is_mla = False
    connector._role = role
    connector.unique_id = "experiment"
    connector._dp_rank = 2
    connector._gc_owner = False
    connector.device_id = 3
    connector.blocks_per_chunk = 1
    connector._set_default_shm_buffer_capacity = lambda config: None
    connector._publish_block_size = lambda size: None
    layout = SimpleNamespace(
        tensor_size_list=[64],
        shard_size=64,
        block_size=128,
        base_ptrs=np.array([4096]),
        buffer_sizes=np.array([128]),
    )
    config = connector._create_store(layout if role == "worker" else None)
    assert config["unique_id"] == "experiment_dp2"
    assert config["context_tp_size"] == 4
    assert config["context_tp_rank"] == (3 if role == "worker" else 0)
    assert config["share_buffer_enable"] is False
    connector.is_mla = True
    with pytest.raises(ValueError, match="non-MLA"):
        connector._create_store(layout)
    connector.is_mla = False
    parallel.pipeline_parallel_size = 2
    with pytest.raises(ValueError, match="PP=CP=1"):
        connector._create_store(layout)
    parallel.pipeline_parallel_size = 1
    connector.cp_world_size = 2
    with pytest.raises(ValueError, match="PP=CP=1"):
        connector._create_store(layout)
