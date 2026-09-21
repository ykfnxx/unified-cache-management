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
                    "_consistency_manager_enabled",
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
@pytest.mark.parametrize("is_mla", [False, True])
def test_context_keys_match_io_on_every_rank(rank, is_mla):
    connector = connector_functions()["Connector"]()
    connector._context_store_enabled = True
    connector.tp_rank = rank
    connector.tp_size = 4
    connector.is_mla = is_mla
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
    expected = (
        blocks if rank == 0 or is_mla else [connector.request_hasher(k) for k in blocks]
    )
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
    connector._context_store_enabled = True
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
    mla_config = connector._create_store(layout)
    assert mla_config["share_buffer_enable"] is True
    if role == "worker":
        assert mla_config["local_rank_size"] == 4
    connector.is_mla = False
    parallel.pipeline_parallel_size = 2
    with pytest.raises(ValueError, match="PP=CP=1"):
        connector._create_store(layout)
    parallel.pipeline_parallel_size = 1
    for field in ("prefill_context_parallel_size", "decode_context_parallel_size"):
        setattr(parallel, field, 2)
        with pytest.raises(ValueError, match="PP=CP=1"):
            connector._create_store(layout)
        setattr(parallel, field, 1)


@pytest.mark.parametrize("layerwise", [False, True])
@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("is_mla", [False, True])
def test_scheduler_constructor_context_store_initialization(layerwise, cp_size, is_mla):
    # Run the real constructors until the native-store factory boundary. Engine
    # dependencies are fixtures; do not prepopulate cp_world_size on the object.
    ns = connector_functions()
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            rank=0,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=cp_size,
            decode_context_parallel_size=1,
        ),
        cache_config=SimpleNamespace(block_size=16),
        model_config=SimpleNamespace(
            is_deepseek_mla=is_mla,
            get_num_layers=lambda _: 2,
            get_num_kv_heads=lambda _: 2,
            get_head_size=lambda: 8,
            dtype=SimpleNamespace(itemsize=2),
            hf_config=SimpleNamespace(),
        ),
        kv_transfer_config=SimpleNamespace(engine_id="constructor_test"),
    )
    launch = {
        "ucm_connectors": [
            {
                "ucm_connector_name": "UcmPipelineStore",
                "ucm_connector_config": {"store_pipeline": "ContextStore"},
            }
        ]
    }

    class FactoryReached(Exception):
        pass

    class Base(ns["Connector"]):
        def __init__(self, vllm_config, role, kv_cache_config):
            self._vllm_config, self._role = vllm_config, role
            self._dp_rank, self._gc_owner = 0, False

        def _set_default_shm_buffer_capacity(self, config):
            pass

        def _make_other_rank_hashers(self, config):
            return []

        def _create_store(self, layout):
            assert self.cp_world_size == 1
            return super()._create_store(layout)

    def factory(name, store_config, path):
        assert store_config["context_tp_size"] == 2
        raise FactoryReached()

    ns.update(
        {
            "Base": Base,
            "torch": SimpleNamespace(),
            "current_platform": SimpleNamespace(is_cuda_alike=lambda: True),
            "Config": lambda _: SimpleNamespace(get_config=lambda: launch),
            "RequestHasher": lambda *args: SimpleNamespace(seed=b"seed"),
            "UcmConnectorFactoryV1": SimpleNamespace(create_connector=factory),
        }
    )
    tree = ast.parse(
        (ROOT / "ucm/integration/vllm/ucm_connector.py").read_text(encoding="utf-8-sig")
    )
    classes = []
    for name, base in [
        ("UCMDirectConnector", "Base"),
        ("UCMLayerWiseConnector", "UCMDirectConnector"),
    ]:
        source = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name
        )
        constructor = next(
            n
            for n in source.body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        )
        classes.append(
            ast.ClassDef(
                name=name,
                bases=[ast.Name(id=base, ctx=ast.Load())],
                keywords=[],
                body=[constructor],
                decorator_list=[],
            )
        )
    code = ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[]))
    exec(compile(code, "connector_constructors", "exec"), ns)  # noqa: S102
    cls = ns["UCMLayerWiseConnector" if layerwise else "UCMDirectConnector"]
    with pytest.raises(FactoryReached if cp_size == 1 else ValueError):
        cls(config, "scheduler", None)


def mla_connector_methods():
    ns = connector_functions()
    tree = ast.parse(
        (ROOT / "ucm/integration/vllm/ucm_connector.py").read_text(encoding="utf-8-sig")
    )
    selected = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "UCMWorkerMetadata":
            selected.append(node)
        if isinstance(node, ast.ClassDef) and node.name in {
            "UCMDirectConnector",
            "UCMLayerWiseConnector",
        }:
            names = (
                {"wait_for_save"}
                if node.name == "UCMDirectConnector"
                else {"save_kv_layer", "wait_for_layer_load"}
            )
            selected.extend(
                n
                for n in node.body
                if isinstance(n, ast.FunctionDef) and n.name in names
            )
    ns.update(
        {
            "KVConnectorWorkerMetadata": object,
            "Any": object,
            "torch": SimpleNamespace(Tensor=object),
            "UCMConnectorMetadata": SimpleNamespace,
            "PendingDumpTask": lambda **kwargs: SimpleNamespace(**kwargs),
            "ucmmetrics": SimpleNamespace(update_stats=lambda *args: None),
            "logger": SimpleNamespace(
                debug=lambda *args: None, info=lambda *args: None
            ),
        }
    )
    exec(  # noqa: S102
        compile(
            ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
            "mla_methods",
            "exec",
        ),
        ns,
    )
    return ns


@pytest.mark.parametrize("layerwise", [False, True])
@pytest.mark.parametrize("context", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_mla_only_rank_zero_saves(layerwise, context, rank):
    ns = mla_connector_methods()
    connector = ns["Connector"]()
    connector.is_mla, connector._context_store_enabled = True, context
    connector.tp_rank, connector.tp_size = rank, 2
    block = bytes([1]) * 16
    metadata = SimpleNamespace(
        request_meta={"r": SimpleNamespace(dump_block_ids=([block], [0]))}
    )
    connector._connector_metadata = metadata
    connector._get_connector_metadata = lambda: metadata
    connector._store_block_ids = lambda keys: keys
    connector._dumped_layer_ids = set()
    connector.layer_name_to_id = {"mtp": 2}
    calls = []
    connector._submit_layerwise_dump_task = lambda *args: calls.append(args) or True
    connector._poll_pending_dump_tasks = lambda: None
    connector._async_dump_req_ids = set()
    connector._skip_null_vllm_blocks = False
    connector._pending_dump_tasks = []
    connector._get_dump_event_handle = lambda: 0
    connector.kv_cache_layout = SimpleNamespace(
        extract_block_addrs=lambda _: np.zeros((1, 3), dtype=np.uint64)
    )
    connector.store = object()
    connector._rank_consistency = SimpleNamespace(
        submit_dump=lambda *args: calls.append(args) or 1
    )
    connector.block_data_size = 64
    if layerwise:
        ns["save_kv_layer"](connector, "mtp", None, None)
        ns["save_kv_layer"](connector, "mtp", None, None)
    else:
        ns["wait_for_save"](connector)
    assert len(calls) == int(rank == 0)


def test_mla_success_aggregation_uses_rank_zero_copy():
    cls = mla_connector_methods()["UCMWorkerMetadata"]
    first = cls(is_mla=True, dump_succeeded_blocks={b"a", b"b"})
    second = cls(is_mla=True)
    assert first.aggregate(second).dump_succeeded_blocks == {b"a", b"b"}


def test_mla_mtp_last_layer_revisit_does_not_wait_twice():
    ns = mla_connector_methods()
    connector = SimpleNamespace(
        _connector_metadata=True,
        need_load=True,
        layer_name_to_id={"mtp": 2},
        layer_ids=[0, 1, 2],
        first_layer_id=0,
        load_tasks={2: {"r": "task"}},
        _get_connector_metadata=lambda: SimpleNamespace(
            request_meta={"r": SimpleNamespace(load_block_ids=([b"key"], [0]))}
        ),
        _layerwise_load_bytes=0,
        kv_cache_layout=SimpleNamespace(shard_size=56),
        _record_layerwise_load_duration=lambda *args: False,
    )
    waits = []
    connector._rank_consistency = SimpleNamespace(
        wait_load=lambda task: waits.append(task)
    )
    ns["wait_for_layer_load"](connector, "mtp")
    ns["wait_for_layer_load"](connector, "mtp")
    assert waits == ["task"]


def test_dsa_shared_three_components_and_mtp_shard_native_readback():
    module, library = native()
    name = "dsa_" + uuid.uuid4().hex
    sizes = [32, 16, 8]
    config = {
        "unique_id": name,
        "share_buffer_enable": True,
        "context_tp_size": 2,
        "block_size": 168,
        "shard_size": 56,
        "tensor_size_list": sizes,
        "context_memory_capacity_bytes": 168,
        "context_simulated_ssd_capacity_bytes": 168 * 4,
        "context_max_eviction_blocks": 1,
    }
    watcher = module.PipelineStore()
    watcher.Stack(
        "Context", str(library), {"unique_id": name, "share_buffer_enable": True}
    )
    # A reader can initialize before its owner; it allocates no payload.
    reader = module.PipelineStore()
    reader.Stack("Context", str(library), dict(config, device_id=1, context_tp_rank=1))
    owner = module.PipelineStore()
    owner.Stack("Context", str(library), dict(config, device_id=0, context_tp_rank=0))
    owner.ObserveRequest("r", 1, 1, ids(1))
    reader.ObserveRequest("r", 1, 1, ids(1))
    for layer in range(3):
        tensors = [
            np.full(size, layer * 3 + i, dtype=np.uint8) for i, size in enumerate(sizes)
        ]
        task = owner.Dump(
            ids(1),
            np.array([layer], dtype=np.uint64),
            np.array([[a.ctypes.data for a in tensors]], dtype=np.uint64),
            0,
        )
        owner.Wait(task)
        assert watcher.Lookup(ids(1)) == (b"\1" if layer == 2 else b"\0")
    with pytest.raises(RuntimeError, match="-50008"):
        reader.Dump(
            ids(1),
            np.array([0], dtype=np.uint64),
            np.array([[a.ctypes.data for a in tensors]], dtype=np.uint64),
            0,
        )
    # Read the single host copy on both ranks, first from Memory then SSD.
    for evict in (False, True):
        if evict:
            owner.ObserveRequest("new", 2, 2, ids(2))
            task = owner.Dump(
                ids(2),
                np.array([0], dtype=np.uint64),
                np.array([[a.ctypes.data for a in tensors]], dtype=np.uint64),
                0,
            )
            owner.Wait(task)
            assert owner.ContextStats()["ssd_write_bytes"] == 168
        for worker in (owner, reader):
            for layer in range(3):
                tensors = [np.zeros(size, dtype=np.uint8) for size in sizes]
                task = worker.Load(
                    ids(1),
                    np.array([layer], dtype=np.uint64),
                    np.array([[a.ctypes.data for a in tensors]], dtype=np.uint64),
                )
                worker.Wait(task)
                for i, tensor in enumerate(tensors):
                    assert np.all(tensor == layer * 3 + i)
    assert reader.ContextStats().get("d2h_bytes", 0) == 0
    assert reader.ContextStats().get("ssd_write_bytes", 0) == 0
    assert watcher.Lookup(ids(1, 2)) == b"\1\0"
    del reader
    assert watcher.Lookup(ids(1)) == b"\1"


@pytest.mark.parametrize("layerwise", [False, True])
def test_dsa_layout_preserves_three_components_and_mtp_layer(layerwise):
    import math
    import re

    class Tensor:
        def __init__(self, array):
            self.array, self.shape = array, array.shape

        def __getitem__(self, index):
            return Tensor(self.array[index])

        def dim(self):
            return self.array.ndim

        def element_size(self):
            return self.array.itemsize

        def data_ptr(self):
            return self.array.ctypes.data

    ns = {
        "np": np,
        "math": math,
        "List": list,
        "dataclass": dataclasses.dataclass,
        "torch": SimpleNamespace(Tensor=Tensor),
        "logger": SimpleNamespace(info=lambda *args: None),
        "extract_layer_index": lambda name: int(re.search(r"layers\.(\d+)", name)[1]),
    }
    tree = ast.parse(
        (ROOT / "ucm/integration/vllm/ucm_connector.py").read_text(encoding="utf-8-sig")
    )
    selected = [
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef)
        and n.name in {"KVCacheTensorInfo", "KVCacheLayout"}
    ]
    exec(  # noqa: S102
        compile(
            ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
            "dsa_layout",
            "exec",
        ),
        ns,
    )
    # Two target-model layers plus a registered MTP layer, each with 3 BF16-like
    # components. NumPy backs the tensors; the layout implementation is real.
    caches = {
        f"model.layers.{layer}.self_attn": tuple(
            Tensor(np.zeros((4, 2, 1, width), dtype=np.uint16))
            for width in (512, 64, 128)
        )
        for layer in range(3)
    }
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(num_hidden_layers=2)
        ),
    )
    layout = ns["KVCacheLayout"](
        caches, {"use_layerwise": layerwise}, config, SimpleNamespace(num_blocks=4)
    )
    sizes = [2048, 256, 512]
    assert layout.tensor_size_list == (sizes if layerwise else sizes * 3)
    assert layout.block_size == sum(sizes) * 3
    assert layout.shard_size == (sum(sizes) if layerwise else sum(sizes) * 3)
    addresses = layout.extract_block_addrs([1], layer_first=layerwise)
    expected = [
        component[1].data_ptr()
        for components in caches.values()
        for component in components
    ]
    assert addresses.reshape(-1).tolist() == expected


def shared_mla_reader_process(pipe, config):
    module, library = native()
    reader = module.PipelineStore()
    reader.Stack("Context", str(library), config)
    pipe.send("ready")
    while True:
        command = pipe.recv()
        if command == "close":
            break
        data = np.zeros(64, dtype=np.uint8)
        try:
            task = reader.Load(
                ids(1),
                np.array([0], dtype=np.uint64),
                np.array([[data.ctypes.data]], dtype=np.uint64),
            )
            reader.Wait(task)
            pipe.send((data.tobytes(), reader.ContextStats()))
        except RuntimeError:
            pipe.send("missing")
    del reader
    pipe.close()


def test_mla_shared_payload_across_processes_and_owner_exit():
    module, library = native()
    config = {
        "unique_id": "shared_" + uuid.uuid4().hex,
        "share_buffer_enable": True,
        "context_tp_size": 2,
        "block_size": 64,
        "shard_size": 64,
        "tensor_size": 64,
        "context_memory_capacity_bytes": 64,
        "context_simulated_ssd_capacity_bytes": 512,
        "context_max_eviction_blocks": 1,
    }
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(
        target=shared_mla_reader_process,
        args=(child, dict(config, device_id=1, context_tp_rank=1)),
    )
    process.start()
    child.close()
    try:
        assert parent.poll(30) and parent.recv() == "ready"
        owner = module.PipelineStore()
        owner.Stack(
            "Context", str(library), dict(config, device_id=0, context_tp_rank=0)
        )
        source = np.arange(64, dtype=np.uint8)
        for key in (1, 2):
            owner.ObserveRequest(str(key), key, key, ids(key))
            task = owner.Dump(
                ids(key),
                np.array([0], dtype=np.uint64),
                np.array([[source.ctypes.data]], dtype=np.uint64),
                0,
            )
            owner.Wait(task)
            parent.send("load")
            assert parent.poll(30)
            data, stats = parent.recv()
            assert data == source.tobytes()
            assert stats.get("d2h_bytes", 0) == 0
            assert stats.get("ssd_read_bytes", 0) == (64 if key == 2 else 0)
        assert owner.ContextStats()["ssd_write_bytes"] == 64
        del owner
        parent.send("load")
        assert parent.poll(30) and parent.recv() == "missing"
    finally:
        if process.is_alive():
            parent.send("close")
            process.join(10)
        if process.is_alive():
            process.terminate()
            process.join()
        parent.close()
    assert process.exitcode == 0
