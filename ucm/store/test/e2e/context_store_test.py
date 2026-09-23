"""CPU simu integration tests. Set UCM_CONTEXT_BUILD to the CMake build directory."""

import ast
import copy
import dataclasses
import importlib.util
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

    class Pipeline:
        def __init__(self):
            self.store = module.PipelineStore()

        def Stack(self, name, library, config):
            fake_config = dict(config, share_buffer_enable=True, buffer_number=4096)
            self.store.Stack(
                "Fake", str(build / "ucm/store/fake/libfakestore.so"), fake_config
            )
            self.store.Stack(name, library, config)

        def __getattr__(self, name):
            return getattr(self.store, name)

    return SimpleNamespace(
        PipelineStore=Pipeline
    ), build / "ucm/store/context/libcontextstore.so"


def ids(*numbers):
    return np.frombuffer(
        b"".join(n.to_bytes(16, "little") for n in numbers), dtype=np.uint8
    )


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
                    "_store_block_ids",
                    "_consistency_manager_enabled",
                    "_create_store",
                    "_set_default_shm_buffer_capacity",
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
@pytest.mark.parametrize("rank", [0, 1])
def test_mla_only_rank_zero_saves(layerwise, rank):
    ns = mla_connector_methods()
    connector = ns["Connector"]()
    connector.is_mla = True
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


def test_native_context_fake_shared_layerwise():
    module, library = native()
    config = {
        "unique_id": "clock_" + uuid.uuid4().hex,
        "share_buffer_enable": True,
        "device_id": 0,
        "context_memory_capacity_gb": 1,
        "cache_load_exclusive_buffer_number": 0,
        "shard_size": 1 << 20,
        "block_size": 3 << 20,
        "tensor_size_list": [32, 16, 8],
        "waiting_queue_depth": 64,
        "running_queue_depth": 128,
    }
    owner = module.PipelineStore()
    owner.Stack("Context", str(library), config)
    reader = module.PipelineStore()
    reader.Stack("Context", str(library), dict(config, device_id=1))
    watcher = module.PipelineStore()
    watcher.Stack("Context", str(library), dict(config, device_id=-1))
    for layer in range(3):
        source = [
            np.full(n, layer * 3 + i, dtype=np.uint8)
            for i, n in enumerate(config["tensor_size_list"])
        ]
        layer_ids = np.array([layer], dtype=np.uint64)
        ptrs = np.array([[a.ctypes.data for a in source]], dtype=np.uint64)
        owner.Wait(owner.Dump(ids(1), layer_ids, ptrs, 0))
        target = [np.zeros_like(a) for a in source]
        ptrs = np.array([[a.ctypes.data for a in target]], dtype=np.uint64)
        reader.Wait(reader.Load(ids(1), layer_ids, ptrs))
        for before, after in zip(source, target):
            np.testing.assert_array_equal(before, after)
    assert watcher.Lookup(ids(1)) == b"\1"


def test_default_metrics_match_yaml_and_include_shard_counters():
    import runpy

    import yaml

    generated = runpy.run_path(str(ROOT / "ucm/default_metrics_config.py"))[
        "DEFAULT_METRICS_CONFIG"
    ]
    source = yaml.safe_load(
        (ROOT / "examples/metrics/metrics_configs.yaml").read_text()
    )
    for kind in ("counter", "histogram", "gauge"):
        left = {m["name"]: m for m in generated[kind]}
        right = {m["name"]: m for m in source[kind]}
        assert left == right
    counters = {m["name"] for m in generated["counter"]}
    assert {
        "context_evict_shards_total",
        "context_writeback_shards_total",
        "context_drop_shards_total",
    } <= counters
    assert "context_evict_blocks_total" not in counters


@pytest.mark.parametrize(
    "field,value,gb",
    [
        ("context_memory_capacity_gb", 64, 64),
        ("context_memory_capacity_bytes", 65536, 65536 / (1 << 30)),
    ],
)
def test_context_capacity_keeps_original_names_without_cache_default(field, value, gb):
    ns = connector_functions()
    checks = []
    ns["_check_shm_capacity"] = lambda *args: checks.append(args)
    config = {
        "store_pipeline": "ContextStore|Fake",
        "share_buffer_enable": True,
        field: value,
    }
    original = dict(config)
    ns["Connector"]()._set_default_shm_buffer_capacity(config)
    assert config == original
    assert checks == [(gb, field)]


def test_cache_capacity_default_is_unchanged():
    ns = connector_functions()
    checks = []
    ns["_check_shm_capacity"] = lambda *args: checks.append(args)
    config = {"store_pipeline": "Cache|Fake", "share_buffer_enable": True}
    ns["Connector"]()._set_default_shm_buffer_capacity(config)
    assert config["cache_buffer_capacity_gb"] == 128
    assert checks == [(128,)]
