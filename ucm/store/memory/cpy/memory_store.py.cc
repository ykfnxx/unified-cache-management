/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
#include "memory_store.h"
#include "template/store_binder.h"

PYBIND11_MODULE(ucmmemorystore, module)
{
    namespace py = pybind11;
    using namespace UC::MemoryStore;
    using MemoryStorePy = UC::Detail::StoreBinder<MemoryStore, Config>;

    module.attr("project") = UCM_PROJECT_NAME;
    module.attr("version") = UCM_PROJECT_VERSION;
    module.attr("commit_id") = UCM_COMMIT_ID;
    module.attr("build_type") = UCM_BUILD_TYPE;

    auto store = py::class_<MemoryStorePy, std::unique_ptr<MemoryStorePy>>(module, "MemoryStore");
    auto config = py::class_<Config>(store, "Config");
    config.def(py::init<>());
    config.def_readwrite("storeBackend", &Config::storeBackend);
    config.def_readwrite("uniqueId", &Config::uniqueId);
    config.def_readwrite("deviceId", &Config::deviceId);
    config.def_readwrite("shardSize", &Config::shardSize);
    config.def_readwrite("blockSize", &Config::blockSize);
    config.def_readwrite("tensorSizeList", &Config::tensorSizeList);
    config.def_readwrite("memoryTokenChunkSize", &Config::memoryTokenChunkSize);
    config.def_readwrite("memoryBufferCapacity", &Config::memoryBufferCapacity);
    config.def_readwrite("waitingQueueDepth", &Config::waitingQueueDepth);
    config.def_readwrite("runningQueueDepth", &Config::runningQueueDepth);
    config.def_readwrite("timeoutMs", &Config::timeoutMs);
    config.def_readwrite("requiredTensorTypes", &Config::requiredTensorTypes);
    config.def_readwrite("tensorSizesByType", &Config::tensorSizesByType);
    config.def_readwrite("tokensPerBlock", &Config::tokensPerBlock);

    store.def(py::init<>());
    store.def("Self", &MemoryStorePy::Self);
    store.def("Setup", &MemoryStorePy::Setup);
    store.def("Lookup", &MemoryStorePy::Lookup, py::arg("ids").noconvert());
    store.def("LookupOnPrefix", &MemoryStorePy::LookupOnPrefix, py::arg("ids").noconvert());
    store.def("Prefetch", &MemoryStorePy::Prefetch, py::arg("ids").noconvert());
    store.def("LookupTokens", &MemoryStorePy::LookupTokens, py::arg("ids").noconvert(),
              py::arg("layer_ids").noconvert(), py::arg("token_offsets").noconvert(),
              py::arg("tensor_types").noconvert());
    store.def("Load", &MemoryStorePy::Load, py::arg("ids").noconvert(),
              py::arg("indexes").noconvert(), py::arg("addrs").noconvert());
    store.def("LoadTokens", &MemoryStorePy::LoadTokens, py::arg("ids").noconvert(),
              py::arg("layer_ids").noconvert(), py::arg("token_offsets").noconvert(),
              py::arg("tensor_types").noconvert(), py::arg("addrs").noconvert());
    store.def("Dump", &MemoryStorePy::Dump, py::arg("ids").noconvert(),
              py::arg("indexes").noconvert(), py::arg("addrs").noconvert(),
              py::arg("prerequisite_handle") = 0);
    store.def("DumpTokens", &MemoryStorePy::DumpTokens, py::arg("ids").noconvert(),
              py::arg("layer_ids").noconvert(), py::arg("token_offsets").noconvert(),
              py::arg("tensor_types").noconvert(), py::arg("addrs").noconvert(),
              py::arg("prerequisite_handle") = 0);
    store.def("Check", &MemoryStorePy::Check);
    store.def("Wait", &MemoryStorePy::Wait);
}
