// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "knowhere/index/vector_index/helpers/FaissGpuResourceMgr.h"

#include <fiu-local.h>
#include <utility>

namespace knowhere {

FaissGpuResourceMgr&
FaissGpuResourceMgr::GetInstance() {
    static FaissGpuResourceMgr instance;
    return instance;
}

void
FaissGpuResourceMgr::AllocateTempMem(ResPtr& resource, const int64_t& device_id, const int64_t& size) {
    if (size) {
        resource->faiss_res->setTempMemory(size);
    } else {
        auto search = devices_params_.find(device_id);
        if (search != devices_params_.end()) {
            resource->faiss_res->setTempMemory(search->second.temp_mem_size);
        }
        // else do nothing. allocate when use.
    }
}

void
FaissGpuResourceMgr::InitDevice(int64_t device_id, int64_t pin_mem_size, int64_t temp_mem_size, int64_t res_num) {
    DeviceParams params;
    params.pinned_mem_size = pin_mem_size;
    params.temp_mem_size = temp_mem_size;
    params.resource_num = res_num;

    devices_params_.emplace(device_id, params);
}

void
FaissGpuResourceMgr::InitResource() {
    if (is_init)
        return;

    is_init = true;

    // std::cout << "InitResource" << std::endl;
    for (auto& device : devices_params_) {
        auto& device_id = device.first;

        mutex_cache_.emplace(device_id, std::make_unique<std::mutex>());

        // std::cout << "Device Id: " << DEVICEID << std::endl;
        auto& device_param = device.second;
        auto& bq = idle_map_[device_id];

        for (int64_t i = 0; i < device_param.resource_num; ++i) {
            // std::cout << "Resource Id: " << i << std::endl;
            auto raw_resource = std::make_shared<faiss::gpu::StandardGpuResources>();

            // TODO(linxj): enable set pinned memory
            auto res_wrapper = std::make_shared<Resource>(raw_resource);
            AllocateTempMem(res_wrapper, device_id, 0);

            bq.Put(res_wrapper);
        }
    }
    // std::cout << "End initResource" << std::endl;
}

ResPtr
FaissGpuResourceMgr::GetRes(const int64_t& device_id, const int64_t& alloc_size) {
    fiu_return_on("FaissGpuResourceMgr.GetRes.ret_null", nullptr);
    InitResource();

    auto finder = idle_map_.find(device_id);
    if (finder != idle_map_.end()) {
        auto& bq = finder->second;
        auto&& resource = bq.Take();
        AllocateTempMem(resource, device_id, alloc_size);
        return resource;
    }
    return nullptr;
}

void
FaissGpuResourceMgr::MoveToIdle(const int64_t& device_id, const ResPtr& res) {
    auto finder = idle_map_.find(device_id);
    if (finder != idle_map_.end()) {
        auto& bq = finder->second;
        bq.Put(res);
    }
}

void
FaissGpuResourceMgr::Free() {
    for (auto& item : idle_map_) {
        auto& bq = item.second;
        while (!bq.Empty()) {
            bq.Take();
        }
    }
    is_init = false;
}

void
FaissGpuResourceMgr::Dump() {
    for (auto& item : idle_map_) {
        auto& bq = item.second;
        std::cout << "DEVICEID: " << item.first << ", resource count:" << bq.Size();
    }
}

}  // namespace knowhere
