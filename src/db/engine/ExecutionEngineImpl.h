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

#pragma once

#include "ExecutionEngine.h"
#include "wrapper/VecIndex.h"

#include <memory>
#include <string>

namespace milvus {
namespace engine {

class ExecutionEngineImpl : public ExecutionEngine {
 public:
    ExecutionEngineImpl(uint16_t dimension, const std::string& location, EngineType index_type, MetricType metric_type,
                        int32_t nlist, const std::string& enc_type = "Flat");

    ExecutionEngineImpl(VecIndexPtr index, const std::string& location, EngineType index_type, MetricType metric_type,
                        int32_t nlist);

    Status
    Reconstruct(std::vector<int64_t> ids, std::vector<float>& xb,
                std::vector<bool>& found)override;
    Status
    GetIds(std::vector<int64_t>& ids)override;

    Status
    AddWithIds(int64_t n, const float* xdata, const int64_t* xids) override;

    Status
    AddWithIds(int64_t n, const uint8_t* xdata, const int64_t* xids) override;

    size_t
    Count() const override;

    size_t
    Size() const override;

    size_t
    Dimension() const override;

    size_t
    PhysicalSize() const override;

    Status
    Serialize() override;

    Status
    Load(bool to_cache, bool force = false) override;

    Status
    CopyToGpu(uint64_t device_id, bool hybrid = false) override;

    Status
    CopyToIndexFileToGpu(uint64_t device_id) override;

    Status
    CopyToCpu() override;

    //    ExecutionEnginePtr
    //    Clone() override;

    Status
    Reserve(uint64_t bytes, uint64_t vec_cnt) override;

    faiss::IndexIDMap2*
    GetFaissIndex()override;

    Status
    Merge(const std::string& location) override;

    Status
    Search(int64_t n, const float* data, int64_t k, int64_t nprobe, float* distances, int64_t* labels,
           bool hybrid = false) override;

    Status
    Search(int64_t n, const uint8_t* data, int64_t k, int64_t nprobe, float* distances, int64_t* labels,
           bool hybrid = false) override;

    ExecutionEnginePtr
    BuildIndex(const std::string& location, EngineType engine_type) override;

    Status
    Cache() override;

    Status
    GpuCache(uint64_t gpu_id) override;

    Status
    Init() override;

    EngineType
    IndexEngineType() const override {
        return index_type_;
    }

    MetricType
    IndexMetricType() const override {
        return metric_type_;
    }

    std::string
    GetLocation() const override {
        return location_;
    }

 private:
    VecIndexPtr
    CreatetVecIndex(EngineType type);

    VecIndexPtr
    Load(const std::string& location);

    void
    HybridLoad() const;

    void
    HybridUnset() const;

 protected:
    VecIndexPtr index_ = nullptr;
    EngineType index_type_;
    MetricType metric_type_;

    int64_t dim_;
    std::string location_;

    int64_t nlist_ = 0;
    int64_t gpu_num_ = 0;
    std::string enc_type_;
};

}  // namespace engine
}  // namespace milvus
