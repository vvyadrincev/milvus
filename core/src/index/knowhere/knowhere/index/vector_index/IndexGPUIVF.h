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

#include <memory>
#include <utility>

#include "IndexIVF.h"
#include "knowhere/index/vector_index/helpers/FaissGpuResourceMgr.h"

namespace knowhere {

class GPUIndex {
 public:
    explicit GPUIndex(const int& device_id) : gpu_id_(device_id) {
    }

    GPUIndex(const int& device_id, const ResPtr& resource) : gpu_id_(device_id), res_(resource) {
    }

    virtual VectorIndexPtr
    CopyGpuToCpu(const Config& config) = 0;

    virtual VectorIndexPtr
    CopyGpuToGpu(const int64_t& device_id, const Config& config, size_t& size) = 0;

    void
    SetGpuDevice(const int& gpu_id);

    const int64_t&
    GetGpuDevice();

 protected:
    int64_t gpu_id_;
    ResWPtr res_;
};

class GPUIVF : public GenericIVF, public GPUIndex {
 public:
    explicit GPUIVF(const int& device_id) : GenericIVF(), GPUIndex(device_id) {
    }

    explicit GPUIVF(std::shared_ptr<faiss::Index> index, const int64_t& device_id, const ResPtr& resource)
        : GenericIVF(std::move(index)), GPUIndex(device_id, resource) {
    }

    IndexModelPtr
    Train(const DatasetPtr& dataset, const Config& config) override;

    void
    Add(const DatasetPtr& dataset, const Config& config) override;

    void
    set_index_model(IndexModelPtr model) override;

    // DatasetPtr Search(const DatasetPtr &dataset, const Config &config) override;
    VectorIndexPtr
    CopyGpuToCpu(const Config& config) override;

    VectorIndexPtr
    CopyGpuToGpu(const int64_t& device_id, const Config& config, size_t& size) override;

    //    VectorIndexPtr
    //    Clone() final;

 protected:

    virtual
    void set_nprobe(size_t nprobe);

    void
    search_impl(int64_t n, const float* data, int64_t k, float* distances, int64_t* labels, const Config& cfg) override;

    BinarySet
    SerializeImpl() override;

    void
    LoadImpl(const BinarySet& index_binary) override;
};


class GenericGPUIVF : public GPUIVF {
public:

    explicit GenericGPUIVF(const int& device_id) : GPUIVF(device_id){
    }

    explicit GenericGPUIVF(std::shared_ptr<faiss::Index> index, const int64_t& device_id,
                           const ResPtr& resource)
        : GPUIVF(index, device_id, resource) {
    }

    void
    set_index_model(IndexModelPtr model) override{}

    IndexModelPtr
    Train(const DatasetPtr& dataset, const Config& config) override;


    VectorIndexPtr
    CopyGpuToGpu(const int64_t& device_id, const Config& config, size_t& size)override;
    VectorIndexPtr
    CopyCpuToGpu(const int64_t& device_id, const Config& config)override;

protected:
    void set_nprobe(size_t nprobe)override;

    BinarySet
    SerializeImpl() override;

    void
    LoadImpl(const BinarySet& index_binary)override;

    uint64_t
    LoadToGPU();

};


}  // namespace knowhere
