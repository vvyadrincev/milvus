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

#include "knowhere/index/vector_index/IndexGPUIDMAP.h"

#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/AutoTune.h>
#include <faiss/IndexFlat.h>
#include <faiss/MetaIndexes.h>
#include <faiss/index_io.h>
#include <fiu-local.h>

#ifdef MILVUS_GPU_VERSION
#include <faiss/gpu/GpuCloner.h>
#endif

#include "knowhere/adapter/VectorAdapter.h"
#include "knowhere/common/Exception.h"
#include "knowhere/index/vector_index/IndexIDMAP.h"
#include "knowhere/index/vector_index/helpers/FaissIO.h"

namespace knowhere {

VectorIndexPtr
GPUIDMAP::CopyGpuToCpu(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);

    faiss::Index* device_index = index_.get();
    faiss::Index* host_index = faiss::gpu::index_gpu_to_cpu(device_index);

    std::shared_ptr<faiss::Index> new_index;
    new_index.reset(host_index);
    return std::make_shared<IDMAP>(new_index);
}

// VectorIndexPtr
// GPUIDMAP::Clone() {
//    auto cpu_idx = CopyGpuToCpu(Config());
//
//    if (auto idmap = std::dynamic_pointer_cast<IDMAP>(cpu_idx)) {
//        return idmap->CopyCpuToGpu(gpu_id_, Config());
//    } else {
//        KNOWHERE_THROW_MSG("IndexType not Support GpuClone");
//    }
//}

BinarySet
GPUIDMAP::SerializeImpl() {
    try {
        fiu_do_on("GPUIDMP.SerializeImpl.throw_exception", throw std::exception());
        MemoryIOWriter writer;
        {
            faiss::Index* index = index_.get();
            faiss::Index* host_index = faiss::gpu::index_gpu_to_cpu(index);

            faiss::write_index(host_index, &writer);
            delete host_index;
        }
        auto data = std::make_shared<uint8_t>();
        data.reset(writer.data_, [](uint8_t* p){delete [] p;});

        BinarySet res_set;
        res_set.Append("FLAT", data, writer.rp);

        return res_set;
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

void
GPUIDMAP::LoadImpl(const BinarySet& index_binary) {
    auto binary = index_binary.GetByName("FLAT");
    MemoryIOReader reader;
    {
        reader.total = binary->size;
        reader.data_ = binary->data.get();

        faiss::Index* index = faiss::read_index(&reader);

        if (auto res = FaissGpuResourceMgr::GetInstance().GetRes(gpu_id_)) {
            ResScope rs(res, gpu_id_, false);
            auto device_index = faiss::gpu::index_cpu_to_gpu(res->faiss_res.get(), gpu_id_, index);
            index_.reset(device_index);
            res_ = res;
        } else {
            KNOWHERE_THROW_MSG("Load error, can't get gpu resource");
        }

        delete index;
    }
}

VectorIndexPtr
GPUIDMAP::CopyGpuToGpu(const int64_t& device_id, const Config& config, size_t& size) {
    auto cpu_index = CopyGpuToCpu(config);
    return  std::static_pointer_cast<IDMAP>(cpu_index)->CopyCpuToGpu(device_id, config);

}

float*
GPUIDMAP::GetRawVectors() {
    KNOWHERE_THROW_MSG("Not support");
}

int64_t*
GPUIDMAP::GetRawIds() {
    KNOWHERE_THROW_MSG("Not support");
}

void
GPUIDMAP::search_impl(int64_t n, const float* data, int64_t k, float* distances, int64_t* labels, const Config& cfg) {
    ResScope rs(res_, gpu_id_);
    index_->search(n, (float*)data, k, distances, labels);
}

void
GPUIDMAP::GenGraph(const float* data, const int64_t& k, Graph& graph, const Config& config) {
    int64_t K = k + 1;
    auto ntotal = Count();

    size_t dim = config->d;
    auto batch_size = 1000;
    auto tail_batch_size = ntotal % batch_size;
    auto batch_search_count = ntotal / batch_size;
    auto total_search_count = tail_batch_size == 0 ? batch_search_count : batch_search_count + 1;

    std::vector<float> res_dis(K * batch_size);
    graph.resize(ntotal);
    Graph res_vec(total_search_count);
    for (int i = 0; i < total_search_count; ++i) {
        auto b_size = (i == (total_search_count - 1)) && tail_batch_size != 0 ? tail_batch_size : batch_size;

        auto& res = res_vec[i];
        res.resize(K * b_size);

        auto xq = data + batch_size * dim * i;
        search_impl(b_size, (float*)xq, K, res_dis.data(), res.data(), config);

        for (int j = 0; j < b_size; ++j) {
            auto& node = graph[batch_size * i + j];
            node.resize(k);
            auto start_pos = j * K + 1;
            for (int m = 0, cursor = start_pos; m < k && cursor < start_pos + k; ++m, ++cursor) {
                node[m] = res[cursor];
            }
        }
    }
}

IndexModelPtr
GPUFlatFP16::Train(const DatasetPtr& dataset, const Config& config) {
    config->CheckValid();

    gpu_id_ = config->gpu_id;

    auto temp_resource = FaissGpuResourceMgr::GetInstance().GetRes(gpu_id_);
    if (temp_resource == nullptr)
        KNOWHERE_THROW_MSG("can't get gpu resource");

    ResScope rs(temp_resource, gpu_id_, true);


    auto idmap = new faiss::IndexIDMap2(new faiss::IndexFlatIP(config->d));
    idmap->own_fields = true;
    index_.reset(idmap);

    GETTENSOR(dataset)

    index_->train(rows, (float*)p_data);

    return nullptr;
}

BinarySet
GPUFlatFP16::SerializeImpl() {
    try {
        MemoryIOWriter writer;
        {
            auto idmap_index = dynamic_cast<faiss::IndexIDMap2*>(index_.get());
            if (not idmap_index)
                KNOWHERE_THROW_MSG("index is not IndexIDMap2!");

            faiss::write_index(idmap_index, &writer);
        }
        auto data = std::make_shared<uint8_t>();
        data.reset(writer.data_);

        BinarySet res_set;
        res_set.Append("FLAT", data, writer.rp);

        return res_set;
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

void
GPUFlatFP16::
LoadImpl(const BinarySet& index_binary){
    if (index_)
        return;

    auto binary = index_binary.GetByName("FLAT");
    MemoryIOReader reader;
    {
        reader.total = binary->size;
        reader.data_ = binary->data.get();

        std::shared_ptr<faiss::Index> index;
        index.reset(faiss::read_index(&reader));

        index_ = index;
        if (gpu_id_ == -1){
            return;

        }
        LoadToGPU();

    }

}

uint64_t
GPUFlatFP16::
LoadToGPU(){
    if (auto res = FaissGpuResourceMgr::GetInstance().GetRes(gpu_id_)) {
        ResScope rs(res, gpu_id_, false);
        // faiss::gpu::GpuClonerOptions opts;
        // opts.useFloat16 = true;
        // opts.verbose = true;
        faiss::gpu::GpuIndexFlatConfig config;
        // config.useFloat16 = true;
        config.device = gpu_id_;

        auto idmap_index = dynamic_cast<faiss::IndexIDMap2*>(index_.get());
        if (not idmap_index)
            KNOWHERE_THROW_MSG("index is not IndexIDMap2!");

        std::shared_ptr<faiss::Index> temp;
        temp.reset(idmap_index->index);
        const auto* host_index = dynamic_cast<const faiss::IndexFlat *>(temp.get());
        if (not host_index)
            KNOWHERE_THROW_MSG("host index is not IndexFlat!");


        idmap_index->index = new faiss::gpu::GpuIndexFlatIP(res->faiss_res.get(),
                                                            host_index->d, config);

        const int64_t batch = 3'000'000;
        const auto batches_cnt = host_index->ntotal/batch + 1;
        for (int i=0; i < batches_cnt; ++i){
            auto offs = i * batch;
            idmap_index->index->add(std::min(batch, host_index->ntotal-offs),
                                    host_index->xb.data()+offs);
        }

        // auto device_index = faiss::gpu::index_cpu_to_gpu(res->faiss_res.get(),
        //                                                  gpu_id_, host_index.get(), &opts);
        // idmap_index->index = device_index;
        auto size = host_index->ntotal * host_index->d * 2;
        // delete host_index;
        res_ = res;
        return size;
    } else {
        KNOWHERE_THROW_MSG("Load error, can't get gpu resource");
    }

}

VectorIndexPtr
GPUFlatFP16::
CopyGpuToGpu(const int64_t& device_id, const Config& config, size_t& size){
    //dont look at bs function name
    SetGpuDevice(device_id);
    size = LoadToGPU();

    return std::make_shared<GPUFlatFP16>(index_, device_id, ResPtr(res_));

}



}  // namespace knowhere
