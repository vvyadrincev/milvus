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

#include <memory>

#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexIVF.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/index_io.h>
#include <faiss/index_factory.h>
#include <faiss/MetaIndexes.h>
#include <faiss/IndexPreTransform.h>
#include <fiu-local.h>

#include "knowhere/adapter/VectorAdapter.h"
#include "knowhere/common/Exception.h"
#include "knowhere/index/vector_index/IndexGPUIVF.h"
#include "knowhere/index/vector_index/IndexIVFPQ.h"
#include "knowhere/index/vector_index/helpers/Cloner.h"
#include "knowhere/index/vector_index/helpers/FaissIO.h"

#include <boost/algorithm/string/predicate.hpp>

namespace knowhere {

IndexModelPtr
GPUIVF::Train(const DatasetPtr& dataset, const Config& config) {
    auto build_cfg = std::dynamic_pointer_cast<IVFCfg>(config);
    if (build_cfg != nullptr) {
        build_cfg->CheckValid();  // throw exception
    }
    gpu_id_ = build_cfg->gpu_id;

    GETTENSOR(dataset)

    auto temp_resource = FaissGpuResourceMgr::GetInstance().GetRes(gpu_id_);
    if (temp_resource != nullptr) {
        ResScope rs(temp_resource, gpu_id_, true);
        faiss::gpu::GpuIndexIVFFlatConfig idx_config;
        idx_config.device = gpu_id_;
        faiss::gpu::GpuIndexIVFFlat device_index(temp_resource->faiss_res.get(), dim, build_cfg->nlist,
                                                 GetMetricType(build_cfg->metric_type), idx_config);
        device_index.train(rows, (float*)p_data);

        std::shared_ptr<faiss::Index> host_index = nullptr;
        host_index.reset(faiss::gpu::index_gpu_to_cpu(&device_index));

        return std::make_shared<IVFIndexModel>(host_index);
    } else {
        KNOWHERE_THROW_MSG("Build IVF can't get gpu resource");
    }
}

void
GPUIVF::set_index_model(IndexModelPtr model) {
    std::lock_guard<std::mutex> lk(mutex_);

    auto host_index = std::static_pointer_cast<IVFIndexModel>(model);
    if (auto gpures = FaissGpuResourceMgr::GetInstance().GetRes(gpu_id_)) {
        ResScope rs(gpures, gpu_id_, false);
        auto device_index = faiss::gpu::index_cpu_to_gpu(gpures->faiss_res.get(), gpu_id_, host_index->index_.get());
        index_.reset(device_index);
        res_ = gpures;
    } else {
        KNOWHERE_THROW_MSG("load index model error, can't get gpu_resource");
    }
}

BinarySet
GPUIVF::SerializeImpl() {
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }

    try {
        fiu_do_on("GPUIVF.SerializeImpl.throw_exception", throw std::exception());
        MemoryIOWriter writer;
        {
            faiss::Index* index = index_.get();
            faiss::Index* host_index = faiss::gpu::index_gpu_to_cpu(index);

            faiss::write_index(host_index, &writer);
            delete host_index;
        }
        auto data = std::make_shared<uint8_t>();
        data.reset(writer.data_);

        BinarySet res_set;
        res_set.Append("IVF", data, writer.rp);

        return res_set;
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

void
GPUIVF::LoadImpl(const BinarySet& index_binary) {
    auto binary = index_binary.GetByName("IVF");
    MemoryIOReader reader;
    {
        reader.total = binary->size;
        reader.data_ = binary->data.get();

        faiss::Index* index = faiss::read_index(&reader);

        if (auto temp_res = FaissGpuResourceMgr::GetInstance().GetRes(gpu_id_)) {
            ResScope rs(temp_res, gpu_id_, false);
            auto device_index = faiss::gpu::index_cpu_to_gpu(temp_res->faiss_res.get(), gpu_id_, index);
            index_.reset(device_index);
            res_ = temp_res;
        } else {
            KNOWHERE_THROW_MSG("Load error, can't get gpu resource");
        }

        delete index;
    }
}

void
GPUIVF::
set_nprobe(size_t nprobe){

    auto device_index = dynamic_cast<faiss::gpu::GpuIndexIVF*>(index_.get());
    fiu_do_on("GPUIVF.search_impl.invald_index", device_index = nullptr);

    if (!device_index)
        KNOWHERE_THROW_MSG("Not a GpuIndexIVF type.");

    device_index->nprobe = nprobe;
}

void
GPUIVF::search_impl(int64_t n, const float* data, int64_t k, float* distances, int64_t* labels, const Config& cfg) {
    std::lock_guard<std::mutex> lk(mutex_);

    auto search_cfg = std::dynamic_pointer_cast<IVFCfg>(cfg);
    set_nprobe(search_cfg->nprobe);
    ResScope rs(res_, gpu_id_);
    index_->search(n, (float*)data, k, distances, labels);
}

VectorIndexPtr
GPUIVF::CopyGpuToCpu(const Config& config) {
    std::lock_guard<std::mutex> lk(mutex_);

    if (auto device_idx = std::dynamic_pointer_cast<faiss::gpu::GpuIndexIVF>(index_)) {
        faiss::Index* device_index = index_.get();
        faiss::Index* host_index = faiss::gpu::index_gpu_to_cpu(device_index);

        std::shared_ptr<faiss::Index> new_index;
        new_index.reset(host_index);
        return std::make_shared<IVF>(new_index);
    } else {
        return std::make_shared<IVF>(index_);
    }
}

// VectorIndexPtr
// GPUIVF::Clone() {
//    auto cpu_idx = CopyGpuToCpu(Config());
//    return knowhere::cloner::CopyCpuToGpu(cpu_idx, gpu_id_, Config());
//}

VectorIndexPtr
GPUIVF::CopyGpuToGpu(const int64_t& device_id, const Config& config, size_t& size) {
    auto host_index = CopyGpuToCpu(config);
    return std::static_pointer_cast<IVF>(host_index)->CopyCpuToGpu(device_id, config);
}

void
GPUIVF::Add(const DatasetPtr& dataset, const Config& config) {
    if (auto spt = res_.lock()) {
        ResScope rs(res_, gpu_id_);
        IVF::Add(dataset, config);
    } else {
        KNOWHERE_THROW_MSG("Add IVF can't get gpu resource");
    }
}

void
GPUIndex::SetGpuDevice(const int& gpu_id) {
    gpu_id_ = gpu_id;
}

const int64_t&
GPUIndex::GetGpuDevice() {
    return gpu_id_;
}


faiss::gpu::GpuClonerOptions
CreateFaissOpts(){

    faiss::gpu::GpuClonerOptions opts;
    //useFloat16 affects only PQ indexes
    //TODO make it dynamic; if PQ > 48
    opts.useFloat16 = true;
    // opts.useFloat16 = false;
    opts.useFloat16CoarseQuantizer = false;
    opts.indicesOptions = faiss::gpu::INDICES_64_BIT;
    // opts.indicesOptions = faiss::gpu::INDICES_IVF;
    return opts;
}


IndexModelPtr
GenericGPUIVF::
Train(const DatasetPtr& dataset, const Config& config){

    auto build_cfg = std::dynamic_pointer_cast<IVFCfg>(config);
    if (build_cfg != nullptr) {
        build_cfg->CheckValid();  // throw exception
    }
    gpu_id_ = build_cfg->gpu_id;

    GETTENSOR(dataset)

    std::string pretransform;
    std::string enc = build_cfg->enc_type;
    if (boost::starts_with(build_cfg->enc_type, "OPQ")){
        pretransform=build_cfg->enc_type + ",";
        enc = build_cfg->enc_type.substr(1);
    }

    std::stringstream index_type;
    index_type <<pretransform<< "IVF" << build_cfg->nlist << "," << enc;




    KNOWHERE_LOG_DEBUG << "Index type: " << index_type.str();


    std::shared_ptr<faiss::Index> host_index;
    host_index.reset(faiss::index_factory(dim, index_type.str().c_str(),
                                          GetMetricType(build_cfg->metric_type)));

    auto temp_resource = FaissGpuResourceMgr::GetInstance().GetRes(gpu_id_);
    if (temp_resource != nullptr) {
        ResScope rs(temp_resource, gpu_id_, true);
        auto opts = CreateFaissOpts();

        auto device_index = faiss::gpu::index_cpu_to_gpu(temp_resource->faiss_res.get(),
                                                         gpu_id_, host_index.get(), &opts);


        auto idmap = new faiss::IndexIDMap2(device_index);
        idmap->own_fields = true;
        index_.reset(idmap);

        index_->train(rows, (float*)p_data);
        res_ = temp_resource;


        return nullptr;
    } else {
        KNOWHERE_THROW_MSG("Build IVF can't get gpu resource");
    }


}

BinarySet
GenericGPUIVF::SerializeImpl() {
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }

    try {
        MemoryIOWriter writer;
        {
            auto idmap_index = dynamic_cast<faiss::IndexIDMap2*>(index_.get());
            if (not idmap_index)
                KNOWHERE_THROW_MSG("index is not IndexIDMap2!");

            auto device_index = idmap_index->index;
            std::shared_ptr<faiss::Index> host_index;
            host_index.reset(faiss::gpu::index_gpu_to_cpu(device_index));

            idmap_index->index = host_index.get();
            faiss::write_index(idmap_index, &writer);
            idmap_index->index = device_index;
        }
        auto data = std::make_shared<uint8_t>();
        data.reset(writer.data_);

        BinarySet res_set;
        res_set.Append("IVF", data, writer.rp);

        return res_set;
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

void
GenericGPUIVF::
LoadImpl(const BinarySet& index_binary){

    if (index_)
        return;

    auto binary = index_binary.GetByName("IVF");
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

void
GenericGPUIVF::
set_nprobe(size_t nprobe){

    auto idmap_index = dynamic_cast<faiss::IndexIDMap2*>(index_.get());
    if (not idmap_index)
        KNOWHERE_THROW_MSG("index is not IndexIDMap2!");

    auto* index = idmap_index->index;
    auto pretransform = dynamic_cast<faiss::IndexPreTransform*>(idmap_index->index);
    if (pretransform)
        index = pretransform->index;

    auto device_index =dynamic_cast<faiss::gpu::GpuIndexIVF*>(index);

    if (!device_index)
        KNOWHERE_THROW_MSG("Not a GpuIndexIVF type.");

    device_index->nprobe = nprobe;
}

uint64_t
GenericGPUIVF::
LoadToGPU(){

    if (auto res = FaissGpuResourceMgr::GetInstance().GetRes(gpu_id_)) {
        ResScope rs(res, gpu_id_, false);
        auto opts = CreateFaissOpts();

        auto idmap_index = dynamic_cast<faiss::IndexIDMap2*>(index_.get());
        if (not idmap_index)
            KNOWHERE_THROW_MSG("index is not IndexIDMap2!");

        auto host_index = idmap_index->index;
        auto device_index = faiss::gpu::index_cpu_to_gpu(res->faiss_res.get(),
                                                         gpu_id_, host_index, &opts);

        idmap_index->index = device_index;
        // auto size = host_index->ntotal * host_index->d * sizeof(float);
        delete host_index;
        res_ = res;
        return 0;
    } else {
        KNOWHERE_THROW_MSG("Load error, can't get gpu resource");
    }

}


VectorIndexPtr
GenericGPUIVF::
CopyGpuToGpu(const int64_t& device_id, const Config& config, size_t& size){
    //^dont look at bs function name ^
    SetGpuDevice(device_id);
    LoadToGPU();

    return std::make_shared<GenericGPUIVF>(index_, device_id, ResPtr(res_));

}

}  // namespace knowhere
