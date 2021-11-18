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

#include "db/engine/ExecutionEngineImpl.h"
#include <faiss/IndexFlat.h>

#include <fiu-local.h>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cache/CpuCacheMgr.h"
#include "cache/GpuCacheMgr.h"
#include "knowhere/common/Config.h"
#include "metrics/Metrics.h"
#include "scheduler/Utils.h"
#include "server/config.h"
#include "utils/CommonUtil.h"
#include "utils/Exception.h"
#include "utils/Log.h"
#include "utils/ValidationUtil.h"

#include "wrapper/BinVecImpl.h"
#include "wrapper/ConfAdapter.h"
#include "wrapper/ConfAdapterMgr.h"
#include "wrapper/VecImpl.h"
#include "wrapper/VecIndex.h"

//#define ON_SEARCH
namespace milvus {
namespace engine {

namespace {

Status
MappingMetricType(MetricType metric_type, knowhere::METRICTYPE& kw_type) {
    switch (metric_type) {
        case MetricType::IP:
            kw_type = knowhere::METRICTYPE::IP;
            break;
        case MetricType::L2:
            kw_type = knowhere::METRICTYPE::L2;
            break;
        case MetricType::HAMMING:
            kw_type = knowhere::METRICTYPE::HAMMING;
            break;
        case MetricType::JACCARD:
            kw_type = knowhere::METRICTYPE::JACCARD;
            break;
        case MetricType::TANIMOTO:
            kw_type = knowhere::METRICTYPE::TANIMOTO;
            break;
        default:
            return Status(DB_ERROR, "Unsupported metric type");
    }

    return Status::OK();
}

bool
IsBinaryIndexType(IndexType type) {
    return type == IndexType::FAISS_BIN_IDMAP || type == IndexType::FAISS_BIN_IVFLAT_CPU;
}

}  // namespace

class CachedQuantizer : public cache::DataObj {
 public:
    explicit CachedQuantizer(knowhere::QuantizerPtr data) : data_(std::move(data)) {
    }

    knowhere::QuantizerPtr
    Data() {
        return data_;
    }

    int64_t
    Size() override {
        return data_->size;
    }

 private:
    knowhere::QuantizerPtr data_;
};

ExecutionEngineImpl::ExecutionEngineImpl(uint16_t dimension, const std::string& location, EngineType index_type,
                                         MetricType metric_type, int32_t nlist,
                                         const std::string& enc_type)
    : location_(location), dim_(dimension), index_type_(index_type), metric_type_(metric_type), nlist_(nlist),
      enc_type_(enc_type){
    EngineType tmp_index_type = server::ValidationUtil::IsBinaryMetricType((int32_t)metric_type)
                                    ? EngineType::FAISS_BIN_IDMAP
                                    : EngineType::FAISS_IDMAP;
    index_ = CreatetVecIndex(tmp_index_type);
    if (!index_) {
        throw Exception(DB_ERROR, "Unsupported index type");
    }

    TempMetaConf temp_conf;
    temp_conf.gpu_id = gpu_num_;
    temp_conf.dim = dimension;
    auto status = MappingMetricType(metric_type, temp_conf.metric_type);
    if (!status.ok()) {
        throw Exception(DB_ERROR, status.message());
    }

    auto adapter = AdapterMgr::GetInstance().GetAdapter(index_->GetType());
    auto conf = adapter->Match(temp_conf);
    conf->enc_type = enc_type;

    ErrorCode ec = KNOWHERE_UNEXPECTED_ERROR;
    if (auto bf_index = std::dynamic_pointer_cast<BFIndex>(index_)) {
        ec = bf_index->Build(conf);
    } else if (auto bf_bin_index = std::dynamic_pointer_cast<BinBFIndex>(index_)) {
        ec = bf_bin_index->Build(conf);
    }
    if (ec != KNOWHERE_SUCCESS) {
        throw Exception(DB_ERROR, "Build index error");
    }
}

ExecutionEngineImpl::ExecutionEngineImpl(VecIndexPtr index, const std::string& location, EngineType index_type,
                                         MetricType metric_type, int32_t nlist)
    : index_(std::move(index)), location_(location), index_type_(index_type), metric_type_(metric_type), nlist_(nlist) {
}

VecIndexPtr
ExecutionEngineImpl::CreatetVecIndex(EngineType type) {
#ifdef MILVUS_GPU_VERSION
    server::Config& config = server::Config::GetInstance();
    bool gpu_resource_enable = true;
    config.GetGpuResourceConfigEnable(gpu_resource_enable);
    fiu_do_on("ExecutionEngineImpl.CreatetVecIndex.gpu_res_disabled", gpu_resource_enable = false);
#endif

    fiu_do_on("ExecutionEngineImpl.CreatetVecIndex.invalid_type", type = EngineType::INVALID);
    std::shared_ptr<VecIndex> index;
    switch (type) {
        case EngineType::FAISS_IDMAP: {
            index = GetVecIndexFactory(IndexType::FAISS_IDMAP);
            break;
        }
        case EngineType::FAISS_FLAT: {
            index = GetVecIndexFactory(IndexType::FAISS_FLAT);
            break;
        }
        case EngineType::FAISS_GPU_FLAT_FP16: {
            index = GetVecIndexFactory(IndexType::FAISS_GPU_FLAT_FP16);
            break;
        }
        case EngineType::FAISS_IVFFLAT: {
            // if (gpu_resource_enable)
            //     index = GetVecIndexFactory(IndexType::FAISS_IVFFLAT_MIX);
            // else
            index = GetVecIndexFactory(IndexType::FAISS_IVFFLAT_CPU);
            break;
        }
        case EngineType::FAISS_GPU_IVF_FP16: {
            index = GetVecIndexFactory(IndexType::FAISS_GPU_IVF_FP16);
            break;
        }
        case EngineType::FAISS_IVFSQ8: {
#ifdef MILVUS_GPU_VERSION
            if (gpu_resource_enable)
                index = GetVecIndexFactory(IndexType::FAISS_IVFSQ8_MIX);
            else
#endif
                index = GetVecIndexFactory(IndexType::FAISS_IVFSQ8_CPU);
            break;
        }
        case EngineType::NSG_MIX: {
            index = GetVecIndexFactory(IndexType::NSG_MIX);
            break;
        }
#ifdef CUSTOMIZATION
#ifdef MILVUS_GPU_VERSION
        case EngineType::FAISS_IVFSQ8H: {
            if (gpu_resource_enable) {
                index = GetVecIndexFactory(IndexType::FAISS_IVFSQ8_HYBRID);
            } else {
                throw Exception(DB_ERROR, "No GPU resources for IVFSQ8H");
            }
            break;
        }
#endif
#endif
        case EngineType::FAISS_PQ: {
#ifdef MILVUS_GPU_VERSION
            if (gpu_resource_enable)
                index = GetVecIndexFactory(IndexType::FAISS_IVFPQ_MIX);
            else
#endif
                index = GetVecIndexFactory(IndexType::FAISS_IVFPQ_CPU);
            break;
        }
        case EngineType::SPTAG_KDT: {
            index = GetVecIndexFactory(IndexType::SPTAG_KDT_RNT_CPU);
            break;
        }
        case EngineType::SPTAG_BKT: {
            index = GetVecIndexFactory(IndexType::SPTAG_BKT_RNT_CPU);
            break;
        }
        case EngineType::HNSW: {
            index = GetVecIndexFactory(IndexType::HNSW);
            break;
        }
        case EngineType::FAISS_BIN_IDMAP: {
            index = GetVecIndexFactory(IndexType::FAISS_BIN_IDMAP);
            break;
        }
        case EngineType::FAISS_BIN_IVFFLAT: {
            index = GetVecIndexFactory(IndexType::FAISS_BIN_IVFLAT_CPU);
            break;
        }
        default: {
            ENGINE_LOG_ERROR << "Unsupported index type";
            return nullptr;
        }
    }
    return index;
}

void
ExecutionEngineImpl::HybridLoad() const {
    if (index_type_ != EngineType::FAISS_IVFSQ8H) {
        return;
    }

    if (index_->GetType() == IndexType::FAISS_IDMAP) {
        ENGINE_LOG_WARNING << "HybridLoad with type FAISS_IDMAP, ignore";
        return;
    }

#ifdef MILVUS_GPU_VERSION
    const std::string key = location_ + ".quantizer";

    server::Config& config = server::Config::GetInstance();
    std::vector<int64_t> gpus;
    Status s = config.GetGpuResourceConfigSearchResources(gpus);
    if (!s.ok()) {
        ENGINE_LOG_ERROR << s.message();
        return;
    }

    // cache hit
    {
        const int64_t NOT_FOUND = -1;
        int64_t device_id = NOT_FOUND;
        knowhere::QuantizerPtr quantizer = nullptr;

        for (auto& gpu : gpus) {
            auto cache = cache::GpuCacheMgr::GetInstance(gpu);
            if (auto cached_quantizer = cache->GetIndex(key)) {
                device_id = gpu;
                quantizer = std::static_pointer_cast<CachedQuantizer>(cached_quantizer)->Data();
            }
        }

        if (device_id != NOT_FOUND) {
            index_->SetQuantizer(quantizer);
            return;
        }
    }

    // cache miss
    {
        std::vector<int64_t> all_free_mem;
        for (auto& gpu : gpus) {
            auto cache = cache::GpuCacheMgr::GetInstance(gpu);
            auto free_mem = cache->CacheCapacity() - cache->CacheUsage();
            all_free_mem.push_back(free_mem);
        }

        auto max_e = std::max_element(all_free_mem.begin(), all_free_mem.end());
        auto best_index = std::distance(all_free_mem.begin(), max_e);
        auto best_device_id = gpus[best_index];

        auto quantizer_conf = std::make_shared<knowhere::QuantizerCfg>();
        quantizer_conf->mode = 1;
        quantizer_conf->gpu_id = best_device_id;
        auto quantizer = index_->LoadQuantizer(quantizer_conf);
        if (quantizer == nullptr) {
            ENGINE_LOG_ERROR << "quantizer is nullptr";
        }
        index_->SetQuantizer(quantizer);
        auto cache_quantizer = std::make_shared<CachedQuantizer>(quantizer);
        cache::GpuCacheMgr::GetInstance(best_device_id)->InsertItem(key, cache_quantizer);
    }
#endif
}

void
ExecutionEngineImpl::HybridUnset() const {
    if (index_type_ != EngineType::FAISS_IVFSQ8H) {
        return;
    }
    if (index_->GetType() == IndexType::FAISS_IDMAP) {
        return;
    }
    index_->UnsetQuantizer();
}

Status
ExecutionEngineImpl::Reconstruct(std::vector<int64_t> ids, std::vector<float>& xb,
                                 std::vector<bool>& found) {
    auto status = index_->Reconstruct(ids, xb, found);
    return status;
}

Status
ExecutionEngineImpl::GetIds(std::vector<int64_t>& ids){
    return index_->GetIds(ids);
}


Status
ExecutionEngineImpl::AddWithIds(int64_t n, const float* xdata, const int64_t* xids) {
    auto status = index_->Add(n, xdata, xids);
    return status;
}

Status
ExecutionEngineImpl::AddWithIds(int64_t n, const uint8_t* xdata, const int64_t* xids) {
    auto status = index_->Add(n, xdata, xids);
    return status;
}

size_t
ExecutionEngineImpl::Count() const {
    if (index_ == nullptr) {
        ENGINE_LOG_ERROR << "ExecutionEngineImpl: index is null, return count 0";
        return 0;
    }
    return index_->Count();
}

size_t
ExecutionEngineImpl::Size() const {
    if (IsBinaryIndexType(index_->GetType())) {
        return (size_t)(Count() * Dimension() / 8);
    } else {
        return (size_t)(Count() * Dimension()) * sizeof(float);
    }
}

size_t
ExecutionEngineImpl::Dimension() const {
    if (index_ == nullptr) {
        ENGINE_LOG_ERROR << "ExecutionEngineImpl: index is null, return dimension " << dim_;
        return dim_;
    }
    return index_->Dimension();
}

size_t
ExecutionEngineImpl::PhysicalSize() const {
    return server::CommonUtil::GetFileSize(location_);
}

Status
ExecutionEngineImpl::Serialize() {
    auto status = write_index(index_, location_);

    // here we reset index size by file size,
    // since some index type(such as SQ8) data size become smaller after serialized
    index_->set_size(PhysicalSize());
    ENGINE_LOG_DEBUG << "Finish serialize index file: " << location_ << " size: " << index_->Size();

    if (index_->Size() == 0) {
        std::string msg = "Failed to serialize file: " + location_ + " reason: out of disk space or memory";
        status = Status(DB_ERROR, msg);
    }

    return status;
}

Status
ExecutionEngineImpl::Load(bool to_cache, bool force) {
    //LOL
    auto index_type = CreatetVecIndex(index_type_)->GetType();
    if (force)
        index_ = nullptr;
    else
        index_ = std::static_pointer_cast<VecIndex>(cache::CpuCacheMgr::GetInstance()->GetIndex(location_));
    bool already_in_cache = (index_ != nullptr);
    if (!already_in_cache) {
        try {
            double physical_size = PhysicalSize();
            server::CollectExecutionEngineMetrics metrics(physical_size);

            index_ = read_index(index_type, location_);
            index_->set_size(physical_size);
            if (index_ == nullptr) {
                std::string msg = "Failed to load index from " + location_;
                ENGINE_LOG_ERROR << msg;
                return Status(DB_ERROR, msg);
            } else {
                ENGINE_LOG_DEBUG << "Disk io from: " << location_;
            }
        } catch (std::exception& e) {
            ENGINE_LOG_ERROR << e.what();
            return Status(DB_ERROR, e.what());
        }
    }

    if (!already_in_cache && to_cache) {
        Cache();
    }
    return Status::OK();
}

Status
ExecutionEngineImpl::CopyToGpu(uint64_t device_id, bool hybrid) {
#if 0
    if (hybrid) {
        const std::string key = location_ + ".quantizer";
        std::vector<uint64_t> gpus{device_id};

        const int64_t NOT_FOUND = -1;
        int64_t device_id = NOT_FOUND;

        // cache hit
        {
            knowhere::QuantizerPtr quantizer = nullptr;

            for (auto& gpu : gpus) {
                auto cache = cache::GpuCacheMgr::GetInstance(gpu);
                if (auto cached_quantizer = cache->GetIndex(key)) {
                    device_id = gpu;
                    quantizer = std::static_pointer_cast<CachedQuantizer>(cached_quantizer)->Data();
                }
            }

            if (device_id != NOT_FOUND) {
                // cache hit
                auto config = std::make_shared<knowhere::QuantizerCfg>();
                config->gpu_id = device_id;
                config->mode = 2;
                auto new_index = index_->LoadData(quantizer, config);
                index_ = new_index;
            }
        }

        if (device_id == NOT_FOUND) {
            // cache miss
            std::vector<int64_t> all_free_mem;
            for (auto& gpu : gpus) {
                auto cache = cache::GpuCacheMgr::GetInstance(gpu);
                auto free_mem = cache->CacheCapacity() - cache->CacheUsage();
                all_free_mem.push_back(free_mem);
            }

            auto max_e = std::max_element(all_free_mem.begin(), all_free_mem.end());
            auto best_index = std::distance(all_free_mem.begin(), max_e);
            device_id = gpus[best_index];

            auto pair = index_->CopyToGpuWithQuantizer(device_id);
            index_ = pair.first;

            // cache
            auto cached_quantizer = std::make_shared<CachedQuantizer>(pair.second);
            cache::GpuCacheMgr::GetInstance(device_id)->InsertItem(key, cached_quantizer);
        }
        return Status::OK();
    }
#endif

#ifdef MILVUS_GPU_VERSION
    auto index = std::static_pointer_cast<VecIndex>(cache::GpuCacheMgr::GetInstance(device_id)->GetIndex(location_));
    bool already_in_cache = (index != nullptr);
    if (already_in_cache) {
        index_ = index;
    } else {
        if (index_ == nullptr) {
            ENGINE_LOG_ERROR << "ExecutionEngineImpl: index is null, failed to copy to gpu";
            return Status(DB_ERROR, "index is null");
        }

        try {
            ENGINE_LOG_DEBUG << "ExecutionEngineImpl: Loading index but do not add to CPU cache";
            Load(false, true);
            index_ = index_->CopyToGpu(device_id);
            ENGINE_LOG_DEBUG << "CPU to GPU" << device_id;
        } catch (std::exception& e) {
            ENGINE_LOG_ERROR << e.what();
            return Status(DB_ERROR, e.what());
        }
    }

    if (!already_in_cache) {
        GpuCache(device_id);
    }
#endif

    return Status::OK();
}

Status
ExecutionEngineImpl::CopyToIndexFileToGpu(uint64_t device_id) {
#ifdef MILVUS_GPU_VERSION
    // the ToIndexData is only a placeholder, cpu-copy-to-gpu action is performed in
    gpu_num_ = device_id;
    auto to_index_data = std::make_shared<ToIndexData>(PhysicalSize());
    cache::DataObjPtr obj = std::static_pointer_cast<cache::DataObj>(to_index_data);
    milvus::cache::GpuCacheMgr::GetInstance(device_id)->InsertItem(location_ + "_placeholder", obj);
#endif
    return Status::OK();
}

Status
ExecutionEngineImpl::CopyToCpu() {
    auto index = std::static_pointer_cast<VecIndex>(cache::CpuCacheMgr::GetInstance()->GetIndex(location_));
    bool already_in_cache = (index != nullptr);
    if (already_in_cache) {
        index_ = index;
    } else {
        if (index_ == nullptr) {
            ENGINE_LOG_ERROR << "ExecutionEngineImpl: index is null, failed to copy to cpu";
            return Status(DB_ERROR, "index is null");
        }

        try {
            index_ = index_->CopyToCpu();
            ENGINE_LOG_DEBUG << "GPU to CPU";
        } catch (std::exception& e) {
            ENGINE_LOG_ERROR << e.what();
            return Status(DB_ERROR, e.what());
        }
    }

    if (!already_in_cache) {
        Cache();
    }
    return Status::OK();
}

// ExecutionEnginePtr
// ExecutionEngineImpl::Clone() {
//    if (index_ == nullptr) {
//        ENGINE_LOG_ERROR << "ExecutionEngineImpl: index is null, failed to clone";
//        return nullptr;
//    }
//
//    auto ret = std::make_shared<ExecutionEngineImpl>(dim_, location_, index_type_, metric_type_, nlist_);
//    ret->Init();
//    ret->index_ = index_->Clone();
//    return ret;
//}


Status
ExecutionEngineImpl::Reserve(uint64_t bytes, uint64_t vec_cnt){
    return index_->Reserve(bytes, vec_cnt);
}

faiss::IndexIDMap2*
ExecutionEngineImpl::
GetFaissIndex(){
    return index_->GetFaissIndex();
}


Status
ExecutionEngineImpl::Merge(const std::string& location) {
    if (location == location_) {
        return Status(DB_ERROR, "Cannot Merge Self");
    }
    ENGINE_LOG_DEBUG << "Merge index file: " << location << " to: " << location_;

    auto to_merge = cache::CpuCacheMgr::GetInstance()->GetIndex(location);
    if (!to_merge) {
        try {
            double physical_size = server::CommonUtil::GetFileSize(location);
            server::CollectExecutionEngineMetrics metrics(physical_size);
            auto index = read_index(index_->GetType(), location);
            index->set_size(physical_size);
            to_merge = index;
        } catch (std::exception& e) {
            ENGINE_LOG_ERROR << e.what();
            return Status(DB_ERROR, e.what());
        }
    }

    if (index_ == nullptr) {
        ENGINE_LOG_ERROR << "ExecutionEngineImpl: index is null, failed to merge";
        return Status(DB_ERROR, "index is null");
    }

    if (auto file_index = std::dynamic_pointer_cast<BFIndex>(to_merge)) {
        auto status = index_->Add(file_index->Count(), file_index->GetRawVectors(), file_index->GetRawIds());
        if (!status.ok()) {
            ENGINE_LOG_ERROR << "Failed to merge: " << location << " to: " << location_;
        } else {
            ENGINE_LOG_DEBUG << "Finish merge index file: " << location;
        }
        return status;
    } else if (auto bin_index = std::dynamic_pointer_cast<BinBFIndex>(to_merge)) {
        auto status = index_->Add(bin_index->Count(), bin_index->GetRawVectors(), bin_index->GetRawIds());
        if (!status.ok()) {
            ENGINE_LOG_ERROR << "Failed to merge: " << location << " to: " << location_;
        } else {
            ENGINE_LOG_DEBUG << "Finish merge index file: " << location;
        }
        return status;
    } else {
        return Status(DB_ERROR, "file index type is not idmap");
    }
}

ExecutionEnginePtr
ExecutionEngineImpl::BuildIndex(const std::string& location, EngineType engine_type) {
    ENGINE_LOG_DEBUG << "Build index file: " << location << " from: " << location_;

    auto idmap_index = index_->GetFaissIndex();
    if (idmap_index == nullptr ) {
        ENGINE_LOG_ERROR << "ExecutionEngineImpl: idmap is null, failed to build index";
        return nullptr;
    }
    auto flat_index = dynamic_cast<faiss::IndexFlat*>(idmap_index->index);
    if (flat_index == nullptr ) {
        ENGINE_LOG_ERROR << "ExecutionEngineImpl: flat_index is null, failed to build index";
        return nullptr;
    }

    auto to_index = CreatetVecIndex(engine_type);
    if (!to_index) {
        throw Exception(DB_ERROR, "Unsupported index type");
    }

    TempMetaConf temp_conf;
    temp_conf.gpu_id = gpu_num_;
    temp_conf.dim = Dimension();
    temp_conf.nlist = nlist_;
    temp_conf.size = Count();
    auto status = MappingMetricType(metric_type_, temp_conf.metric_type);
    if (!status.ok()) {
        throw Exception(DB_ERROR, status.message());
    }

    auto adapter = AdapterMgr::GetInstance().GetAdapter(to_index->GetType());
    auto conf = adapter->Match(temp_conf);

    conf->enc_type = enc_type_;

    status = to_index->BuildAll(Count(), flat_index->xb.data(), idmap_index->id_map.data(), conf);

    if (!status.ok()) {
        throw Exception(DB_ERROR, status.message());
    }

    ENGINE_LOG_DEBUG << "Finish build index file: " << location << " size: " << to_index->Size();
    return std::make_shared<ExecutionEngineImpl>(to_index, location, engine_type, metric_type_, nlist_);
}

Status
ExecutionEngineImpl::Search(int64_t n, const float* data, int64_t k, int64_t nprobe, float* distances, int64_t* labels,
                            bool hybrid) {
#if 0
    if (index_type_ == EngineType::FAISS_IVFSQ8H) {
        if (!hybrid) {
            const std::string key = location_ + ".quantizer";
            std::vector<uint64_t> gpus = scheduler::get_gpu_pool();

            const int64_t NOT_FOUND = -1;
            int64_t device_id = NOT_FOUND;

            // cache hit
            {
                knowhere::QuantizerPtr quantizer = nullptr;

                for (auto& gpu : gpus) {
                    auto cache = cache::GpuCacheMgr::GetInstance(gpu);
                    if (auto cached_quantizer = cache->GetIndex(key)) {
                        device_id = gpu;
                        quantizer = std::static_pointer_cast<CachedQuantizer>(cached_quantizer)->Data();
                    }
                }

                if (device_id != NOT_FOUND) {
                    // cache hit
                    auto config = std::make_shared<knowhere::QuantizerCfg>();
                    config->gpu_id = device_id;
                    config->mode = 2;
                    auto new_index = index_->LoadData(quantizer, config);
                    index_ = new_index;
                }
            }

            if (device_id == NOT_FOUND) {
                // cache miss
                std::vector<int64_t> all_free_mem;
                for (auto& gpu : gpus) {
                    auto cache = cache::GpuCacheMgr::GetInstance(gpu);
                    auto free_mem = cache->CacheCapacity() - cache->CacheUsage();
                    all_free_mem.push_back(free_mem);
                }

                auto max_e = std::max_element(all_free_mem.begin(), all_free_mem.end());
                auto best_index = std::distance(all_free_mem.begin(), max_e);
                device_id = gpus[best_index];

                auto pair = index_->CopyToGpuWithQuantizer(device_id);
                index_ = pair.first;

                // cache
                auto cached_quantizer = std::make_shared<CachedQuantizer>(pair.second);
                cache::GpuCacheMgr::GetInstance(device_id)->InsertItem(key, cached_quantizer);
            }
        }
    }
#endif

    if (index_ == nullptr) {
        ENGINE_LOG_ERROR << "ExecutionEngineImpl: index is null, failed to search";
        return Status(DB_ERROR, "index is null");
    }

    ENGINE_LOG_DEBUG << "Search Params: [k]  " << k << " [nprobe] " << nprobe;

    // TODO(linxj): remove here. Get conf from function
    TempMetaConf temp_conf;
    temp_conf.k = k;
    temp_conf.nprobe = nprobe;

    auto adapter = AdapterMgr::GetInstance().GetAdapter(index_->GetType());
    auto conf = adapter->MatchSearch(temp_conf, index_->GetType());

    if (hybrid) {
        HybridLoad();
    }

    auto status = index_->Search(n, data, distances, labels, conf);

    if (hybrid) {
        HybridUnset();
    }

    if (!status.ok()) {
        ENGINE_LOG_ERROR << "Search error:" << status.message();
    }
    return status;
}

Status
ExecutionEngineImpl::Search(int64_t n, const uint8_t* data, int64_t k, int64_t nprobe, float* distances,
                            int64_t* labels, bool hybrid) {
    if (index_ == nullptr) {
        ENGINE_LOG_ERROR << "ExecutionEngineImpl: index is null, failed to search";
        return Status(DB_ERROR, "index is null");
    }

    ENGINE_LOG_DEBUG << "Search Params: [k]  " << k << " [nprobe] " << nprobe;

    // TODO(linxj): remove here. Get conf from function
    TempMetaConf temp_conf;
    temp_conf.k = k;
    temp_conf.nprobe = nprobe;

    auto adapter = AdapterMgr::GetInstance().GetAdapter(index_->GetType());
    auto conf = adapter->MatchSearch(temp_conf, index_->GetType());

    if (hybrid) {
        HybridLoad();
    }

    auto status = index_->Search(n, data, distances, labels, conf);

    if (hybrid) {
        HybridUnset();
    }

    if (!status.ok()) {
        ENGINE_LOG_ERROR << "Search error:" << status.message();
    }
    return status;
}

Status
ExecutionEngineImpl::Cache() {
    cache::DataObjPtr obj = std::static_pointer_cast<cache::DataObj>(index_);
    milvus::cache::CpuCacheMgr::GetInstance()->InsertItem(location_, obj);

    return Status::OK();
}

Status
ExecutionEngineImpl::GpuCache(uint64_t gpu_id) {
#ifdef MILVUS_GPU_VERSION
    cache::DataObjPtr obj = std::static_pointer_cast<cache::DataObj>(index_);
    milvus::cache::GpuCacheMgr::GetInstance(gpu_id)->InsertItem(location_, obj);
#endif
    return Status::OK();
}

// TODO(linxj): remove.
Status
ExecutionEngineImpl::Init() {
#ifdef MILVUS_GPU_VERSION
    server::Config& config = server::Config::GetInstance();
    std::vector<int64_t> gpu_ids;
    Status s = config.GetGpuResourceConfigBuildIndexResources(gpu_ids);
    if (!s.ok() or gpu_ids.empty()) {
        gpu_num_ = knowhere::INVALID_VALUE;
        return s;
    }
    gpu_num_ = gpu_ids.front();
    return Status::OK();

#else
    return Status::OK();
#endif
}

}  // namespace engine
}  // namespace milvus
