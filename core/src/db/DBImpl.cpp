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

#include "db/DBImpl.h"

#include <assert.h>

#include <algorithm>
#include <boost/filesystem.hpp>
#include <chrono>
#include <cstring>
#include <iostream>
#include <set>
#include <thread>
#include <utility>

#include <faiss/gpu/GpuDistance.h>
#include <faiss/IndexIVF.h>


#include "Utils.h"
#include "cache/CpuCacheMgr.h"
#include "cache/GpuCacheMgr.h"
#include "engine/EngineFactory.h"
#include "insert/MemMenagerFactory.h"
#include "meta/MetaConsts.h"
#include "meta/MetaFactory.h"
#include "meta/SqliteMetaImpl.h"
#include "metrics/Metrics.h"
#include "scheduler/SchedInst.h"
#include "scheduler/job/BuildIndexJob.h"
#include "scheduler/job/DeleteJob.h"
#include "scheduler/job/SearchJob.h"
#include "scheduler/job/ClusterizeJob.h"
#include "utils/Log.h"
#include "utils/StringHelpFunctions.h"
#include "utils/TimeRecorder.h"

#include "knowhere/index/vector_index/helpers/FaissGpuResourceMgr.h"
#include "knowhere/index/vector_index/IndexIVF.h"


namespace milvus {
namespace engine {

namespace {

constexpr uint64_t METRIC_ACTION_INTERVAL = 1;
constexpr uint64_t COMPACT_ACTION_INTERVAL = 1;
constexpr uint64_t INDEX_ACTION_INTERVAL = 1;

static const Status SHUTDOWN_ERROR = Status(DB_ERROR, "Milvus server is shutdown!");

void
TraverseFiles(const meta::DatePartionedTableFilesSchema& date_files, meta::TableFilesSchema& files_array) {
    for (auto& day_files : date_files) {
        for (auto& file : day_files.second) {
            files_array.push_back(file);
        }
    }
}

}  // namespace

DBImpl::DBImpl(const DBOptions& options)
    : options_(options), initialized_(false), compact_thread_pool_(1, 1), index_thread_pool_(1, 1) {
    meta_ptr_ = MetaFactory::Build(options.meta_, options.mode_);
    mem_mgr_ = MemManagerFactory::Build(meta_ptr_, options_);
    Start();
}

DBImpl::~DBImpl() {
    Stop();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// external api
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Status
DBImpl::Start() {
    if (initialized_.load(std::memory_order_acquire)) {
        return Status::OK();
    }

    // ENGINE_LOG_TRACE << "DB service start";
    initialized_.store(true, std::memory_order_release);

    // for distribute version, some nodes are read only
    if (options_.mode_ != DBOptions::MODE::CLUSTER_READONLY) {
        // ENGINE_LOG_TRACE << "StartTimerTasks";
        bg_timer_thread_ = std::thread(&DBImpl::BackgroundTimerTask, this);
    }

    return Status::OK();
}

Status
DBImpl::Stop() {
    if (!initialized_.load(std::memory_order_acquire)) {
        return Status::OK();
    }
    initialized_.store(false, std::memory_order_release);

    // makesure all memory data serialized
    std::set<std::string> sync_table_ids;
    SyncMemData(sync_table_ids);

    // wait compaction/buildindex finish
    bg_timer_thread_.join();

    if (options_.mode_ != DBOptions::MODE::CLUSTER_READONLY) {
        meta_ptr_->CleanUpShadowFiles();
    }

    // ENGINE_LOG_TRACE << "DB service stop";
    return Status::OK();
}

Status
DBImpl::DropAll() {
    return meta_ptr_->DropAll();
}

Status
DBImpl::CreateTable(meta::TableSchema& table_schema) {
    if (!initialized_.load(std::memory_order_acquire)) {
        return SHUTDOWN_ERROR;
    }

    meta::TableSchema temp_schema = table_schema;
    temp_schema.index_file_size_ *= ONE_MB;  // store as MB
    return meta_ptr_->CreateTable(temp_schema);
}

Status
DBImpl::DropTable(const std::string& table_id, const meta::DatesT& dates) {
    if (!initialized_.load(std::memory_order_acquire)) {
        return SHUTDOWN_ERROR;
    }

    return DropTableRecursively(table_id, dates);
}

Status
DBImpl::DescribeTable(meta::TableSchema& table_schema) {
    if (!initialized_.load(std::memory_order_acquire)) {
        return SHUTDOWN_ERROR;
    }

    auto stat = meta_ptr_->DescribeTable(table_schema);
    table_schema.index_file_size_ /= ONE_MB;  // return as MB
    return stat;
}

Status
DBImpl::HasTable(const std::string& table_id, bool& has_or_not) {
    if (!initialized_.load(std::memory_order_acquire)) {
        return SHUTDOWN_ERROR;
    }

    return meta_ptr_->HasTable(table_id, has_or_not);
}

Status
DBImpl::AllTables(std::vector<meta::TableSchema>& table_schema_array) {
    if (!initialized_.load(std::memory_order_acquire)) {
        return SHUTDOWN_ERROR;
    }

    std::vector<meta::TableSchema> all_tables;
    auto status = meta_ptr_->AllTables(all_tables);

    // only return real tables, dont return partition tables
    table_schema_array.clear();
    for (auto& schema : all_tables) {
        if (schema.owner_table_.empty()) {
            table_schema_array.push_back(schema);
        }
    }

    return status;
}

Status
DBImpl::PreloadTable(const std::string& table_id) {
    if (!initialized_.load(std::memory_order_acquire)) {
        return SHUTDOWN_ERROR;
    }

    // step 1: get all table files from parent table
    meta::DatesT dates;
    std::vector<size_t> ids;
    meta::TableFilesSchema files_array;
    auto status = GetFilesToSearch(table_id, ids, dates, files_array);
    if (!status.ok()) {
        return status;
    }

    // step 2: get files from partition tables
    std::vector<meta::TableSchema> partition_array;
    status = meta_ptr_->ShowPartitions(table_id, partition_array);
    for (auto& schema : partition_array) {
        status = GetFilesToSearch(schema.table_id_, ids, dates, files_array);
    }

    int64_t size = 0;
    int64_t cache_total = cache::CpuCacheMgr::GetInstance()->CacheCapacity();
    int64_t cache_usage = cache::CpuCacheMgr::GetInstance()->CacheUsage();
    int64_t available_size = cache_total - cache_usage;

    // step 3: load file one by one
    ENGINE_LOG_DEBUG << "Begin pre-load table:" + table_id + ", totally " << files_array.size()
                     << " files need to be pre-loaded";
    TimeRecorderAuto rc("Pre-load table:" + table_id);
    for (auto& file : files_array) {
        ExecutionEnginePtr engine = EngineFactory::Build(file.dimension_, file.location_, (EngineType)file.engine_type_,
                                                         (MetricType)file.metric_type_, file.nlist_,
                                                         file.enc_type_);
        fiu_do_on("DBImpl.PreloadTable.null_engine", engine = nullptr);
        if (engine == nullptr) {
            ENGINE_LOG_ERROR << "Invalid engine type";
            return Status(DB_ERROR, "Invalid engine type");
        }

        size += engine->PhysicalSize();
        fiu_do_on("DBImpl.PreloadTable.exceed_cache", size = available_size + 1);
        if (size > available_size) {
            ENGINE_LOG_DEBUG << "Pre-load canceled since cache almost full";
            return Status(SERVER_CACHE_FULL, "Cache is full");
        } else {
            try {
                fiu_do_on("DBImpl.PreloadTable.engine_throw_exception", throw std::exception());
                std::string msg = "Pre-loaded file: " + file.file_id_ + " size: " + std::to_string(file.file_size_);
                TimeRecorderAuto rc_1(msg);
                engine->Load(true);
            } catch (std::exception& ex) {
                std::string msg = "Pre-load table encounter exception: " + std::string(ex.what());
                ENGINE_LOG_ERROR << msg;
                return Status(DB_ERROR, msg);
            }
        }
    }

    return Status::OK();
}

Status
DBImpl::UpdateTableFlag(const std::string& table_id, int64_t flag) {
    if (!initialized_.load(std::memory_order_acquire)) {
        return SHUTDOWN_ERROR;
    }

    return meta_ptr_->UpdateTableFlag(table_id, flag);
}

Status
DBImpl::GetTableRowCount(const std::string& table_id, uint64_t& row_count) {
    if (!initialized_.load(std::memory_order_acquire)) {
        return SHUTDOWN_ERROR;
    }

    return GetTableRowCountRecursively(table_id, row_count);
}

Status
DBImpl::CreatePartition(const std::string& table_id, const std::string& partition_name,
                        const std::string& partition_tag) {
    if (!initialized_.load(std::memory_order_acquire)) {
        return SHUTDOWN_ERROR;
    }

    return meta_ptr_->CreatePartition(table_id, partition_name, partition_tag);
}

Status
DBImpl::DropPartition(const std::string& partition_name) {
    if (!initialized_.load(std::memory_order_acquire)) {
        return SHUTDOWN_ERROR;
    }

    auto status = mem_mgr_->EraseMemVector(partition_name);  // not allow insert
    status = meta_ptr_->DropPartition(partition_name);       // soft delete table

    // scheduler will determine when to delete table files
    auto nres = scheduler::ResMgrInst::GetInstance()->GetNumOfComputeResource();
    scheduler::DeleteJobPtr job = std::make_shared<scheduler::DeleteJob>(partition_name, meta_ptr_, nres);
    scheduler::JobMgrInst::GetInstance()->Put(job);
    job->WaitAndDelete();

    return Status::OK();
}

Status
DBImpl::DropPartitionByTag(const std::string& table_id, const std::string& partition_tag) {
    if (!initialized_.load(std::memory_order_acquire)) {
        return SHUTDOWN_ERROR;
    }

    std::string partition_name;
    auto status = meta_ptr_->GetPartitionName(table_id, partition_tag, partition_name);
    if (!status.ok()) {
        ENGINE_LOG_ERROR << status.message();
        return status;
    }

    return DropPartition(partition_name);
}

Status
DBImpl::ShowPartitions(const std::string& table_id, std::vector<meta::TableSchema>& partition_schema_array) {
    if (!initialized_.load(std::memory_order_acquire)) {
        return SHUTDOWN_ERROR;
    }

    return meta_ptr_->ShowPartitions(table_id, partition_schema_array);
}

Status
DBImpl::InsertVectors(const std::string& table_id, const std::string& partition_tag, VectorsData& vectors) {
    //    ENGINE_LOG_DEBUG << "Insert " << n << " vectors to cache";
    if (!initialized_.load(std::memory_order_acquire)) {
        return SHUTDOWN_ERROR;
    }

    // if partition is specified, use partition as target table
    Status status;
    std::string target_table_name = table_id;
    if (!partition_tag.empty()) {
        std::string partition_name;
        status = meta_ptr_->GetPartitionName(table_id, partition_tag, target_table_name);
        if (!status.ok()) {
            ENGINE_LOG_ERROR << status.message();
            return status;
        }
    }

    // insert vectors into target table
    milvus::server::CollectInsertMetrics metrics(vectors.vector_count_, status);
    status = mem_mgr_->InsertVectors(target_table_name, vectors);

    return status;
}

Status
DBImpl::CreateIndex(const std::string& table_id, const TableIndex& index) {
    if (!initialized_.load(std::memory_order_acquire)) {
        return SHUTDOWN_ERROR;
    }

    // serialize memory data
    std::set<std::string> sync_table_ids;
    auto status = SyncMemData(sync_table_ids);

    {
        std::unique_lock<std::mutex> lock(build_index_mutex_);

        // step 1: check index difference
        TableIndex old_index;
        status = DescribeIndex(table_id, old_index);
        if (!status.ok()) {
            ENGINE_LOG_ERROR << "Failed to get table index info for table: " << table_id;
            return status;
        }

        // step 2: update index info
        TableIndex new_index = index;
        new_index.metric_type_ = old_index.metric_type_;  // dont change metric type, it was defined by CreateTable
        if (!utils::IsSameIndex(old_index, new_index)) {
            status = UpdateTableIndexRecursively(table_id, new_index);
            if (!status.ok()) {
                return status;
            }
        }
    }

    // step 3: let merge file thread finish
    // to avoid duplicate data bug
    WaitMergeFileFinish();

    // step 4: wait and build index
    status = index_failed_checker_.CleanFailedIndexFileOfTable(table_id);
    status = BuildTableIndexRecursively(table_id, index);

    return status;
}

Status
DBImpl::DescribeIndex(const std::string& table_id, TableIndex& index) {
    if (!initialized_.load(std::memory_order_acquire)) {
        return SHUTDOWN_ERROR;
    }

    return meta_ptr_->DescribeTableIndex(table_id, index);
}

Status
DBImpl::DropIndex(const std::string& table_id) {
    if (!initialized_.load(std::memory_order_acquire)) {
        return SHUTDOWN_ERROR;
    }

    ENGINE_LOG_DEBUG << "Drop index for table: " << table_id;
    return DropTableIndexRecursively(table_id);
}

Status
DBImpl::Clusterize(const std::shared_ptr<server::Context> &context,
                   const ClusterizeOptions &opts,
                   const VectorsData &vectors){
    Status status;
    meta::TableFilesSchema direct_files;
    ENGINE_LOG_DEBUG <<"Input vectors count: "<<vectors.vector_count_
                     <<" Input float data size: " << vectors.float_data_.size()
                     <<" Input ids size: "<<vectors.id_array_.size()
                     <<" Query tables size: "<<vectors.query_table_ids.size();


    if (vectors.float_data_.empty()){

        std::vector<size_t> ids;
        meta::DatesT dates;
        for(const auto& table_id : vectors.query_table_ids){
            meta::TableFilesSchema temp;
            status = GetDirectFiles(table_id, ids, dates, temp);
            if (!status.ok()) {
                return status;
            }
            direct_files.insert(direct_files.end(), temp.begin(), temp.end());
        }

        ENGINE_LOG_DEBUG << "Engine clustering begin, direct files count: " << direct_files.size();
        auto status = ongoing_files_checker_.MarkOngoingFiles(direct_files);
    }

    auto job = std::make_shared<scheduler::ClusterizeJob>(context, opts, meta_ptr_,
                                                          direct_files, vectors);
    scheduler::JobMgrInst::GetInstance()->Put(job);
    job->WaitResult();

    if(not direct_files.empty())
        status = ongoing_files_checker_.UnmarkOngoingFiles(direct_files);

    if (!job->GetStatus().ok()) {
        return job->GetStatus();
    }

    return Status::OK();

}

Status
DBImpl::GetVectors(const std::shared_ptr<server::Context>& context,
                   const std::vector<std::string>& table_names,
                   VectorsData& vectors){

    auto tables = std::accumulate(std::begin(table_names), std::end(table_names), std::string{},
                                  [](auto& t, const auto& s) {return t += "," + s;});
    ENGINE_LOG_DEBUG << "GetVectors tables: " << tables
                     << " vecors #: " << vectors.id_array_.size();

    Status status;
    meta::TableFilesSchema direct_files;

    meta::DatesT dates;
    std::vector<size_t> ids;

    for(const auto& table_id : table_names){
        meta::TableFilesSchema temp;
        status = GetDirectFiles(table_id, ids, dates, temp);
        if (!status.ok()) {
            return status;
        }

        direct_files.insert(direct_files.end(), temp.begin(), temp.end());
    }


    std::vector<bool> found_query_ids;
    vectors.vector_count_ = LoadVectors(vectors.id_array_, direct_files, found_query_ids,
                                        const_cast<std::vector<float>&>(vectors.float_data_));

    for (int i = 0; i < vectors.id_array_.size(); ++i)
        if (not found_query_ids[i])
            vectors.id_array_[i] = 0;
    return Status::OK();
}

Status
DBImpl::Query(const std::shared_ptr<server::Context>& context,
              const std::vector<std::string>& table_names,
              const std::vector<std::string>& partition_tags, uint64_t k, uint64_t nprobe, const VectorsData& vectors,
              ResultIds& result_ids, ResultDistances& result_distances) {
    if (!initialized_.load(std::memory_order_acquire)) {
        return SHUTDOWN_ERROR;
    }

    // meta::DatesT dates = {utils::GetDate()};
    meta::DatesT dates;
    Status result = Query(context, table_names, partition_tags, k, nprobe, vectors, dates, result_ids, result_distances);
    return result;
}

Status
DBImpl::Query(const std::shared_ptr<server::Context>& context,
              const std::vector<std::string>& table_names,
              const std::vector<std::string>& partition_tags, uint64_t k, uint64_t nprobe,
              const VectorsData& vectors, const meta::DatesT& dates,
              ResultIds& result_ids, ResultDistances& result_distances) {
    auto query_ctx = context->Child("Query");

    if (!initialized_.load(std::memory_order_acquire)) {
        return SHUTDOWN_ERROR;
    }

    // ENGINE_LOG_DEBUG << "Query by dates for table: " << table_id << " date range count: " << dates.size();

    Status status;
    std::vector<size_t> ids;
    meta::TableFilesSchema files_array;
    meta::TableFilesSchema direct_files;

    if (partition_tags.empty()) {
        // no partition tag specified, means search in whole table
        // get all table files from parent table
        for(const auto& table_id : table_names){
            meta::TableFilesSchema temp;
            status = GetFilesToSearch(table_id, ids, dates, temp);
            if (!status.ok()) {
                return status;
            }

            files_array.insert(files_array.end(), temp.begin(), temp.end());
        }

        if (!vectors.id_array_.empty()){

            auto* tables = &table_names;
            if (not vectors.query_table_ids.empty())
                tables = &vectors.query_table_ids;
            for(const auto& table_id : vectors.query_table_ids){
                meta::TableFilesSchema temp;
                status = GetDirectFiles(table_id, ids, dates, temp);
                if (!status.ok()) {
                    return status;
                }

                direct_files.insert(direct_files.end(), temp.begin(), temp.end());
            }

        }

        //TODO for direct??
        std::vector<meta::TableSchema> partition_array;
        //TEMP
        status = meta_ptr_->ShowPartitions(table_names.front(), partition_array);
        for (auto& schema : partition_array) {
            status = GetFilesToSearch(schema.table_id_, ids, dates, files_array);
        }
    } else {
        // get files from specified partitions
        std::set<std::string> partition_name_array;
        GetPartitionsByTags(table_names.front(), partition_tags, partition_name_array);

        for (auto& partition_name : partition_name_array) {
            status = GetFilesToSearch(partition_name, ids, dates, files_array);
        }
    }

    ENGINE_LOG_DEBUG << "CPU cache info";
    cache::CpuCacheMgr::GetInstance()->PrintInfo();  // print cache info before query
    // ENGINE_LOG_DEBUG << "GPU cache info 0";
    // cache::GpuCacheMgr::GetInstance(0)->PrintInfo();  // print cache info before query
    // ENGINE_LOG_DEBUG << "GPU cache info 1";
    // cache::GpuCacheMgr::GetInstance(1)->PrintInfo();  // print cache info before query

    std::vector<bool> found_query_ids;
    LoadVectors(vectors.id_array_, direct_files, found_query_ids,
                const_cast<std::vector<float>&>(vectors.float_data_));


    status = QueryAsync(query_ctx, table_names.front(), files_array, k, nprobe,
                        vectors, result_ids, result_distances);
    ENGINE_LOG_DEBUG << "CPU cache info";
    cache::CpuCacheMgr::GetInstance()->PrintInfo();  // print cache info after query
    // ENGINE_LOG_DEBUG << "GPU cache info 0";
    // cache::GpuCacheMgr::GetInstance(0)->PrintInfo();  // print cache info before query
    // ENGINE_LOG_DEBUG << "GPU cache info 1";
    // cache::GpuCacheMgr::GetInstance(1)->PrintInfo();  // print cache info before query

    query_ctx->GetTraceContext()->GetSpan()->Finish();

    //TODO assign table_id to each result
    if (not status.ok())
        return status;

    if (result_ids.size() != found_query_ids.size() * k)
        throw std::runtime_error("Wrong result size. Maybe a collection is empty.");


    if (!found_query_ids.empty())
        for (int i = 0; i < vectors.id_array_.size(); ++i){
            if (not found_query_ids[i]){
                for (int j = 0; j < k; ++j){
                    result_ids[i*k + j] = 0;
                    result_distances[i*k + j] = std::numeric_limits<float>::infinity();
                }

            }
        }



    return status;
}

struct DBImpl::compare_fragments_stat_t{
    uint32_t query_sents_total          = 0;
    uint32_t compared_query_sents_total = 0;
    uint32_t query_sents_wo_sim         = 0;
    uint32_t other_sents_total          = 0;
    uint32_t compared_other_sents_total = 0;
};

template<class Json>
void to_json(Json& j, const DBImpl::compare_fragments_stat_t& stat){

    j = {{"query_sents_total", stat.query_sents_total},
         {"compared_query_sents_total", stat.compared_query_sents_total},
         {"query_sents_wo_sim", stat.query_sents_wo_sim},
         {"other_sents_total", stat.other_sents_total},
         {"compared_other_sents_total", stat.compared_other_sents_total}};
}

auto decode_fragment_id(int64_t fragment_id){
    const static unsigned sent_num_mask = 0xffff;
    const static unsigned doc_id_mask = 0xffffffff;

    uint16_t end_sent = fragment_id & sent_num_mask;
    uint16_t beg_sent = (fragment_id >> 16) & sent_num_mask;
    uint32_t doc_id = (fragment_id >> 32) & doc_id_mask;
    return std::make_tuple(doc_id, beg_sent, end_sent);
}

int64_t encode_fragment_id(uint32_t doc_id, uint16_t beg_sent, uint16_t end_sent){
    int64_t fragment_id = doc_id;
    return (fragment_id << 32) | (uint32_t(beg_sent) << 16) | end_sent;
}


void
DBImpl::
InitDirectMap(const std::string& table_id){
    std::lock_guard g (direct_map_access_);
    if (direct_map_per_coll_.empty())
        direct_map_per_coll_.set_empty_key(std::string());

    auto cit = direct_map_per_coll_.find(table_id);
    if (cit != direct_map_per_coll_.end())
        return;

    meta::TableFilesSchema direct_files;
    meta::DatesT dates;
    std::vector<size_t> ids;
    auto status = GetDirectFiles(table_id, ids, dates, direct_files);
    if (!status.ok()) {
        throw std::runtime_error("Failed to find direct files " + status.ToString());
    }

    auto row_count = std::accumulate(
        std::begin(direct_files), std::end(direct_files), 0,
        [](int total, const auto& o) {return total + o.row_count_;});

    auto& map = direct_map_per_coll_[table_id];
    map.set_empty_key(0);
    map.resize(row_count/14);

    for (const auto& file : direct_files ){
        auto direct_engine = EngineFactory::Build(file.dimension_, file.location_,
                                                  (EngineType)file.engine_type_,
                                                  (MetricType)file.metric_type_, file.nlist_,
                                                  file.enc_type_);
        direct_indexes_.push_back(direct_engine);
        direct_engine->Load();
        auto index = direct_engine->GetFaissIndex();
        uint32_t prev_doc_id = 0;
        for(const auto& [k, v] : index->rev_map){
            auto [doc_id, beg_sent, t] = decode_fragment_id(k);
            if (prev_doc_id != doc_id){
                prev_doc_id = doc_id;
                map.insert(std::pair(doc_id, direct_indexes_.size()-1));
            }
        }
        auto ivf_index = knowhere::cast_to_ivf_index(index->index, false);
        if (ivf_index)
            ivf_index->make_direct_map(true);
    }
}

std::pair<int, int>
DBImpl::
LoadFragmentVectors(const std::string& table_id,
                    int64_t fragment_id, ResultIds& ids,
                    std::vector<float>& vectors){
    InitDirectMap(table_id);
    auto& map = direct_map_per_coll_[table_id];
    auto [doc_id, beg_sent, end_sent] = decode_fragment_id(fragment_id);
    auto dit = map.find(doc_id);
    if(dit == map.end())
        return std::pair(0,0);

    direct_indexes_[dit->second]->Load();
    auto index = direct_indexes_[dit->second]->GetFaissIndex();

    int total = end_sent - beg_sent + 1;
    int found = 0;
    while(beg_sent <= end_sent){
        auto sent_id = encode_fragment_id(doc_id, beg_sent, beg_sent);
        ++beg_sent;

        auto fit = index->rev_map.find(sent_id);
        if (fit == index->rev_map.end())
            continue;

        ++found;
        ids.push_back(sent_id);

        auto vit = vectors.insert(vectors.end(), index->d, 0);
        index->index->reconstruct(fit->second, &*vit);

    }
    return std::pair(total, found);

}

Status
DBImpl::
CompareFragments(const CompareFragmentsReq& req, json& resp){
    auto status = Status();
    auto result = json::array();
    compare_fragments_stat_t stat;
    for (const auto& fragment : req.fragments){
        auto query_id = fragment.at(0).get<uint64_t>();
        const auto& fragment_req = fragment.at(1);

        try{
            json fragment_resp = CompareFragmentImpl(req, query_id, fragment_req, stat);
            result.push_back({query_id, fragment_resp});
        }catch(const std::exception& e){
            return Status(SERVER_UNEXPECTED_ERROR, e.what());
        }
    }
    resp = {{"result", result}, {"stat", json(stat)}};
    return status;

}

//not used
void remove_not_found(const std::vector<bool>& found,
                      int dim,
                      ResultIds& ids,
                      std::vector<float>& data){
    auto found_it = std::find(found.cbegin(), found.cend(), false);
    if (found_it == found.cend())
        return;
    auto offs = std::distance(found.cbegin(), found_it);

    auto ids_it = ids.begin() + offs;
    for (auto it=ids_it+1; ++found_it != found.cend(); ++it)
        if(*found_it)
            *ids_it++ = *it;

    ids.erase(ids_it, ids.end());

    found_it = found.cbegin();
    std::advance(found_it, offs);
    auto data_it = data.begin() + offs * dim;
    for(auto it = data_it+dim; ++found_it != found.cend(); it += dim)
        if(*found_it){
            std::memcpy(&*data_it, &*it, dim * sizeof(float));
            data_it += dim;
        }
    data.erase(data_it, data.end());

}


auto brute_force_search_gpu(const CompareFragmentsReq& req, int dim,
                            const std::vector<float>& query,
                            const std::vector<float>& other){

    faiss::gpu::GpuDistanceParams params;
    //TODO take metric from table
    params.metric = faiss::MetricType::METRIC_INNER_PRODUCT;
    params.k = req.topk;
    params.dims = dim;

    params.queries = query.data();
    params.numQueries = query.size() / dim;

    params.vectors = other.data();
    params.numVectors = other.size() / dim;

    std::vector<float> distances(req.topk * params.numQueries);
    params.outDistances = distances.data();
    std::vector<faiss::Index::idx_t> sim_indices(req.topk * params.numQueries);
    params.outIndices = sim_indices.data();

    auto temp_resource = knowhere::FaissGpuResourceMgr::GetInstance().GetRes(req.gpu_id);
    knowhere::ResScope rs(temp_resource, req.gpu_id, true);


    faiss::gpu::bfKnn(temp_resource->faiss_res.get(), params);

    return std::pair(std::move(sim_indices), std::move(distances));

}

auto brute_force_search(const CompareFragmentsReq& req, int dim,
                        const std::vector<float>& query,
                        const std::vector<float>& other){
    if (req.gpu_id == -1){
        throw std::runtime_error("Cpu brute force search is not supported!");
    }
    return brute_force_search_gpu(req, dim, query, other);
}

std::tuple<ResultIds, std::vector<float>, uint32_t>
DBImpl::
PrepareQueryVectors(const CompareFragmentsReq& req,
                    uint64_t query_fragment_id,
                    const engine::meta::TableSchema& table_info,
                    compare_fragments_stat_t& stat){

    auto [doc_id, beg_sent, end_sent] = decode_fragment_id(query_fragment_id);
    auto sents_per_fragment = end_sent - beg_sent + 1;
    stat.query_sents_total += sents_per_fragment;

    ResultIds query_ids;
    query_ids.reserve(sents_per_fragment);
    std::vector<float> query_vectors;
    query_vectors.reserve(sents_per_fragment * table_info.dimension_);
    auto [total, found] = LoadFragmentVectors(req.query_table, query_fragment_id,
                                              query_ids, query_vectors);
    stat.compared_query_sents_total += found;

    return std::tuple(std::move(query_ids), std::move(query_vectors), sents_per_fragment);
}

std::tuple<ResultIds, std::vector<float>, std::vector<uint16_t>, std::vector<int>>
DBImpl::
PrepareOtherVectors(const CompareFragmentsReq& req,
                    const json& fragment_req,
                    uint32_t sents_per_fragment,
                    const engine::meta::TableSchema& table_info,
                    compare_fragments_stat_t& stat){

    auto other_fragments_cnt = std::accumulate(std::begin(fragment_req), std::end(fragment_req), 0,
                                               [](int total, const auto& o) {return total + o.at(1).size();});
    std::vector<float> other_vectors;
    //estimate size
    other_vectors.reserve(sents_per_fragment * other_fragments_cnt * table_info.dimension_);
    ResultIds other_ids;
    other_ids.reserve(sents_per_fragment * other_fragments_cnt);
    std::vector<uint16_t> coll_ids;
    std::vector<int> coll_ids_offs;

    for (const auto& other : fragment_req){
        const auto& table_name = other.at(0).get_ref<const std::string&>();
        auto coll_id = other.at(1).get<uint16_t>();
        const auto& fragment_ids = other.at(2);
        for(const auto& fragment_id : fragment_ids){
            auto [total, found] = LoadFragmentVectors(table_name, fragment_id.get<int64_t>(),
                                                      other_ids, other_vectors);
            stat.other_sents_total += total;
            stat.compared_other_sents_total += found;
        }

        coll_ids.push_back(coll_id);
        coll_ids_offs.push_back(other_ids.size());
    }
    return std::tuple(std::move(other_ids), std::move(other_vectors),
                      std::move(coll_ids), std::move(coll_ids_offs));
}

json
DBImpl::
CompareFragmentImpl(const CompareFragmentsReq& req, uint64_t query_fragment_id,
                    const json& fragment_req, compare_fragments_stat_t& stat){
    if (fragment_req.empty())
        return json::array();

    engine::meta::TableSchema table_info;
    table_info.table_id_ = req.query_table;
    DescribeTable(table_info);

    auto [query_ids, query_vectors, sents_per_fragment] =
        PrepareQueryVectors(req, query_fragment_id,
                            table_info, stat);

    auto [other_ids, other_vectors, coll_ids, coll_ids_offs] =
        PrepareOtherVectors(req, fragment_req,
                            sents_per_fragment,
                            table_info, stat);

    auto [sim_indeces, distances] = brute_force_search(req, table_info.dimension_,
                                                       query_vectors, other_vectors);
    json resp = json::array();
    for(int i = 0; i<query_ids.size(); ++i){
        auto [doc_id, sent_num, t] = decode_fragment_id(query_ids[i]);
        // std::cout<<"result for query "<<doc_id<<":"<<sent_num<<std::endl;
        json found_sents = json::array();
        for(int k = 0; k < req.topk; ++k){
            auto pos = i*req.topk + k;
            if (distances[pos] < req.min_sim){
                if (k == 0)
                    stat.query_sents_wo_sim++;
                break;
            }

            auto [fdoc, fsent, ft] = decode_fragment_id( other_ids[ sim_indeces[pos] ] );

            int coll_id_off = 0;
            while(sim_indeces[pos] >= coll_ids_offs[coll_id_off]) ++coll_id_off;
            found_sents.push_back({coll_ids[coll_id_off], fdoc, fsent, distances[pos]});
            // std::cout<<"k"<<k<<": "<<fdoc<<":"<<fsent<<" - "<<distances[pos]<<std::endl;
        }
        resp.push_back({doc_id, sent_num, std::move(found_sents)});

    }

    return resp;
}

uint64_t
DBImpl::
LoadVectors(const ResultIds& query_ids,
            const std::string& table_id,
            std::vector<bool>& found, std::vector<float>& float_data){

    meta::TableFilesSchema direct_files;

    meta::DatesT dates;
    std::vector<size_t> ids;

    auto status = GetDirectFiles(table_id, ids, dates, direct_files);
    if (!status.ok()) {
        throw std::runtime_error("Failed to find direct files " + status.ToString());
    }

    return LoadVectors(query_ids, direct_files, found, float_data);
}

uint64_t
DBImpl::LoadVectors(const ResultIds& query_ids, const meta::TableFilesSchema& direct_files,
                    std::vector<bool>& found, std::vector<float>& float_data){

    if (query_ids.empty() or direct_files.empty())
        return 0;

    float_data.resize(direct_files.front().dimension_ * query_ids.size(), 0.0);

    found.resize(query_ids.size(), false);

    for (const auto& file : direct_files ){
        auto direct_engine = EngineFactory::Build(file.dimension_, file.location_,
                                                  (EngineType)file.engine_type_,
                                                  (MetricType)file.metric_type_, file.nlist_,
                                                  file.enc_type_);
        direct_engine->Load();
        direct_engine->Reconstruct(query_ids, float_data, found);

    }
    int found_cnt = 0;
    for(auto f : found)
        if (f) ++found_cnt;
    ENGINE_LOG_DEBUG << "Found "<<found_cnt<<" vectors from "<<query_ids.size();

    return found_cnt;
}

Status
DBImpl::QueryByFileID(const std::shared_ptr<server::Context>& context, const std::string& table_id,
                      const std::vector<std::string>& file_ids, uint64_t k, uint64_t nprobe, const VectorsData& vectors,
                      const meta::DatesT& dates, ResultIds& result_ids, ResultDistances& result_distances) {
    auto query_ctx = context->Child("Query by file id");

    if (!initialized_.load(std::memory_order_acquire)) {
        return SHUTDOWN_ERROR;
    }

    ENGINE_LOG_DEBUG << "Query by file ids for table: " << table_id << " date range count: " << dates.size();

    // get specified files
    std::vector<size_t> ids;
    for (auto& id : file_ids) {
        meta::TableFileSchema table_file;
        table_file.table_id_ = table_id;
        std::string::size_type sz;
        ids.push_back(std::stoul(id, &sz));
    }

    meta::TableFilesSchema files_array;
    auto status = GetFilesToSearch(table_id, ids, dates, files_array);
    if (!status.ok()) {
        return status;
    }

    fiu_do_on("DBImpl.QueryByFileID.empty_files_array", files_array.clear());
    if (files_array.empty()) {
        return Status(DB_ERROR, "Invalid file id");
    }


    cache::CpuCacheMgr::GetInstance()->PrintInfo();  // print cache info before query
    status = QueryAsync(query_ctx, table_id, files_array, k, nprobe, vectors, result_ids, result_distances);
    cache::CpuCacheMgr::GetInstance()->PrintInfo();  // print cache info after query

    query_ctx->GetTraceContext()->GetSpan()->Finish();

    return status;
}

Status
DBImpl::Size(uint64_t& result) {
    if (!initialized_.load(std::memory_order_acquire)) {
        return SHUTDOWN_ERROR;
    }

    return meta_ptr_->Size(result);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// internal methods
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Status
DBImpl::QueryAsync(const std::shared_ptr<server::Context>& context, const std::string& table_id,
                   const meta::TableFilesSchema& files, uint64_t k, uint64_t nprobe, const VectorsData& vectors,
                   ResultIds& result_ids, ResultDistances& result_distances) {
    auto query_async_ctx = context->Child("Query Async");

    server::CollectQueryMetrics metrics(vectors.vector_count_);

    TimeRecorder rc("");

    // step 1: construct search job
    auto status = ongoing_files_checker_.MarkOngoingFiles(files);

    ENGINE_LOG_DEBUG << "Engine query begin, index file count: " << files.size();
    scheduler::SearchJobPtr job = std::make_shared<scheduler::SearchJob>(query_async_ctx, k, nprobe, vectors);
    for (auto& file : files) {
        scheduler::TableFileSchemaPtr file_ptr = std::make_shared<meta::TableFileSchema>(file);
        job->AddIndexFile(file_ptr);
    }

    // step 2: put search job to scheduler and wait result
    scheduler::JobMgrInst::GetInstance()->Put(job);
    job->WaitResult();

    status = ongoing_files_checker_.UnmarkOngoingFiles(files);
    if (!job->GetStatus().ok()) {
        return job->GetStatus();
    }

    // step 3: construct results
    result_ids = job->GetResultIds();
    result_distances = job->GetResultDistances();
    rc.ElapseFromBegin("Engine query totally cost");

    query_async_ctx->GetTraceContext()->GetSpan()->Finish();

    return Status::OK();
}

void
DBImpl::BackgroundTimerTask() {
    Status status;
    server::SystemInfo::GetInstance().Init();
    while (true) {
        if (!initialized_.load(std::memory_order_acquire)) {
            WaitMergeFileFinish();
            WaitBuildIndexFinish();

            ENGINE_LOG_DEBUG << "DB background thread exit";
            break;
        }


        StartMetricTask();
        StartCompactionTask();
        StartBuildIndexTask();

        std::this_thread::sleep_for(std::chrono::seconds(300));
    }
}

void
DBImpl::WaitMergeFileFinish() {
    std::lock_guard<std::mutex> lck(compact_result_mutex_);
    for (auto& iter : compact_thread_results_) {
        iter.wait();
    }
}

void
DBImpl::WaitBuildIndexFinish() {
    std::lock_guard<std::mutex> lck(index_result_mutex_);
    for (auto& iter : index_thread_results_) {
        iter.wait();
    }
}

void
DBImpl::StartMetricTask() {
    static uint64_t metric_clock_tick = 0;
    ++metric_clock_tick;
    if (metric_clock_tick % METRIC_ACTION_INTERVAL != 0) {
        return;
    }

    server::Metrics::GetInstance().KeepingAliveCounterIncrement(METRIC_ACTION_INTERVAL);
    int64_t cache_usage = cache::CpuCacheMgr::GetInstance()->CacheUsage();
    int64_t cache_total = cache::CpuCacheMgr::GetInstance()->CacheCapacity();
    fiu_do_on("DBImpl.StartMetricTask.InvalidTotalCache", cache_total = 0);

    if (cache_total > 0) {
        double cache_usage_double = cache_usage;
        server::Metrics::GetInstance().CpuCacheUsageGaugeSet(cache_usage_double * 100 / cache_total);
    } else {
        server::Metrics::GetInstance().CpuCacheUsageGaugeSet(0);
    }

    server::Metrics::GetInstance().GpuCacheUsageGaugeSet();
    uint64_t size;
    Size(size);
    server::Metrics::GetInstance().DataFileSizeGaugeSet(size);
    server::Metrics::GetInstance().CPUUsagePercentSet();
    server::Metrics::GetInstance().RAMUsagePercentSet();
    server::Metrics::GetInstance().GPUPercentGaugeSet();
    server::Metrics::GetInstance().GPUMemoryUsageGaugeSet();
    server::Metrics::GetInstance().OctetsSet();

    server::Metrics::GetInstance().CPUCoreUsagePercentSet();
    server::Metrics::GetInstance().GPUTemperature();
    server::Metrics::GetInstance().CPUTemperature();
    server::Metrics::GetInstance().PushToGateway();
}

Status
DBImpl::SyncMemData(std::set<std::string>& sync_table_ids) {
    std::lock_guard<std::mutex> lck(mem_serialize_mutex_);
    std::set<std::string> temp_table_ids;
    mem_mgr_->Serialize(temp_table_ids);
    for (auto& id : temp_table_ids) {
        sync_table_ids.insert(id);
    }

    if (!temp_table_ids.empty()) {
        SERVER_LOG_DEBUG << "Insert cache serialized";
    }

    return Status::OK();
}

void
DBImpl::StartCompactionTask() {
    static uint64_t compact_clock_tick = 0;
    ++compact_clock_tick;
    if (compact_clock_tick % COMPACT_ACTION_INTERVAL != 0) {
        return;
    }

    // serialize memory data
    SyncMemData(compact_table_ids_);

    // compactiong has been finished?
    {
        std::lock_guard<std::mutex> lck(compact_result_mutex_);
        if (!compact_thread_results_.empty()) {
            std::chrono::milliseconds span(10);
            if (compact_thread_results_.back().wait_for(span) == std::future_status::ready) {
                compact_thread_results_.pop_back();
            }
        }
    }

    // add new compaction task
    {
        std::lock_guard<std::mutex> lck(compact_result_mutex_);
        if (compact_thread_results_.empty()) {
            // collect merge files for all tables(if compact_table_ids_ is empty) for two reasons:
            // 1. other tables may still has un-merged files
            // 2. server may be closed unexpected, these un-merge files need to be merged when server restart
            if (compact_table_ids_.empty()) {
                std::vector<meta::TableSchema> table_schema_array;
                meta_ptr_->AllTables(table_schema_array);
                for (auto& schema : table_schema_array) {
                    compact_table_ids_.insert(schema.table_id_);
                }
            }

            // start merge file thread
            compact_thread_results_.push_back(
                compact_thread_pool_.enqueue(&DBImpl::BackgroundCompaction, this, compact_table_ids_));
            compact_table_ids_.clear();
        }
    }
}

Status
DBImpl::MergeFiles(const std::string& table_id, const meta::DateT& date, const meta::TableFilesSchema& files) {
    ENGINE_LOG_DEBUG << "Merge files for table: " << table_id;

    // step 1: create table file
    meta::TableFileSchema table_file;
    table_file.table_id_ = table_id;
    table_file.date_ = date;
    table_file.file_type_ = meta::TableFileSchema::NEW_MERGE;
    Status status = meta_ptr_->CreateTableFile(table_file);

    if (!status.ok()) {
        ENGINE_LOG_ERROR << "Failed to create table: " << status.ToString();
        return status;
    }

    // step 2: merge files
    ExecutionEnginePtr index =
        EngineFactory::Build(table_file.dimension_, table_file.location_, (EngineType)table_file.engine_type_,
                             (MetricType)table_file.metric_type_, table_file.nlist_,
                             table_file.enc_type_);

    meta::TableFilesSchema updated;
    int64_t index_size = 0, row_count = 0;

    for (auto& file : files) {
        index_size += file.file_size_;
        row_count += file.row_count_;
        if (index_size >= file.index_file_size_) {
            break;
        }
    }
    index->Reserve(index_size, row_count);

    for (auto& file : files) {
        server::CollectMergeFilesMetrics metrics;

        auto status = index->Merge(file.location_);
        if (status.ok()){
            auto file_schema = file;
            file_schema.file_type_ = meta::TableFileSchema::TO_DELETE;
            updated.push_back(file_schema);
        }

        if (index->Size() >= file.index_file_size_) {
            break;
        }
    }

    // step 3: serialize to disk
    try {
        status = index->Serialize();
        fiu_do_on("DBImpl.MergeFiles.Serialize_ThrowException", throw std::exception());
        fiu_do_on("DBImpl.MergeFiles.Serialize_ErrorStatus", status = Status(DB_ERROR, ""));
        if (!status.ok()) {
            ENGINE_LOG_ERROR << status.message();
        }
    } catch (std::exception& ex) {
        std::string msg = "Serialize merged index encounter exception: " + std::string(ex.what());
        ENGINE_LOG_ERROR << msg;
        status = Status(DB_ERROR, msg);
    }

    if (!status.ok()) {
        // if failed to serialize merge file to disk
        // typical error: out of disk space, out of memory or permition denied
        table_file.file_type_ = meta::TableFileSchema::TO_DELETE;
        status = meta_ptr_->UpdateTableFile(table_file);
        ENGINE_LOG_DEBUG << "Failed to update file to index, mark file: " << table_file.file_id_ << " to to_delete";

        ENGINE_LOG_ERROR << "Failed to persist merged file: " << table_file.location_
                         << ", possible out of disk space or memory";

        return status;
    }

    // step 4: update table files state
    // if index type isn't IDMAP, set file type to TO_INDEX if file size execeed index_file_size
    // else set file type to RAW, no need to build index
    if (table_file.engine_type_ != (int)EngineType::FAISS_IDMAP) {
        table_file.file_type_ = (index->PhysicalSize() >= table_file.index_file_size_) ? meta::TableFileSchema::TO_INDEX
                                                                                       : meta::TableFileSchema::RAW;
    } else {
        table_file.file_type_ = meta::TableFileSchema::RAW;
    }
    table_file.file_size_ = index->PhysicalSize();
    table_file.row_count_ = index->Count();
    updated.push_back(table_file);
    status = meta_ptr_->UpdateTableFiles(updated);
    ENGINE_LOG_DEBUG << "New merged file " << table_file.file_id_ << " of size " << index->PhysicalSize() << " bytes";

    if (options_.insert_cache_immediately_) {
        index->Cache();
    }

    return status;
}

Status
DBImpl::BackgroundMergeFiles(const std::string& table_id) {
    meta::DatePartionedTableFilesSchema raw_files;
    auto status = meta_ptr_->FilesToMerge(table_id, raw_files);
    if (!status.ok()) {
        ENGINE_LOG_ERROR << "Failed to get merge files for table: " << table_id;
        return status;
    }

    for (auto& kv : raw_files) {
        meta::TableFilesSchema& files = kv.second;
        if (files.size() < options_.merge_trigger_number_) {
            ENGINE_LOG_TRACE << "Files number not greater equal than merge trigger number, skip merge action";
            continue;
        }

        status = ongoing_files_checker_.MarkOngoingFiles(files);
        MergeFiles(table_id, kv.first, kv.second);
        status = ongoing_files_checker_.UnmarkOngoingFiles(files);

        if (!initialized_.load(std::memory_order_acquire)) {
            ENGINE_LOG_DEBUG << "Server will shutdown, skip merge action for table: " << table_id;
            break;
        }
    }

    return Status::OK();
}

void
DBImpl::BackgroundCompaction(std::set<std::string> table_ids) {
    // ENGINE_LOG_TRACE << " Background compaction thread start";

    Status status;
    for (auto& table_id : table_ids) {
        status = BackgroundMergeFiles(table_id);
        if (!status.ok()) {
            ENGINE_LOG_ERROR << "Merge files for table " << table_id << " failed: " << status.ToString();
        }

        if (!initialized_.load(std::memory_order_acquire)) {
            ENGINE_LOG_DEBUG << "Server will shutdown, skip merge action";
            break;
        }
    }

    meta_ptr_->Archive();

    {
        uint64_t ttl = 10 * meta::SECOND;  // default: file will be hard-deleted few seconds after soft-deleted
        if (options_.mode_ == DBOptions::MODE::CLUSTER_WRITABLE) {
            ttl = meta::HOUR;
        }

        meta_ptr_->CleanUpFilesWithTTL(ttl, &ongoing_files_checker_);
    }

    // ENGINE_LOG_TRACE << " Background compaction thread exit";
}

void
DBImpl::StartBuildIndexTask(bool force) {
    static uint64_t index_clock_tick = 0;
    ++index_clock_tick;
    if (!force && (index_clock_tick % INDEX_ACTION_INTERVAL != 0)) {
        return;
    }

    // build index has been finished?
    {
        std::lock_guard<std::mutex> lck(index_result_mutex_);
        if (!index_thread_results_.empty()) {
            std::chrono::milliseconds span(10);
            if (index_thread_results_.back().wait_for(span) == std::future_status::ready) {
                index_thread_results_.pop_back();
            }
        }
    }

    // add new build index task
    {
        std::lock_guard<std::mutex> lck(index_result_mutex_);
        if (index_thread_results_.empty()) {
            index_thread_results_.push_back(index_thread_pool_.enqueue(&DBImpl::BackgroundBuildIndex, this));
        }
    }
}

void
DBImpl::BackgroundBuildIndex() {
    std::unique_lock<std::mutex> lock(build_index_mutex_);
    meta::TableFilesSchema to_index_files;
    meta_ptr_->FilesToIndex(to_index_files);
    Status status = index_failed_checker_.IgnoreFailedIndexFiles(to_index_files);

    if (!to_index_files.empty()) {
        ENGINE_LOG_DEBUG << "Background build index thread begin";
        status = ongoing_files_checker_.MarkOngoingFiles(to_index_files);

        // step 2: put build index task to scheduler
        std::vector<std::pair<scheduler::BuildIndexJobPtr, scheduler::TableFileSchemaPtr>> job2file_map;
        for (auto& file : to_index_files) {
            scheduler::BuildIndexJobPtr job = std::make_shared<scheduler::BuildIndexJob>(meta_ptr_, options_);
            scheduler::TableFileSchemaPtr file_ptr = std::make_shared<meta::TableFileSchema>(file);
            job->AddToIndexFiles(file_ptr);
            scheduler::JobMgrInst::GetInstance()->Put(job);
            job->WaitBuildIndexFinish();
            meta::TableFileSchema& file_schema = *file_ptr.get();
            if (!job->GetStatus().ok()) {
                Status status = job->GetStatus();
                ENGINE_LOG_ERROR << "Building index job " << job->id() << " failed: " << status.ToString();

                index_failed_checker_.MarkFailedIndexFile(file_schema, status.message());
            } else {
                ENGINE_LOG_DEBUG << "Building index job " << job->id() << " succeed.";

                index_failed_checker_.MarkSucceedIndexFile(file_schema);
            }
            status = ongoing_files_checker_.UnmarkOngoingFile(file_schema);

            // job2file_map.push_back(std::make_pair(job, file_ptr));
        }

        // step 3: wait build index finished and mark failed files
        // for (auto iter = job2file_map.begin(); iter != job2file_map.end(); ++iter) {
        //     scheduler::BuildIndexJobPtr job = iter->first;
        //     meta::TableFileSchema& file_schema = *(iter->second.get());
        //     job->WaitBuildIndexFinish();
        // }

        ENGINE_LOG_DEBUG << "Background build index thread finished";
    }
}

Status
DBImpl::GetFilesToBuildIndex(const std::string& table_id, const std::vector<int>& file_types,
                             meta::TableFilesSchema& files) {
    files.clear();
    auto status = meta_ptr_->FilesByType(table_id, file_types, files);

    // only build index for files that row count greater than certain threshold
    for (auto it = files.begin(); it != files.end();) {
        if ((*it).file_type_ == static_cast<int>(meta::TableFileSchema::RAW) &&
            (*it).row_count_ < meta::BUILD_INDEX_THRESHOLD) {
            it = files.erase(it);
        } else {
            ++it;
        }
    }

    return Status::OK();
}

Status
DBImpl::GetFilesToSearch(const std::string& table_id, const std::vector<size_t>& file_ids, const meta::DatesT& dates,
                         meta::TableFilesSchema& files) {
    ENGINE_LOG_DEBUG << "Collect files from table: " << table_id;

    meta::DatePartionedTableFilesSchema date_files;
    auto status = meta_ptr_->FilesToSearch(table_id, file_ids, dates, date_files);
    if (!status.ok()) {
        return status;
    }

    TraverseFiles(date_files, files);
    return Status::OK();
}

Status
DBImpl::GetDirectFiles(const std::string& table_id, const std::vector<size_t>& file_ids,
                       const meta::DatesT& dates,
                       meta::TableFilesSchema& files) {
    ENGINE_LOG_DEBUG << "Collect files from table: " << table_id;

    meta::DatePartionedTableFilesSchema date_files;
    auto status = meta_ptr_->DirectFiles(table_id, file_ids, dates, date_files);
    if (!status.ok()) {
        return status;
    }

    TraverseFiles(date_files, files);
    return Status::OK();
}

Status
DBImpl::GetPartitionsByTags(const std::string& table_id, const std::vector<std::string>& partition_tags,
                            std::set<std::string>& partition_name_array) {
    std::vector<meta::TableSchema> partition_array;
    auto status = meta_ptr_->ShowPartitions(table_id, partition_array);

    for (auto& tag : partition_tags) {
        // trim side-blank of tag, only compare valid characters
        // for example: " ab cd " is treated as "ab cd"
        std::string valid_tag = tag;
        server::StringHelpFunctions::TrimStringBlank(valid_tag);
        for (auto& schema : partition_array) {
            if (server::StringHelpFunctions::IsRegexMatch(schema.partition_tag_, valid_tag)) {
                partition_name_array.insert(schema.table_id_);
            }
        }
    }

    return Status::OK();
}

Status
DBImpl::DropTableRecursively(const std::string& table_id, const meta::DatesT& dates) {
    // dates partly delete files of the table but currently we don't support
    ENGINE_LOG_DEBUG << "Prepare to delete table " << table_id;

    Status status;
    if (dates.empty()) {
        status = mem_mgr_->EraseMemVector(table_id);  // not allow insert
        status = meta_ptr_->DropTable(table_id);      // soft delete table
        index_failed_checker_.CleanFailedIndexFileOfTable(table_id);

        // scheduler will determine when to delete table files
        auto nres = scheduler::ResMgrInst::GetInstance()->GetNumOfComputeResource();
        scheduler::DeleteJobPtr job = std::make_shared<scheduler::DeleteJob>(table_id, meta_ptr_, nres);
        scheduler::JobMgrInst::GetInstance()->Put(job);
        job->WaitAndDelete();
    } else {
        status = meta_ptr_->DropDataByDate(table_id, dates);
    }

    std::vector<meta::TableSchema> partition_array;
    status = meta_ptr_->ShowPartitions(table_id, partition_array);
    for (auto& schema : partition_array) {
        status = DropTableRecursively(schema.table_id_, dates);
        fiu_do_on("DBImpl.DropTableRecursively.failed", status = Status(DB_ERROR, ""));
        if (!status.ok()) {
            return status;
        }
    }

    return Status::OK();
}

Status
DBImpl::UpdateTableIndexRecursively(const std::string& table_id, const TableIndex& index) {
    DropIndex(table_id);

    auto status = meta_ptr_->UpdateTableIndex(table_id, index);
    fiu_do_on("DBImpl.UpdateTableIndexRecursively.fail_update_table_index",
              status = Status(DB_META_TRANSACTION_FAILED, ""));
    if (!status.ok()) {
        ENGINE_LOG_ERROR << "Failed to update table index info for table: " << table_id;
        return status;
    }

    std::vector<meta::TableSchema> partition_array;
    status = meta_ptr_->ShowPartitions(table_id, partition_array);
    for (auto& schema : partition_array) {
        status = UpdateTableIndexRecursively(schema.table_id_, index);
        if (!status.ok()) {
            return status;
        }
    }

    return Status::OK();
}

Status
DBImpl::BuildTableIndexRecursively(const std::string& table_id, const TableIndex& index) {
    // for IDMAP type, only wait all NEW file converted to RAW file
    // for other type, wait NEW/RAW/NEW_MERGE/NEW_INDEX/TO_INDEX files converted to INDEX files
    std::vector<int> file_types;
    if (index.engine_type_ == static_cast<int32_t>(EngineType::FAISS_IDMAP)) {
        file_types = {
            static_cast<int32_t>(meta::TableFileSchema::NEW),
            static_cast<int32_t>(meta::TableFileSchema::NEW_MERGE),
        };
    } else {
        file_types = {
            static_cast<int32_t>(meta::TableFileSchema::RAW),
            static_cast<int32_t>(meta::TableFileSchema::NEW),
            static_cast<int32_t>(meta::TableFileSchema::NEW_MERGE),
            static_cast<int32_t>(meta::TableFileSchema::NEW_INDEX),
            static_cast<int32_t>(meta::TableFileSchema::TO_INDEX),
        };
    }

    // get files to build index
    meta::TableFilesSchema table_files;
    auto status = GetFilesToBuildIndex(table_id, file_types, table_files);
    int times = 1;

    while (!table_files.empty()) {
        ENGINE_LOG_DEBUG << "Non index files detected! Will build index " << times
                         <<" files: "<<table_files.size();
        if (index.engine_type_ != (int)EngineType::FAISS_IDMAP) {
            status = meta_ptr_->UpdateTableFilesToIndex(table_id, index.engine_type_);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(std::min(10 * 1000, times * 100)));
        GetFilesToBuildIndex(table_id, file_types, table_files);
        ++times;

        index_failed_checker_.IgnoreFailedIndexFiles(table_files);
    }

    // build index for partition
    std::vector<meta::TableSchema> partition_array;
    status = meta_ptr_->ShowPartitions(table_id, partition_array);
    for (auto& schema : partition_array) {
        status = BuildTableIndexRecursively(schema.table_id_, index);
        fiu_do_on("DBImpl.BuildTableIndexRecursively.fail_build_table_Index_for_partition",
                  status = Status(DB_ERROR, ""));
        if (!status.ok()) {
            return status;
        }
    }

    // failed to build index for some files, return error
    std::string err_msg;
    index_failed_checker_.GetErrMsgForTable(table_id, err_msg);
    fiu_do_on("DBImpl.BuildTableIndexRecursively.not_empty_err_msg", err_msg.append("fiu"));
    if (!err_msg.empty()) {
        return Status(DB_ERROR, err_msg);
    }

    return Status::OK();
}

Status
DBImpl::DropTableIndexRecursively(const std::string& table_id) {
    ENGINE_LOG_DEBUG << "Drop index for table: " << table_id;
    index_failed_checker_.CleanFailedIndexFileOfTable(table_id);
    auto status = meta_ptr_->DropTableIndex(table_id);
    if (!status.ok()) {
        return status;
    }

    // drop partition index
    std::vector<meta::TableSchema> partition_array;
    status = meta_ptr_->ShowPartitions(table_id, partition_array);
    for (auto& schema : partition_array) {
        status = DropTableIndexRecursively(schema.table_id_);
        fiu_do_on("DBImpl.DropTableIndexRecursively.fail_drop_table_Index_for_partition",
                  status = Status(DB_ERROR, ""));
        if (!status.ok()) {
            return status;
        }
    }

    return Status::OK();
}

Status
DBImpl::GetTableRowCountRecursively(const std::string& table_id, uint64_t& row_count) {
    row_count = 0;
    auto status = meta_ptr_->Count(table_id, row_count);
    if (!status.ok()) {
        return status;
    }

    // get partition row count
    std::vector<meta::TableSchema> partition_array;
    status = meta_ptr_->ShowPartitions(table_id, partition_array);
    for (auto& schema : partition_array) {
        uint64_t partition_row_count = 0;
        status = GetTableRowCountRecursively(schema.table_id_, partition_row_count);
        fiu_do_on("DBImpl.GetTableRowCountRecursively.fail_get_table_rowcount_for_partition",
                  status = Status(DB_ERROR, ""));
        if (!status.ok()) {
            return status;
        }

        row_count += partition_row_count;
    }

    return Status::OK();
}



}  // namespace engine
}  // namespace milvus
