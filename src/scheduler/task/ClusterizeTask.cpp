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

#include "ClusterizeTask.h"

#include "db/Utils.h"
#include "db/engine/EngineFactory.h"
#include "metrics/Metrics.h"
#include "scheduler/job/BuildIndexJob.h"
#include "utils/Exception.h"
#include "utils/Log.h"
#include "utils/TimeRecorder.h"

#include "knowhere/index/vector_index/helpers/FaissGpuResourceMgr.h"

#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/Clustering.h>

#include <memory>
#include <string>
#include <thread>
#include <utility>

namespace milvus {
namespace scheduler {

XClusterizeTask::XClusterizeTask(engine::meta::TableSchema dest_table, TaskLabelPtr label)
    : Task(TaskType::ClusterizeTask, std::move(label)),
      dest_table_(dest_table){
}

void
XClusterizeTask::Load(milvus::scheduler::LoadType type, uint8_t device_id) {
    TimeRecorder rc("");
    Status stat = Status::OK();
    std::string error_msg;
    std::string type_str;

    ENGINE_LOG_DEBUG<< "loading begin "<<(int)type<< " device id"<<device_id;

    if (auto job = job_.lock()) {
        auto cluster_job = std::static_pointer_cast<scheduler::ClusterizeJob>(job);
        if (cluster_job->vectors().float_data_.empty()){
            LoadFromDirectFiles(cluster_job,
                                const_cast<std::vector<float>&>(cluster_job->vectors().float_data_),
                                const_cast<uint64_t&>(cluster_job->vectors().vector_count_));
        }

        faiss::MetricType fm = faiss::METRIC_L2;
        if (dest_table_.metric_type_ == static_cast<int32_t>(engine::MetricType::IP))
            fm = faiss::METRIC_INNER_PRODUCT;

        try {
            if (type == LoadType::DISK2CPU) {
                index_ = std::make_unique<faiss::IndexFlat>(dest_table_.dimension_, fm);
                stat = Status(SERVER_SUCCESS, "");

            } else if (type == LoadType::CPU2GPU) {
                if (auto res = knowhere::FaissGpuResourceMgr::GetInstance().GetRes(device_id)) {
                    knowhere::ResScope rs(res, device_id, false);

                    faiss::gpu::GpuIndexFlatConfig gpu_config;
                    gpu_config.useFloat16 = false;
                    gpu_config.device = device_id;
                    index_ = std::make_unique<faiss::gpu::GpuIndexFlat>(
                        res->faiss_res.get(), dest_table_.dimension_, fm, gpu_config);
                    res_ = res;
                }else{
                    error_msg = "can't get gpu resource";
                    stat = Status(SERVER_UNEXPECTED_ERROR, error_msg);
                }
            } else {
                error_msg = "Wrong load type";
                stat = Status(SERVER_UNEXPECTED_ERROR, error_msg);
            }
        } catch (std::exception& ex) {
            // typical error: out of disk space or permition denied
            error_msg = "Failed to load to_index file: " + std::string(ex.what());
            stat = Status(SERVER_UNEXPECTED_ERROR, error_msg);
        }
        if (!stat.ok()) {
            if (auto job = job_.lock()) {
                auto cluster_job = std::static_pointer_cast<scheduler::ClusterizeJob>(job);
                cluster_job->Done();
            }
            index_.reset();
            return;
        }

    }
}

void
XClusterizeTask::LoadFromDirectFiles(scheduler::ClusterizeJobPtr job,
                                     std::vector<float>& float_data,
                                     uint64_t& vec_cnt){

    if (job->direct_files().empty())
        throw std::runtime_error("Direct files are empty!");

    std::vector<int64_t> all_ids;
    std::vector<engine::ExecutionEnginePtr> engines;
    for (const auto& file : job->direct_files() ){
        auto direct_engine = EngineFactory::Build(file.dimension_, file.location_,
                                                  (EngineType)file.engine_type_,
                                                  (MetricType)file.metric_type_, file.nlist_,
                                                  file.enc_type_);
        engines.push_back(direct_engine);

        direct_engine->Load();
        std::vector<int64_t> ids;
        direct_engine->GetIds(ids);
        all_ids.insert(all_ids.end(), ids.begin(), ids.end());
    }


    float_data.resize(job->direct_files().front().dimension_ * all_ids.size(), 0.0);

    std::vector<bool> found;
    found.resize(all_ids.size(), false);

    for (const auto& direct_engine : engines)
        direct_engine->Reconstruct(all_ids, float_data, found);

    vec_cnt = all_ids.size();
    ENGINE_LOG_DEBUG<< all_ids.size() << " vectors were loaded ";


}

void
XClusterizeTask::Execute() {
    if (not index_) {
        return;
    }

    // TimeRecorder rc("DoBuildIndex file id:" + std::to_string(to_index_id_));

    if (auto job = job_.lock()) {
        auto cluster_job = std::static_pointer_cast<scheduler::ClusterizeJob>(job);
        const auto& opts = cluster_job->options();

        faiss::ClusteringParameters cp;
        cp.niter = opts.niter;
        cp.nredo = opts.nredo;
        cp.verbose = opts.verbose;
        cp.spherical = false;
        if (dest_table_.metric_type_ == static_cast<int32_t>(engine::MetricType::IP))
            cp.spherical = true;
        faiss::Clustering kMeans(dest_table_.dimension_, opts.number_of_clusters, cp);
        try{
            kMeans.train(cluster_job->vectors().vector_count_,
                         cluster_job->vectors().float_data_.data(), *index_);
        }catch(const std::exception& e){
            cluster_job->GetStatus() = Status(SERVER_UNEXPECTED_ERROR, e.what());
            cluster_job->Done();
            index_.reset();
            return;
        }

        engine::meta::MetaPtr meta_ptr = cluster_job->meta();

        // if table has been deleted, dont save index file
        bool has_table = false;
        meta_ptr->HasTable(opts.table_id, has_table);

        if (!has_table) {
            meta_ptr->DeleteTableFiles(opts.table_id);
            cluster_job->GetStatus() = Status(DB_ERROR, "Table has been deleted, discard index file.");
            cluster_job->Done();
            index_.reset();
            return;
        }

        // save index file
        ExecutionEnginePtr index_engine = nullptr;
        engine::meta::TableFileSchema table_file;
        table_file.table_id_ = opts.table_id;
        table_file.date_ = engine::utils::GetDate();
        table_file.file_type_ = engine::meta::TableFileSchema::RAW;
        Status status;
        try {
            status = meta_ptr->CreateTableFile(table_file);
            if (not status.ok())
                throw std::runtime_error("Failed to create table file " + status.message());
            index_engine = EngineFactory::Build(
                dest_table_.dimension_, table_file.location_, (EngineType)table_file.engine_type_,
                (MetricType)table_file.metric_type_, table_file.nlist_, table_file.enc_type_);

            std::vector<int64_t> ids;
            ids.resize(index_->ntotal);
            for (int64_t i = 0; i < index_->ntotal; ++i)
                ids[i] = i;

            index_engine->AddWithIds(index_->ntotal,
                                     kMeans.centroids.data(),
                                     ids.data());
            status = index_engine->Serialize();
            if (!status.ok())
                throw std::runtime_error("Failed to serialize " + status.message());


            // update meta
            table_file.file_size_ = index_engine->PhysicalSize();
            table_file.row_count_ = index_engine->Count();
            status = meta_ptr->UpdateTableFile(table_file);

        } catch (std::exception& ex) {
            std::string msg = "unexpected exception: " + std::string(ex.what());
            ENGINE_LOG_ERROR << msg;
            status = Status(DB_ERROR, msg);
        }

        if (!status.ok()) {
            // if failed to serialize index file to disk
            // typical error: out of disk space, out of memory or permition denied
            table_file.file_type_ = engine::meta::TableFileSchema::TO_DELETE;
            status = meta_ptr->UpdateTableFile(table_file);
            ENGINE_LOG_DEBUG << "Failed to update file to index, mark file: " << table_file.file_id_ << " to to_delete";

            ENGINE_LOG_ERROR << "Failed to persist index file: " << table_file.location_
                             << ", possible out of disk space or memory";

            cluster_job->Done();
            cluster_job->GetStatus() = status;
            index_.reset();
            return;
        }

        cluster_job->Done();
        index_.reset();
    }
}

}  // namespace scheduler
}  // namespace milvus
