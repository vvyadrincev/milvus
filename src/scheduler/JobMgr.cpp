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

#include "scheduler/JobMgr.h"
#include "SchedInst.h"
#include "TaskCreator.h"
#include "optimizer/Optimizer.h"
#include "scheduler/Algorithm.h"
#include "scheduler/optimizer/Optimizer.h"
#include "scheduler/tasklabel/SpecResLabel.h"
#include "task/Task.h"

#include <utility>

namespace milvus {
namespace scheduler {

JobMgr::JobMgr(ResourceMgrPtr res_mgr) : res_mgr_(std::move(res_mgr)) {
}

void
JobMgr::Start() {
    if (not running_) {
        running_ = true;
        worker_thread_ = std::thread(&JobMgr::worker_function, this);
    }
}

void
JobMgr::Stop() {
    if (running_) {
        this->Put(nullptr);
        worker_thread_.join();
        running_ = false;
    }
}

json
JobMgr::Dump() const {
    json ret{
        {"running", running_},
        {"event_queue_length", queue_.size()},
    };
    return ret;
}

void
JobMgr::Put(const JobPtr& job) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(job);
    }
    cv_.notify_one();
}

void
JobMgr::worker_function() {
    while (running_) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty(); });
        auto job = queue_.front();
        queue_.pop();
        lock.unlock();
        if (job == nullptr) {
            break;
        }

        auto tasks = build_task(job);
        for (auto& task : tasks) {
            OptimizerInst::GetInstance()->Run(task);
        }

        for (auto& task : tasks) {
            calculate_path(res_mgr_, task);
        }

        // disk resources NEVER be empty.
        if (auto disk = res_mgr_->GetDiskResources()[0].lock()) {
            for (auto& task : tasks) {
                disk->task_table().Put(task, nullptr);
            }
        }
    }
}

std::vector<TaskPtr>
JobMgr::build_task(const JobPtr& job) {
    return TaskCreator::Create(job);
}

void
JobMgr::calculate_path(const ResourceMgrPtr& res_mgr, const TaskPtr& task) {
    if (task->type_ != TaskType::SearchTask && task->type_ != TaskType::BuildIndexTask &&
        task->type_ != TaskType::ClusterizeTask) {
        return;
    }

    if (task->label()->Type() != TaskLabelType::SPECIFIED_RESOURCE) {
        return;
    }

    std::vector<std::string> path;
    auto spec_label = std::static_pointer_cast<SpecResLabel>(task->label());
    auto src = res_mgr->GetDiskResources()[0];
    auto dest = spec_label->resource();
    ShortestPath(src.lock(), dest.lock(), res_mgr, path);
    task->path() = Path(path, path.size() - 1);
}

}  // namespace scheduler
}  // namespace milvus
