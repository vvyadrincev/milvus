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

#include "Task.h"
#include "scheduler/Definition.h"
#include "scheduler/job/ClusterizeJob.h"

#include "knowhere/index/vector_index/helpers/FaissGpuResourceMgr.h"

namespace milvus {
namespace scheduler {

class XClusterizeTask : public Task {
 public:
    explicit XClusterizeTask(engine::meta::TableSchema dest_table, TaskLabelPtr label);

    void
    Load(LoadType type, uint8_t device_id) override;

    void
    Execute() override;

    ClusterizeJobPtr
    Job()const{
        auto job = job_.lock();
        return std::static_pointer_cast<scheduler::ClusterizeJob>(job);
    }

private:
    void
    LoadFromDirectFiles(scheduler::ClusterizeJobPtr job,
                        std::vector<float>& float_data,
                        uint64_t& vec_cnt);

    engine::meta::TableSchema dest_table_;
    std::unique_ptr<faiss::Index> index_;
    knowhere::ResWPtr res_;


};

}  // namespace scheduler
}  // namespace milvus
