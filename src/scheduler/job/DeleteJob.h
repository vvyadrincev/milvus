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

#include <condition_variable>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "Job.h"
#include "db/meta/Meta.h"

namespace milvus {
namespace scheduler {

class DeleteJob : public Job {
 public:
    DeleteJob(std::string table_id, engine::meta::MetaPtr meta_ptr, uint64_t num_resource);

 public:
    void
    WaitAndDelete();

    void
    ResourceDone();

    json
    Dump() const override;

 public:
    std::string
    table_id() const {
        return table_id_;
    }

    engine::meta::MetaPtr
    meta() const {
        return meta_ptr_;
    }

 private:
    std::string table_id_;
    engine::meta::MetaPtr meta_ptr_;

    uint64_t num_resource_ = 0;
    uint64_t done_resource = 0;
    std::mutex mutex_;
    std::condition_variable cv_;
};

using DeleteJobPtr = std::shared_ptr<DeleteJob>;

}  // namespace scheduler
}  // namespace milvus
