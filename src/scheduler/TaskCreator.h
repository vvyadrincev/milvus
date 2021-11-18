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

#include "job/DeleteJob.h"
#include "job/Job.h"
#include "job/SearchJob.h"
#include "job/ClusterizeJob.h"
#include "task/BuildIndexTask.h"
#include "task/DeleteTask.h"
#include "task/SearchTask.h"
#include "task/ClusterizeTask.h"
#include "task/Task.h"

namespace milvus {
namespace scheduler {

class TaskCreator {
 public:
    static std::vector<TaskPtr>
    Create(const JobPtr& job);

 public:
    static std::vector<TaskPtr>
    Create(const SearchJobPtr& job);

    static std::vector<TaskPtr>
    Create(const DeleteJobPtr& job);

    static std::vector<TaskPtr>
    Create(const BuildIndexJobPtr& job);

    static std::vector<TaskPtr>
    Create(const ClusterizeJobPtr& job);

};

}  // namespace scheduler
}  // namespace milvus
