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

namespace milvus {
namespace scheduler {

enum class TaskLabelType {
    SPECIFIED_RESOURCE,  // means must executing in special resource
    BROADCAST,           // means all enable-executor resource must execute task
};

class TaskLabel {
 public:
    inline TaskLabelType
    Type() const {
        return type_;
    }

 protected:
    explicit TaskLabel(TaskLabelType type) : type_(type) {
    }

 private:
    TaskLabelType type_;
};

using TaskLabelPtr = std::shared_ptr<TaskLabel>;

}  // namespace scheduler
}  // namespace milvus
