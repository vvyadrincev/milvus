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

#include "TaskLabel.h"
#include "scheduler/ResourceMgr.h"

#include <memory>
#include <string>

// class Resource;
//
// using ResourceWPtr = std::weak_ptr<Resource>;

namespace milvus {
namespace scheduler {

class SpecResLabel : public TaskLabel {
 public:
    explicit SpecResLabel(const ResourceWPtr& resource)
        : TaskLabel(TaskLabelType::SPECIFIED_RESOURCE), resource_(resource) {
    }

    inline ResourceWPtr&
    resource() {
        return resource_;
    }

    inline std::string&
    resource_name() {
        return resource_name_;
    }

 private:
    ResourceWPtr resource_;
    std::string resource_name_;
};

using SpecResLabelPtr = std::shared_ptr<SpecResLabel>();

}  // namespace scheduler
}  // namespace milvus
