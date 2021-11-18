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
#include <string>
#include <utility>

namespace milvus {
namespace scheduler {

enum class EventType { START_UP, LOAD_COMPLETED, FINISH_TASK, TASK_TABLE_UPDATED };

class Resource;

class Event {
 public:
    explicit Event(EventType type, std::shared_ptr<Resource> resource) : type_(type), resource_(std::move(resource)) {
    }

    inline EventType
    Type() const {
        return type_;
    }

    virtual std::string
    Dump() const = 0;

    friend std::ostream&
    operator<<(std::ostream& out, const Event& event);

 public:
    EventType type_;
    std::shared_ptr<Resource> resource_;
};

using EventPtr = std::shared_ptr<Event>;

}  // namespace scheduler
}  // namespace milvus
