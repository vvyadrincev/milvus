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

#ifndef UTILS_STATUS_H
#define UTILS_STATUS_H

#include "utils/Error.h"

#include <string>

namespace milvus {

using StatusCode = ErrorCode;

class Status {
 public:
    Status(StatusCode code, const std::string& msg);
    Status();
    ~Status();

    Status(const Status& s);

    Status&
    operator=(const Status& s);

    Status(Status&& s);

    Status&
    operator=(Status&& s);

    static Status
    OK() {
        return Status();
    }

    bool
    ok() const {
        return state_ == nullptr || code() == 0;
    }

    StatusCode
    code() const {
        return (state_ == nullptr) ? 0 : *(StatusCode*)(state_);
    }

    std::string
    message() const;

    std::string
    ToString() const;

 private:
    inline void
    CopyFrom(const Status& s);

    inline void
    MoveFrom(Status& s);

 private:
    char* state_ = nullptr;
};  // Status

}  // namespace milvus

#endif // UTILS_STATUS_H
