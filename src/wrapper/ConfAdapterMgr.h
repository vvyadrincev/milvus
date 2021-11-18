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

#include "ConfAdapter.h"
#include "VecIndex.h"

#include <functional>
#include <map>
#include <memory>

namespace milvus {
namespace engine {

class AdapterMgr {
 public:
    template <typename T>
    struct register_t {
        explicit register_t(const IndexType& key) {
            AdapterMgr::GetInstance().table_.emplace(key, [] { return std::make_shared<T>(); });
        }
    };

    static AdapterMgr&
    GetInstance() {
        static AdapterMgr instance;
        return instance;
    }

    ConfAdapterPtr
    GetAdapter(const IndexType& indexType);

    void
    RegisterAdapter();

 protected:
    bool init_ = false;
    std::map<IndexType, std::function<ConfAdapterPtr()> > table_;
};

}  // namespace engine
}  // namespace milvus
