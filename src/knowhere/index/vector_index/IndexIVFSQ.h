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
#include <utility>

#include "IndexIVF.h"

namespace knowhere {

class IVFSQ : public IVF {
 public:
    explicit IVFSQ(std::shared_ptr<faiss::Index> index) : IVF(std::move(index)) {
    }

    IVFSQ() = default;

    IndexModelPtr
    Train(const DatasetPtr& dataset, const Config& config) override;

    VectorIndexPtr
    CopyCpuToGpu(const int64_t& device_id, const Config& config) override;

 protected:
    //    VectorIndexPtr
    //    Clone_impl(const std::shared_ptr<faiss::Index>& index) override;
};

}  // namespace knowhere
