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
#include <vector>

#include "VectorIndex.h"

namespace knowhere {

namespace algo {
class NsgIndex;
}

class NSG : public VectorIndex {
 public:
    explicit NSG(const int64_t& gpu_num) : gpu_(gpu_num) {
    }

    NSG() = default;

    IndexModelPtr
    Train(const DatasetPtr& dataset, const Config& config) override;
    DatasetPtr
    Search(const DatasetPtr& dataset, const Config& config) override;

    void Reconstruct(std::vector<int64_t> ids, std::vector<float>& xb,
                     std::vector<bool>& found)override{
        throw std::runtime_error("Not impl!");
    }
    void
    GetIds(std::vector<int64_t>& ids) override{
        throw std::logic_error("unsupported!");
    }
    void
    Add(const DatasetPtr& dataset, const Config& config) override;
    BinarySet
    Serialize() override;
    void
    Load(const BinarySet& index_binary) override;
    int64_t
    Count() override;
    int64_t
    Dimension() override;
    //    VectorIndexPtr
    //    Clone() override;
    void
    Seal() override;

 private:
    std::shared_ptr<algo::NsgIndex> index_;
    int64_t gpu_;
};

using NSGIndexPtr = std::shared_ptr<NSG>();

}  // namespace knowhere
