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

#include <faiss/IndexIVF.h>

#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "FaissBaseBinaryIndex.h"
#include "VectorIndex.h"

namespace knowhere {

class BinaryIVF : public VectorIndex, public FaissBaseBinaryIndex {
 public:
    BinaryIVF() : FaissBaseBinaryIndex(nullptr) {
    }

    explicit BinaryIVF(std::shared_ptr<faiss::IndexBinary> index) : FaissBaseBinaryIndex(std::move(index)) {
    }

    BinarySet
    Serialize() override;

    void
    Load(const BinarySet& index_binary) override;

    DatasetPtr
    Search(const DatasetPtr& dataset, const Config& config) override;

    void
    Reconstruct(std::vector<int64_t> ids, std::vector<float>& xb,
                std::vector<bool>& found) override{
        throw std::logic_error("unsupported!");
    }

    void
    GetIds(std::vector<int64_t>& ids) override{
        throw std::logic_error("unsupported!");
    }

    void
    Add(const DatasetPtr& dataset, const Config& config) override;

    void
    Seal() override;

    IndexModelPtr
    Train(const DatasetPtr& dataset, const Config& config) override;

    int64_t
    Count() override;

    int64_t
    Dimension() override;

 protected:
    virtual std::shared_ptr<faiss::IVFSearchParameters>
    GenParams(const Config& config);

    virtual void
    search_impl(int64_t n, const uint8_t* data, int64_t k, float* distances, int64_t* labels, const Config& cfg);

 protected:
    std::mutex mutex_;
};

using BinaryIVFIndexPtr = std::shared_ptr<BinaryIVF>;

}  // namespace knowhere
