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

#include <cstring>

#include "knowhere/index/vector_index/nsg/NSGIO.h"

namespace knowhere {
namespace algo {

void
write_index(NsgIndex* index, MemoryIOWriter& writer) {
    writer(&index->ntotal, sizeof(index->ntotal), 1);
    writer(&index->dimension, sizeof(index->dimension), 1);
    writer(&index->navigation_point, sizeof(index->navigation_point), 1);
    writer(index->ori_data_, sizeof(float) * index->ntotal * index->dimension, 1);
    writer(index->ids_, sizeof(int64_t) * index->ntotal, 1);

    for (unsigned i = 0; i < index->ntotal; ++i) {
        auto neighbor_num = (node_t)index->nsg[i].size();
        writer(&neighbor_num, sizeof(node_t), 1);
        writer(index->nsg[i].data(), neighbor_num * sizeof(node_t), 1);
    }
}

NsgIndex*
read_index(MemoryIOReader& reader) {
    size_t ntotal;
    size_t dimension;
    reader(&ntotal, sizeof(size_t), 1);
    reader(&dimension, sizeof(size_t), 1);
    auto index = new NsgIndex(dimension, ntotal);
    reader(&index->navigation_point, sizeof(index->navigation_point), 1);

    index->ori_data_ = new float[index->ntotal * index->dimension];
    index->ids_ = new int64_t[index->ntotal];
    reader(index->ori_data_, sizeof(float) * index->ntotal * index->dimension, 1);
    reader(index->ids_, sizeof(int64_t) * index->ntotal, 1);

    index->nsg.reserve(index->ntotal);
    index->nsg.resize(index->ntotal);
    node_t neighbor_num;
    for (unsigned i = 0; i < index->ntotal; ++i) {
        reader(&neighbor_num, sizeof(node_t), 1);
        index->nsg[i].reserve(neighbor_num);
        index->nsg[i].resize(neighbor_num);
        reader(index->nsg[i].data(), neighbor_num * sizeof(node_t), 1);
    }

    index->is_trained = true;
    return index;
}

}  // namespace algo
}  // namespace knowhere
