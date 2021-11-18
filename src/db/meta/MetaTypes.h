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

#include "db/Constants.h"
#include "db/engine/ExecutionEngine.h"
#include "version.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace milvus {
namespace engine {
namespace meta {

constexpr int32_t DEFAULT_ENGINE_TYPE = (int)EngineType::FAISS_IDMAP;
constexpr int32_t DEFAULT_NLIST = 16384;
constexpr int32_t DEFAULT_METRIC_TYPE = (int)MetricType::L2;
constexpr int32_t DEFAULT_INDEX_FILE_SIZE = ONE_GB;
constexpr char DEFAULT_ENC_TYPE[] = "Flat";
constexpr char CURRENT_VERSION[] = MILVUS_VERSION;

constexpr int64_t FLAG_MASK_NO_USERID = 0x1;
constexpr int64_t FLAG_MASK_HAS_USERID = 0x1 << 1;

using DateT = int;
const DateT EmptyDate = -1;
using DatesT = std::vector<DateT>;

struct TableSchema {
    typedef enum {
        NORMAL,
        TO_DELETE,
    } TABLE_STATE;

    size_t id_ = 0;
    std::string table_id_;
    int32_t state_ = (int)NORMAL;
    uint16_t dimension_ = 0;
    int64_t created_on_ = 0;
    int64_t flag_ = 0;
    int64_t index_file_size_ = DEFAULT_INDEX_FILE_SIZE;
    int32_t engine_type_ = DEFAULT_ENGINE_TYPE;
    int32_t nlist_ = DEFAULT_NLIST;
    int32_t metric_type_ = DEFAULT_METRIC_TYPE;
    std::string enc_type_ = DEFAULT_ENC_TYPE;
    std::string owner_table_;
    std::string partition_tag_;
    std::string version_ = CURRENT_VERSION;
};  // TableSchema

struct TableFileSchema {
    typedef enum {
        NEW,
        RAW,
        TO_INDEX,
        INDEX,
        TO_DELETE,
        NEW_MERGE,
        NEW_INDEX,
        BACKUP,
        DIRECT,
    } FILE_TYPE;

    size_t id_ = 0;
    std::string table_id_;
    std::string file_id_;
    int32_t file_type_ = NEW;
    size_t file_size_ = 0;
    size_t row_count_ = 0;
    DateT date_ = EmptyDate;
    uint16_t dimension_ = 0;
    std::string location_;
    int64_t updated_time_ = 0;
    int64_t created_on_ = 0;
    int64_t index_file_size_ = DEFAULT_INDEX_FILE_SIZE;  // not persist to meta
    int32_t engine_type_ = DEFAULT_ENGINE_TYPE;
    int32_t nlist_ = DEFAULT_NLIST;              // not persist to meta
    int32_t metric_type_ = DEFAULT_METRIC_TYPE;  // not persist to meta
    std::string enc_type_ = DEFAULT_ENC_TYPE;
};                                               // TableFileSchema

using TableFileSchemaPtr = std::shared_ptr<meta::TableFileSchema>;
using TableFilesSchema = std::vector<TableFileSchema>;
using DatePartionedTableFilesSchema = std::map<DateT, TableFilesSchema>;

}  // namespace meta
}  // namespace engine
}  // namespace milvus
