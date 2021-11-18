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

#include "db/Types.h"
#include "meta/Meta.h"
#include "utils/Status.h"

#include <map>
#include <mutex>
#include <string>

namespace milvus {
namespace engine {

class IndexFailedChecker {
 public:
    Status
    CleanFailedIndexFileOfTable(const std::string& table_id);

    Status
    GetErrMsgForTable(const std::string& table_id, std::string& err_msg);

    Status
    MarkFailedIndexFile(const meta::TableFileSchema& file, const std::string& err_msg);

    Status
    MarkSucceedIndexFile(const meta::TableFileSchema& file);

    Status
    IgnoreFailedIndexFiles(meta::TableFilesSchema& table_files);

 private:
    std::mutex mutex_;
    Table2FileErr index_failed_files_;  // table id mapping to (file id mapping to failed times)
};

}  // namespace engine
}  // namespace milvus
