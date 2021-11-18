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

#include "Options.h"
#include "db/Types.h"
#include "db/meta/MetaTypes.h"

#include <ctime>
#include <string>

namespace milvus {
namespace engine {
namespace utils {

int64_t
GetMicroSecTimeStamp();

Status
CreateTablePath(const DBMetaOptions& options, const std::string& table_id);
Status
DeleteTablePath(const DBMetaOptions& options, const std::string& table_id, bool force = true);

Status
CreateTableFilePath(const DBMetaOptions& options, meta::TableFileSchema& table_file);
Status
GetTableFilePath(const DBMetaOptions& options, meta::TableFileSchema& table_file);
Status
DeleteTableFilePath(const DBMetaOptions& options, meta::TableFileSchema& table_file);

bool
IsSameIndex(const TableIndex& index1, const TableIndex& index2);

meta::DateT
GetDate(const std::time_t& t, int day_delta = 0);
meta::DateT
GetDate();
meta::DateT
GetDateWithDelta(int day_delta);

struct MetaUriInfo {
    std::string dialect_;
    std::string username_;
    std::string password_;
    std::string host_;
    std::string port_;
    std::string db_name_;
};

Status
ParseMetaUri(const std::string& uri, MetaUriInfo& info);

}  // namespace utils
}  // namespace engine
}  // namespace milvus
