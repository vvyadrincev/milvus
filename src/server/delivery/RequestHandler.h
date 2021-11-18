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

#include "server/delivery/request/BaseRequest.h"
#include "db/Options.h"
#include "utils/Status.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace milvus {
namespace server {

class RequestHandler {
 public:
    RequestHandler() = default;

    Status
    CreateTable(const std::shared_ptr<Context>& context, const std::string& table_name, int64_t dimension,
                int64_t index_file_size, int64_t metric_type, const std::string& enc_type = "Flat");

    Status
    HasTable(const std::shared_ptr<Context>& context, const std::string& table_name, bool& has_table);

    Status
    DropTable(const std::shared_ptr<Context>& context, const std::string& table_name);

    Status
    CreateIndex(const std::shared_ptr<Context>& context, const std::string& table_name, int64_t index_type,
                int64_t nlist, const std::string& enc_type = "Flat");

    Status
    Insert(const std::shared_ptr<Context>& context, const std::string& table_name, engine::VectorsData& vectors,
           const std::string& partition_tag);

    Status
    ShowTables(const std::shared_ptr<Context>& context, std::vector<std::string>& tables);

    Status
    Search(const std::shared_ptr<Context>& context, const std::vector<std::string>& table_names, const engine::VectorsData& vectors,
           const std::vector<Range>& range_list, int64_t topk, int64_t nprobe,
           const std::vector<std::string>& partition_list, const std::vector<std::string>& file_id_list,
           TopKQueryResult& result);

    Status
    CompareFragments(const std::shared_ptr<Context>& context,
                     const engine::CompareFragmentsReq& req, json& resp);

    Status
    GetVectors(const std::shared_ptr<Context>& context, const std::vector<std::string>& table_names,
               engine::VectorsData& vectors);

    Status
    Clusterize(const std::shared_ptr<Context>& context,
               const engine::ClusterizeOptions& opts,
               const engine::VectorsData& vectors);



    Status
    DescribeTable(const std::shared_ptr<Context>& context, const std::string& table_name, TableSchema& table_schema);

    Status
    CountTable(const std::shared_ptr<Context>& context, const std::string& table_name, int64_t& count);

    Status
    Cmd(const std::shared_ptr<Context>& context, const std::string& cmd, std::string& reply);

    Status
    DeleteByRange(const std::shared_ptr<Context>& context, const std::string& table_name, const Range& range);

    Status
    PreloadTable(const std::shared_ptr<Context>& context, const std::string& table_name);

    Status
    DescribeIndex(const std::shared_ptr<Context>& context, const std::string& table_name, IndexParam& param);

    Status
    DropIndex(const std::shared_ptr<Context>& context, const std::string& table_name);

    Status
    CreatePartition(const std::shared_ptr<Context>& context, const std::string& table_name,
                    const std::string& partition_name, const std::string& tag);

    Status
    ShowPartitions(const std::shared_ptr<Context>& context, const std::string& table_name,
                   std::vector<PartitionParam>& partitions);

    Status
    DropPartition(const std::shared_ptr<Context>& context, const std::string& table_name,
                  const std::string& partition_name, const std::string& tag);
};

}  // namespace server
}  // namespace milvus
