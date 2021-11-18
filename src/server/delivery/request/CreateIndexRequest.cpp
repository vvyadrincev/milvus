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

#include "server/delivery/request/CreateIndexRequest.h"
#include "server/config.h"
#include "server/DBWrapper.h"
#include "utils/Log.h"
#include "utils/TimeRecorder.h"
#include "utils/ValidationUtil.h"

#include <fiu-local.h>
#include <memory>
#include <string>

namespace milvus {
namespace server {

CreateIndexRequest::CreateIndexRequest(const std::shared_ptr<Context>& context, const std::string& table_name,
                                       int64_t index_type, int64_t nlist,
                                       const std::string& enc_type )
    : BaseRequest(context, DDL_DML_REQUEST_GROUP), table_name_(table_name), index_type_(index_type),
      nlist_(nlist), enc_type_(enc_type) {
}

BaseRequestPtr
CreateIndexRequest::Create(const std::shared_ptr<Context>& context, const std::string& table_name, int64_t index_type,
                           int64_t nlist, const std::string& enc_type) {
    return std::shared_ptr<BaseRequest>(new CreateIndexRequest(context, table_name, index_type,
                                                               nlist, enc_type));
}

Status
CreateIndexRequest::OnExecute() {
    try {
        std::string hdr = "CreateIndexRequest(table=" + table_name_ + ")";
        TimeRecorderAuto rc(hdr);

        // step 1: check arguments
        auto status = ValidationUtil::ValidateTableName(table_name_);
        if (!status.ok()) {
            return status;
        }

        bool has_table = false;
        status = DBWrapper::DB()->HasTable(table_name_, has_table);
        fiu_do_on("CreateIndexRequest.OnExecute.not_has_table", status = Status(milvus::SERVER_UNEXPECTED_ERROR, ""));
        fiu_do_on("CreateIndexRequest.OnExecute.throw_std.exception", throw std::exception());
        if (!status.ok()) {
            return status;
        }

        if (!has_table) {
            return Status(SERVER_TABLE_NOT_EXIST, TableNotExistMsg(table_name_));
        }

        status = ValidationUtil::ValidateTableIndexType(index_type_);
        if (!status.ok()) {
            return status;
        }

        status = ValidationUtil::ValidateTableIndexNlist(nlist_);
        if (!status.ok()) {
            return status;
        }

        // step 2: binary and float vector support different index/metric type, need to adapt here
        engine::meta::TableSchema table_info;
        table_info.table_id_ = table_name_;
        status = DBWrapper::DB()->DescribeTable(table_info);

        int32_t adapter_index_type = index_type_;
        if (ValidationUtil::IsBinaryMetricType(table_info.metric_type_)) {  // binary vector not allow
            if (adapter_index_type == static_cast<int32_t>(engine::EngineType::FAISS_IDMAP)) {
                adapter_index_type = static_cast<int32_t>(engine::EngineType::FAISS_BIN_IDMAP);
            } else if (adapter_index_type == static_cast<int32_t>(engine::EngineType::FAISS_IVFFLAT)) {
                adapter_index_type = static_cast<int32_t>(engine::EngineType::FAISS_BIN_IVFFLAT);
            } else {
                return Status(SERVER_INVALID_INDEX_TYPE, "Invalid index type for table metric type");
            }
        }

#ifdef MILVUS_GPU_VERSION
        Status s;
        bool enable_gpu = false;
        server::Config& config = server::Config::GetInstance();
        s = config.GetGpuResourceConfigEnable(enable_gpu);
        fiu_do_on("CreateIndexRequest.OnExecute.ip_meteric",
                  table_info.metric_type_ = static_cast<int>(engine::MetricType::IP));

        if (s.ok() && adapter_index_type == (int)engine::EngineType::FAISS_PQ &&
            table_info.metric_type_ == (int)engine::MetricType::IP) {
            return Status(SERVER_UNEXPECTED_ERROR, "PQ not support IP in GPU version!");
        }
#endif

        // step 3: create index
        engine::TableIndex index;
        index.engine_type_ = adapter_index_type;
        index.nlist_ = nlist_;
        index.enc_type_ = enc_type_;
        status = DBWrapper::DB()->CreateIndex(table_name_, index);
        fiu_do_on("CreateIndexRequest.OnExecute.create_index_fail",
                  status = Status(milvus::SERVER_UNEXPECTED_ERROR, ""));
        if (!status.ok()) {
            return status;
        }
    } catch (std::exception& ex) {
        return Status(SERVER_UNEXPECTED_ERROR, ex.what());
    }

    return Status::OK();
}

}  // namespace server
}  // namespace milvus
