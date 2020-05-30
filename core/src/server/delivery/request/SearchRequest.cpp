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

#include "server/delivery/request/SearchRequest.h"
#include "server/DBWrapper.h"
#include "utils/Log.h"
#include "utils/TimeRecorder.h"
#include "utils/ValidationUtil.h"

#include <fiu-local.h>
#include <memory>
#include <numeric>

namespace milvus {
namespace server {

SearchRequest::SearchRequest(const std::shared_ptr<Context>& context,
                             const std::vector<std::string>& table_names,
                             const engine::VectorsData& vectors, const std::vector<Range>& range_list, int64_t topk,
                             int64_t nprobe, const std::vector<std::string>& partition_list,
                             const std::vector<std::string>& file_id_list, TopKQueryResult& result)
    : BaseRequest(context, DQL_REQUEST_GROUP),
      table_names_(table_names),
      vectors_data_(vectors),
      range_list_(range_list),
      topk_(topk),
      nprobe_(nprobe),
      partition_list_(partition_list),
      file_id_list_(file_id_list),
      result_(result) {
}

BaseRequestPtr
SearchRequest::Create(const std::shared_ptr<Context>& context, const std::vector<std::string>& table_names,
                      const engine::VectorsData& vectors, const std::vector<Range>& range_list, int64_t topk,
                      int64_t nprobe, const std::vector<std::string>& partition_list,
                      const std::vector<std::string>& file_id_list, TopKQueryResult& result) {
    return std::shared_ptr<BaseRequest>(new SearchRequest(context, table_names, vectors, range_list, topk, nprobe,
                                                          partition_list, file_id_list, result));
}

Status
SearchRequest::OnExecute() {
    try {
        fiu_do_on("SearchRequest.OnExecute.throw_std_exception", throw std::exception());
        uint64_t vector_count = vectors_data_.vector_count_;
        auto pre_query_ctx = context_->Child("Pre query");

        if (table_names_.empty())
            return Status(SERVER_INVALID_TABLE_NAME, "Pass at least one table name!");

        auto tables = std::accumulate(std::begin(table_names_), std::end(table_names_), std::string{},
                                      [](auto& t, const auto& s) {return t += "," + s;});


        std::string hdr = "SearchRequest(tables=" + tables + ", nq=" + std::to_string(vector_count) +
                          ", k=" + std::to_string(topk_) + ", nprob=" + std::to_string(nprobe_) + ")";

        TimeRecorder rc(hdr);

        // step 1: check table name
        for (const auto& table_name : table_names_){
            auto status = ValidationUtil::ValidateTableName(table_name);
            if (!status.ok()) {
                return status;
            }

            // step 2: check table existence
            engine::meta::TableSchema table_info;
            table_info.table_id_ = table_name;
            status = DBWrapper::DB()->DescribeTable(table_info);
            fiu_do_on("SearchRequest.OnExecute.describe_table_fail", status = Status(milvus::SERVER_UNEXPECTED_ERROR, ""));
            if (!status.ok()) {
                if (status.code() == DB_NOT_FOUND) {
                    return Status(SERVER_TABLE_NOT_EXIST, TableNotExistMsg(table_name));
                } else {
                    return status;
                }
            }

            // step 3: check search parameter
            status = ValidationUtil::ValidateSearchTopk(topk_, table_info);
            if (!status.ok()) {
                return status;
            }

            status = ValidationUtil::ValidateSearchNprobe(nprobe_, table_info);
            if (!status.ok()) {
                return status;
            }

            if (vectors_data_.float_data_.empty() && vectors_data_.binary_data_.empty() &&
                vectors_data_.id_array_.empty()) {
                return Status(SERVER_INVALID_ROWRECORD_ARRAY,
                            "The vector array and query_ids are empty. Make sure you have entered input data.");
            }

            // step 4: check date range, and convert to db dates
            rc.RecordSection("check validation");

            if (ValidationUtil::IsBinaryMetricType(table_info.metric_type_)) {
                // check prepared binary data
                if (vectors_data_.binary_data_.size() % vector_count != 0) {
                    return Status(SERVER_INVALID_ROWRECORD_ARRAY,
                                "The vector dimension must be equal to the table dimension.");
                }

                if (vectors_data_.binary_data_.size() * 8 / vector_count != table_info.dimension_) {
                    return Status(SERVER_INVALID_VECTOR_DIMENSION,
                                "The vector dimension must be equal to the table dimension.");
                }
            } else {
                // check prepared float data
                fiu_do_on("SearchRequest.OnExecute.invalod_rowrecord_array",
                        vector_count = vectors_data_.float_data_.size() + 1);
                if (vectors_data_.float_data_.size() % vector_count != 0) {
                    return Status(SERVER_INVALID_ROWRECORD_ARRAY,
                                "The vector dimension must be equal to the table dimension.");
                }
                fiu_do_on("SearchRequest.OnExecute.invalid_dim", table_info.dimension_ = -1);
                if (not vectors_data_.float_data_.empty() and
                    vectors_data_.float_data_.size() / vector_count != table_info.dimension_) {
                    return Status(SERVER_INVALID_VECTOR_DIMENSION,
                                "The vector dimension must be equal to the table dimension.");
                }
            }

        }
        rc.RecordSection("prepare vector data");

        std::vector<DB_DATE> dates;
        auto status = ConvertTimeRangeToDBDates(range_list_, dates);
        if (!status.ok()) {
            return status;
        }
        // step 6: search vectors
        engine::ResultIds result_ids;
        engine::ResultDistances result_distances;

#ifdef MILVUS_ENABLE_PROFILING
        std::string fname =
            "/tmp/search_nq_" + std::to_string(this->search_param_->query_record_array_size()) + ".profiling";
        ProfilerStart(fname.c_str());
#endif

        pre_query_ctx->GetTraceContext()->GetSpan()->Finish();

        if (file_id_list_.empty()) {
            status = ValidationUtil::ValidatePartitionTags(partition_list_);
            fiu_do_on("SearchRequest.OnExecute.invalid_partition_tags",
                      status = Status(milvus::SERVER_UNEXPECTED_ERROR, ""));
            if (!status.ok()) {
                return status;
            }

            status = DBWrapper::DB()->Query(context_, table_names_, partition_list_, (size_t)topk_, nprobe_,
                                            vectors_data_, dates, result_ids, result_distances);

        } else {
            status = DBWrapper::DB()->QueryByFileID(context_, table_names_.front(),
                                                    file_id_list_, (size_t)topk_, nprobe_,
                                                    vectors_data_, dates, result_ids, result_distances);
        }

#ifdef MILVUS_ENABLE_PROFILING
        ProfilerStop();
#endif

        rc.RecordSection("search vectors from engine");
        fiu_do_on("SearchRequest.OnExecute.query_fail", status = Status(milvus::SERVER_UNEXPECTED_ERROR, ""));
        if (!status.ok()) {
            return status;
        }
        fiu_do_on("SearchRequest.OnExecute.empty_result_ids", result_ids.clear());
        if (result_ids.empty()) {
            return Status::OK();  // empty table
        }

        auto post_query_ctx = context_->Child("Constructing result");

        // step 7: construct result array
        result_.row_num_ = vector_count;
        result_.distance_list_ = result_distances;
        result_.id_list_ = result_ids;

        post_query_ctx->GetTraceContext()->GetSpan()->Finish();

        // step 8: print time cost percent
        rc.RecordSection("construct result and send");
        rc.ElapseFromBegin("totally cost");
    } catch (std::exception& ex) {
        return Status(SERVER_UNEXPECTED_ERROR, ex.what());
    }

    return Status::OK();
}

}  // namespace server
}  // namespace milvus
