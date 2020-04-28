#include "GetVectorsRequest.h"

#include "server/DBWrapper.h"
#include "utils/ValidationUtil.h"

namespace milvus {
namespace server {

GetVectorsRequest::GetVectorsRequest(const std::shared_ptr<Context>& context, const std::string& table_name,
                             engine::VectorsData& vectors)
    : BaseRequest(context, DQL_REQUEST_GROUP),
      table_name_(table_name),
      vectors_data_(vectors)
{}

BaseRequestPtr
GetVectorsRequest::Create(const std::shared_ptr<Context>& context, const std::string& table_name,
                          engine::VectorsData& vectors) {
    return std::shared_ptr<BaseRequest>(new GetVectorsRequest(context, table_name, vectors));
}

Status
GetVectorsRequest::OnExecute() {
    try {

        // step 1: check table name
        auto status = ValidationUtil::ValidateTableName(table_name_);
        if (!status.ok()) {
            return status;
        }

        // step 2: check table existence
        engine::meta::TableSchema table_info;
        table_info.table_id_ = table_name_;
        status = DBWrapper::DB()->DescribeTable(table_info);
        if (!status.ok()) {
            if (status.code() == DB_NOT_FOUND) {
                return Status(SERVER_TABLE_NOT_EXIST, TableNotExistMsg(table_name_));
            } else {
                return status;
            }
        }


        // step 3: get vectors

        status = DBWrapper::DB()->GetVectors(context_, table_name_, vectors_data_);

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
