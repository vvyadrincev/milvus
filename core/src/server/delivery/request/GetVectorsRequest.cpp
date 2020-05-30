#include "GetVectorsRequest.h"

#include "server/DBWrapper.h"
#include "utils/ValidationUtil.h"

namespace milvus {
namespace server {

GetVectorsRequest::GetVectorsRequest(const std::shared_ptr<Context>& context,
                                     const std::vector<std::string>& table_names,
                             engine::VectorsData& vectors)
    : BaseRequest(context, DQL_REQUEST_GROUP),
      table_names_(table_names),
      vectors_data_(vectors)
{}

BaseRequestPtr
GetVectorsRequest::Create(const std::shared_ptr<Context>& context,
                          const std::vector<std::string>& table_names,
                          engine::VectorsData& vectors) {
    return std::shared_ptr<BaseRequest>(new GetVectorsRequest(context, table_names, vectors));
}

Status
GetVectorsRequest::OnExecute() {
    try {
        if (table_names_.empty())
            return Status(SERVER_INVALID_TABLE_NAME, "Pass at least one table name!");

        for (const auto& table_name : table_names_) {
            // step 1: check table name
            auto status = ValidationUtil::ValidateTableName(table_name);
            if (!status.ok()) {
                return status;
            }

            // step 2: check table existence
            engine::meta::TableSchema table_info;
            table_info.table_id_ = table_name;
            status = DBWrapper::DB()->DescribeTable(table_info);
            if (!status.ok()) {
                if (status.code() == DB_NOT_FOUND) {
                    return Status(SERVER_TABLE_NOT_EXIST, TableNotExistMsg(table_name));
                } else {
                    return status;
                }
            }
        }

        // step 3: get vectors

        auto status = DBWrapper::DB()->GetVectors(context_, table_names_, vectors_data_);

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
