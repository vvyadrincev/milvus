#pragma once

#include "server/delivery/request/BaseRequest.h"

namespace milvus {
namespace server {

class GetVectorsRequest : public BaseRequest {
 public:
    static BaseRequestPtr
    Create(const std::shared_ptr<Context>& context, const std::string& table_name,
           engine::VectorsData& vectors);

 protected:
    GetVectorsRequest(const std::shared_ptr<Context>& context, const std::string& table_name,
                      engine::VectorsData& vectors);

    Status
    OnExecute() override;

 private:
    const std::string table_name_;
    engine::VectorsData& vectors_data_;
};

}  // namespace server
}  // namespace milvus
