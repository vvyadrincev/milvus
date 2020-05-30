#pragma once

#include "server/delivery/request/BaseRequest.h"

namespace milvus {
namespace server {

class GetVectorsRequest : public BaseRequest {
 public:
    static BaseRequestPtr
    Create(const std::shared_ptr<Context>& context, const std::vector<std::string>& table_names,
           engine::VectorsData& vectors);

 protected:
    GetVectorsRequest(const std::shared_ptr<Context>& context,
                      const std::vector<std::string>& table_names,
                      engine::VectorsData& vectors);

    Status
    OnExecute() override;

 private:
    const std::vector<std::string>& table_names_;
    engine::VectorsData& vectors_data_;
};

}  // namespace server
}  // namespace milvus
