#pragma once

#include "server/delivery/request/BaseRequest.h"


namespace milvus {
namespace server {

class CompareFragmentsRequest : public BaseRequest {
 public:
    static BaseRequestPtr
    Create(const std::shared_ptr<Context>& context,
           const engine::CompareFragmentsReq& req,
           json& resp);

 protected:
    CompareFragmentsRequest(const std::shared_ptr<Context>& context,
                            const engine::CompareFragmentsReq& req,
                            json& resp);

    Status
    OnExecute() override;

 private:
    const engine::CompareFragmentsReq& req_;
    json& resp_;
};

}  // namespace server
}  // namespace milvus
