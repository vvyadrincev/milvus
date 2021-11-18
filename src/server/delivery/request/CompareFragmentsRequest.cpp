#include "CompareFragmentsRequest.h"

#include "server/DBWrapper.h"
#include "utils/ValidationUtil.h"

namespace milvus {
namespace server {

CompareFragmentsRequest::CompareFragmentsRequest(const std::shared_ptr<Context>& context,
                                                 const engine::CompareFragmentsReq& req,
                                                 json& resp)
    : BaseRequest(context, DQL_REQUEST_GROUP),
      req_(req),
      resp_(resp)
{}

BaseRequestPtr
CompareFragmentsRequest::Create(const std::shared_ptr<Context>& context,
                                const engine::CompareFragmentsReq& req,
                                json& resp) {
    return std::shared_ptr<BaseRequest>(new CompareFragmentsRequest(context, req, resp));
}

Status
CompareFragmentsRequest::OnExecute() {
    try {

        auto status = DBWrapper::DB()->CompareFragments(req_, resp_);

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
