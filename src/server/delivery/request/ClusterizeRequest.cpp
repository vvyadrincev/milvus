#include "ClusterizeRequest.h"

#include "server/DBWrapper.h"
#include "utils/ValidationUtil.h"

namespace milvus {
namespace server {

ClusterizeRequest::ClusterizeRequest(const std::shared_ptr<Context>& context,
                                     const engine::ClusterizeOptions& opts,
                                     const engine::VectorsData& vectors)
    : BaseRequest(context, DQL_REQUEST_GROUP),
      opts_(opts),
      vectors_data_(vectors)
{}

BaseRequestPtr
ClusterizeRequest::Create(const std::shared_ptr<Context>& context,
                          const engine::ClusterizeOptions& opts,
                          const engine::VectorsData& vectors) {
    return std::shared_ptr<BaseRequest>(new ClusterizeRequest(context, opts, vectors));
}

Status
ClusterizeRequest::OnExecute() {
    try {

        auto status = DBWrapper::DB()->Clusterize(context_, opts_, vectors_data_);

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
