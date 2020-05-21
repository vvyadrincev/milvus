#pragma once

#include "server/delivery/request/BaseRequest.h"
#include "db/Options.h"

namespace milvus {
namespace server {

class ClusterizeRequest : public BaseRequest {
 public:
    static BaseRequestPtr
    Create(const std::shared_ptr<Context>& context,
           const engine::ClusterizeOptions& opts,
           const engine::VectorsData& vectors);

 protected:
    ClusterizeRequest(const std::shared_ptr<Context>& context,
                      const engine::ClusterizeOptions& opts,
                      const engine::VectorsData& vectors);

    Status
    OnExecute() override;

 private:
    const engine::ClusterizeOptions& opts_;
    const engine::VectorsData& vectors_data_;
};

}  // namespace server
}  // namespace milvus
