#include "ClusterizePass.h"

#include "server/config.h"
#include "scheduler/SchedInst.h"
#include "scheduler/task/ClusterizeTask.h"
#include "scheduler/tasklabel/SpecResLabel.h"

namespace milvus {
namespace scheduler {

void
FaissClusterizePass::Init() {
    server::Config& config = server::Config::GetInstance();
    auto s = config.GetGpuResourceConfigBuildIndexResources(gpus_);
    if (!s.ok()) {
        throw std::exception();
    }
}

bool
FaissClusterizePass::Run(const TaskPtr& task) {
    SERVER_LOG_DEBUG << "FaissClusterizePass: "<<(int)task->Type();
    if (task->Type() != TaskType::ClusterizeTask) {
        return false;
    }

    auto cluster_task = std::static_pointer_cast<XClusterizeTask>(task);

    auto cluster_job = cluster_task->Job();

    ResourcePtr res_ptr;
    if (not cluster_job->options().use_gpu) {
        SERVER_LOG_DEBUG << "FaissClusterizePass: specify cpu to search!";
        res_ptr = ResMgrInst::GetInstance()->GetResource("cpu");
    } else {
        auto best_device_id = count_ % gpus_.size();
        SERVER_LOG_DEBUG << "FaissClusterizePass: specify gpu" << best_device_id << " to search!";
        ++count_;
        res_ptr = ResMgrInst::GetInstance()->GetResource(ResourceType::GPU, gpus_[best_device_id]);
    }
    auto label = std::make_shared<SpecResLabel>(res_ptr);
    task->label() = label;
    return true;
}

}  // namespace scheduler
}  // namespace milvus
