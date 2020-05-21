#include "ClusterizeJob.h"

#include "utils/Log.h"

namespace milvus {
namespace scheduler {

ClusterizeJob::ClusterizeJob(const std::shared_ptr<server::Context>& context,
                             const engine::ClusterizeOptions& opts,
                             engine::meta::MetaPtr meta_ptr,
                             const engine::meta::TableFilesSchema& direct_files,
                             const engine::VectorsData& vectors)
    : Job(JobType::CLUSTERIZE), context_(context),
      opts_(opts), meta_ptr_(meta_ptr),
      direct_files_(direct_files) , vectors_(vectors),
      done_(false){
}

void
ClusterizeJob::WaitResult() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return done_; });
    SERVER_LOG_DEBUG << "ClusterizeJob " << id() << " all done";
}

engine::meta::TableSchema
ClusterizeJob::
DestTable()const{
    engine::meta::TableSchema table;
    table.table_id_ = opts_.table_id;
    meta_ptr_->DescribeTable(table);
    return table;
}

void
ClusterizeJob::Done() {
    std::unique_lock<std::mutex> lock(mutex_);
    done_ = true;
    cv_.notify_all();
}

Status&
ClusterizeJob::GetStatus() {
    return status_;
}

json
ClusterizeJob::Dump() const {
    json ret{
        {"nq", vectors_.vector_count_},
    };
    auto base = Job::Dump();
    ret.insert(base.begin(), base.end());
    return ret;
}

const std::shared_ptr<server::Context>&
ClusterizeJob::GetContext() const {
    return context_;
}

}  // namespace scheduler
}  // namespace milvus
