#pragma once


#include "Job.h"
#include "db/Types.h"
#include "db/meta/Meta.h"
#include "db/Options.h"


namespace milvus {
namespace scheduler {


class ClusterizeJob : public Job {
 public:
    ClusterizeJob(const std::shared_ptr<server::Context>& context,
                  const engine::ClusterizeOptions& opts,
                  engine::meta::MetaPtr meta_ptr,
                  const engine::meta::TableFilesSchema& direct_files,
                  const engine::VectorsData& vectors);

 public:

    void
    WaitResult();

    void
    Done();

    Status&
    GetStatus();

    json
    Dump() const override;

    engine::meta::TableSchema
    DestTable()const;

 public:
    const std::shared_ptr<server::Context>&
    GetContext() const;


    const engine::VectorsData&
    vectors() {
        return vectors_;
    }
    const engine::meta::TableFilesSchema&
    direct_files()const{
        return direct_files_;
    }

    const engine::ClusterizeOptions&
    options()const{
        return opts_;
    }

    engine::meta::MetaPtr
    meta() const {
        return meta_ptr_;
    }

    std::mutex&
    mutex() {
        return mutex_;
    }

 private:
    const std::shared_ptr<server::Context> context_;
    engine::ClusterizeOptions  opts_;
    engine::meta::MetaPtr meta_ptr_;
    const engine::meta::TableFilesSchema& direct_files_;
    const engine::VectorsData& vectors_;


    // Id2IndexMap index_files_;
    // TODO: column-base better ?
    Status status_;

    bool done_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

using ClusterizeJobPtr = std::shared_ptr<ClusterizeJob>;

}  // namespace scheduler
}  // namespace milvus
