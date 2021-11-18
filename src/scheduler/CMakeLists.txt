add_library(scheduler
    Algorithm.cpp
    BuildMgr.cpp
    JobMgr.cpp
    ResourceFactory.cpp
    ResourceMgr.cpp
    SchedInst.cpp
    Scheduler.cpp
    TaskCreator.cpp
    TaskTable.cpp
    Utils.cpp

    CircleQueue.h

    event/Event.h
    event/TaskTableUpdatedEvent.h

    interface/interfaces.h

    tasklabel/BroadcastLabel.h
    tasklabel/SpecResLabel.h
    )

target_include_directories(scheduler PUBLIC
    ${INCLUDE_DIRECTORIES}
    )

target_link_libraries(scheduler PUBLIC
    scheduler_job
    scheduler_task
    scheduler_optimizer
    scheduler_resource
    scheduler_action

    cache
    server_config
    utils
    )

add_subdirectory(action)
add_subdirectory(job)
add_subdirectory(optimizer)
add_subdirectory(resource)
add_subdirectory(task)
