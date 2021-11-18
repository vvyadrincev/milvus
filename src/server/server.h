// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#ifndef VEC_INDEXER_SERVER_H
#define VEC_INDEXER_SERVER_H

#include <string>
#include "utils/Status.h"

namespace milvus {
namespace server {

class Server {
 public:
    static Server&
    GetInstance();

    void
    Init(int64_t daemonized, const std::string& pid_filename, const std::string& config_filename,
         const std::string& log_config_file);

    Status
    Start();
    void
    Stop();

 private:
    Server() = default;
    ~Server() = default;

    void
    Daemonize();

    Status
    LoadConfig();

    void
    StartService();
    void
    StopService();

 private:
    int64_t daemonized_ = 0;
    int pid_fd_ = -1;
    std::string pid_filename_;
    std::string config_filename_;
    std::string log_config_file_;
};  // Server

}  // namespace server
}  // namespace milvus

#endif // VEC_INDEXER_SERVER_H
