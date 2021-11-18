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

#include "db/meta/MetaFactory.h"
#include "SqliteMetaImpl.h"
#include "db/Utils.h"
#include "utils/Exception.h"
#include "utils/Log.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>

namespace milvus {
namespace engine {

DBMetaOptions
MetaFactory::BuildOption(const std::string& path) {
    auto p = path;
    if (p == "") {
        srand(time(nullptr));
        std::stringstream ss;
        uint32_t seed = 1;
        ss << "/tmp/" << rand_r(&seed);
        p = ss.str();
    }

    DBMetaOptions meta;
    meta.path_ = p;
    return meta;
}

meta::MetaPtr
MetaFactory::Build(const DBMetaOptions& metaOptions, const int& mode) {
    std::string uri = metaOptions.backend_uri_;

    utils::MetaUriInfo uri_info;
    auto status = utils::ParseMetaUri(uri, uri_info);
    if (!status.ok()) {
        ENGINE_LOG_ERROR << "Wrong URI format: URI = " << uri;
        throw InvalidArgumentException("Wrong URI format ");
    }

    if (strcasecmp(uri_info.dialect_.c_str(), "sqlite") == 0) {
        ENGINE_LOG_INFO << "Using SQLite";
        return std::make_shared<meta::SqliteMetaImpl>(metaOptions);
    } else {
        ENGINE_LOG_ERROR << "Invalid dialect in URI: dialect = " << uri_info.dialect_;
        throw InvalidArgumentException("URI dialect is not mysql / sqlite");
    }
}

}  // namespace engine
}  // namespace milvus
