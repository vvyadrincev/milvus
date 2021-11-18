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

#include "db/OngoingFileChecker.h"
#include "utils/Log.h"

#include <utility>

namespace milvus {
namespace engine {

Status
OngoingFileChecker::MarkOngoingFile(const meta::TableFileSchema& table_file) {
    std::lock_guard<std::mutex> lck(mutex_);
    return MarkOngoingFileNoLock(table_file);
}

Status
OngoingFileChecker::MarkOngoingFiles(const meta::TableFilesSchema& table_files) {
    std::lock_guard<std::mutex> lck(mutex_);

    for (auto& table_file : table_files) {
        MarkOngoingFileNoLock(table_file);
    }

    return Status::OK();
}

Status
OngoingFileChecker::UnmarkOngoingFile(const meta::TableFileSchema& table_file) {
    std::lock_guard<std::mutex> lck(mutex_);
    return UnmarkOngoingFileNoLock(table_file);
}

Status
OngoingFileChecker::UnmarkOngoingFiles(const meta::TableFilesSchema& table_files) {
    std::lock_guard<std::mutex> lck(mutex_);

    for (auto& table_file : table_files) {
        UnmarkOngoingFileNoLock(table_file);
    }

    return Status::OK();
}

bool
OngoingFileChecker::IsIgnored(const meta::TableFileSchema& schema) {
    std::lock_guard<std::mutex> lck(mutex_);

    auto iter = ongoing_files_.find(schema.table_id_);
    if (iter == ongoing_files_.end()) {
        return false;
    } else {
        auto it_file = iter->second.find(schema.file_id_);
        if (it_file == iter->second.end()) {
            return false;
        } else {
            return (it_file->second > 0);
        }
    }
}

Status
OngoingFileChecker::MarkOngoingFileNoLock(const meta::TableFileSchema& table_file) {
    if (table_file.table_id_.empty() || table_file.file_id_.empty()) {
        return Status(DB_ERROR, "Invalid table files");
    }

    auto iter = ongoing_files_.find(table_file.table_id_);
    if (iter == ongoing_files_.end()) {
        File2RefCount files_refcount;
        files_refcount.insert(std::make_pair(table_file.file_id_, 1));
        ongoing_files_.insert(std::make_pair(table_file.table_id_, files_refcount));
    } else {
        auto it_file = iter->second.find(table_file.file_id_);
        if (it_file == iter->second.end()) {
            iter->second[table_file.file_id_] = 1;
        } else {
            it_file->second++;
        }
    }

    ENGINE_LOG_DEBUG << "Mark ongoing file:" << table_file.file_id_
                     << " refcount:" << ongoing_files_[table_file.table_id_][table_file.file_id_];

    return Status::OK();
}

Status
OngoingFileChecker::UnmarkOngoingFileNoLock(const meta::TableFileSchema& table_file) {
    if (table_file.table_id_.empty() || table_file.file_id_.empty()) {
        return Status(DB_ERROR, "Invalid table files");
    }

    auto iter = ongoing_files_.find(table_file.table_id_);
    if (iter != ongoing_files_.end()) {
        auto it_file = iter->second.find(table_file.file_id_);
        if (it_file != iter->second.end()) {
            it_file->second--;

            ENGINE_LOG_DEBUG << "Unmark ongoing file:" << table_file.file_id_ << " refcount:" << it_file->second;

            if (it_file->second <= 0) {
                iter->second.erase(table_file.file_id_);
                if (iter->second.empty()) {
                    ongoing_files_.erase(table_file.table_id_);
                }
            }
        }
    }

    return Status::OK();
}

}  // namespace engine
}  // namespace milvus
