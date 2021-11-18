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

#ifndef UTILS_ERROR_H
#define UTILS_ERROR_H

#include <cstdint>
#include <exception>
#include <string>

namespace milvus {

using ErrorCode = int32_t;

constexpr ErrorCode SERVER_SUCCESS = 0;
constexpr ErrorCode SERVER_ERROR_CODE_BASE = 30000;

constexpr ErrorCode
ToServerErrorCode(const ErrorCode error_code) {
    return SERVER_ERROR_CODE_BASE + error_code;
}

constexpr ErrorCode DB_SUCCESS = 0;
constexpr ErrorCode DB_ERROR_CODE_BASE = 40000;

constexpr ErrorCode
ToDbErrorCode(const ErrorCode error_code) {
    return DB_ERROR_CODE_BASE + error_code;
}

constexpr ErrorCode KNOWHERE_SUCCESS = 0;
constexpr ErrorCode KNOWHERE_ERROR_CODE_BASE = 50000;

constexpr ErrorCode
ToKnowhereErrorCode(const ErrorCode error_code) {
    return KNOWHERE_ERROR_CODE_BASE + error_code;
}

// server error code
constexpr ErrorCode SERVER_UNEXPECTED_ERROR = ToServerErrorCode(1);
constexpr ErrorCode SERVER_UNSUPPORTED_ERROR = ToServerErrorCode(2);
constexpr ErrorCode SERVER_NULL_POINTER = ToServerErrorCode(3);
constexpr ErrorCode SERVER_INVALID_ARGUMENT = ToServerErrorCode(4);
constexpr ErrorCode SERVER_FILE_NOT_FOUND = ToServerErrorCode(5);
constexpr ErrorCode SERVER_NOT_IMPLEMENT = ToServerErrorCode(6);
constexpr ErrorCode SERVER_CANNOT_CREATE_FOLDER = ToServerErrorCode(8);
constexpr ErrorCode SERVER_CANNOT_CREATE_FILE = ToServerErrorCode(9);
constexpr ErrorCode SERVER_CANNOT_DELETE_FOLDER = ToServerErrorCode(10);
constexpr ErrorCode SERVER_CANNOT_DELETE_FILE = ToServerErrorCode(11);
constexpr ErrorCode SERVER_BUILD_INDEX_ERROR = ToServerErrorCode(12);

constexpr ErrorCode SERVER_TABLE_NOT_EXIST = ToServerErrorCode(100);
constexpr ErrorCode SERVER_INVALID_TABLE_NAME = ToServerErrorCode(101);
constexpr ErrorCode SERVER_INVALID_TABLE_DIMENSION = ToServerErrorCode(102);
constexpr ErrorCode SERVER_INVALID_TIME_RANGE = ToServerErrorCode(103);
constexpr ErrorCode SERVER_INVALID_VECTOR_DIMENSION = ToServerErrorCode(104);
constexpr ErrorCode SERVER_INVALID_INDEX_TYPE = ToServerErrorCode(105);
constexpr ErrorCode SERVER_INVALID_ROWRECORD = ToServerErrorCode(106);
constexpr ErrorCode SERVER_INVALID_ROWRECORD_ARRAY = ToServerErrorCode(107);
constexpr ErrorCode SERVER_INVALID_TOPK = ToServerErrorCode(108);
constexpr ErrorCode SERVER_ILLEGAL_VECTOR_ID = ToServerErrorCode(109);
constexpr ErrorCode SERVER_ILLEGAL_SEARCH_RESULT = ToServerErrorCode(110);
constexpr ErrorCode SERVER_CACHE_FULL = ToServerErrorCode(111);
constexpr ErrorCode SERVER_WRITE_ERROR = ToServerErrorCode(112);
constexpr ErrorCode SERVER_INVALID_NPROBE = ToServerErrorCode(113);
constexpr ErrorCode SERVER_INVALID_INDEX_NLIST = ToServerErrorCode(114);
constexpr ErrorCode SERVER_INVALID_INDEX_METRIC_TYPE = ToServerErrorCode(115);
constexpr ErrorCode SERVER_INVALID_INDEX_FILE_SIZE = ToServerErrorCode(116);
constexpr ErrorCode SERVER_OUT_OF_MEMORY = ToServerErrorCode(117);

// db error code
constexpr ErrorCode DB_META_TRANSACTION_FAILED = ToDbErrorCode(1);
constexpr ErrorCode DB_ERROR = ToDbErrorCode(2);
constexpr ErrorCode DB_NOT_FOUND = ToDbErrorCode(3);
constexpr ErrorCode DB_ALREADY_EXIST = ToDbErrorCode(4);
constexpr ErrorCode DB_INVALID_PATH = ToDbErrorCode(5);
constexpr ErrorCode DB_INCOMPATIB_META = ToDbErrorCode(6);
constexpr ErrorCode DB_INVALID_META_URI = ToDbErrorCode(7);

// knowhere error code
constexpr ErrorCode KNOWHERE_ERROR = ToKnowhereErrorCode(1);
constexpr ErrorCode KNOWHERE_INVALID_ARGUMENT = ToKnowhereErrorCode(2);
constexpr ErrorCode KNOWHERE_UNEXPECTED_ERROR = ToKnowhereErrorCode(3);
constexpr ErrorCode KNOWHERE_NO_SPACE = ToKnowhereErrorCode(4);

namespace server {
class ServerException : public std::exception {
 public:
    explicit ServerException(ErrorCode error_code, const std::string& message = std::string())
        : error_code_(error_code), message_(message) {
    }

 public:
    ErrorCode
    error_code() const {
        return error_code_;
    }

    virtual const char*
    what() const noexcept {
        return message_.c_str();
    }

 private:
    ErrorCode error_code_;
    std::string message_;
};
}  // namespace server

}  // namespace milvus

#endif // UTILS_ERROR_H
