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

#include "utils/TimeRecorder.h"
#include "utils/Log.h"

namespace milvus {

TimeRecorder::TimeRecorder(const std::string& header, int64_t log_level) : header_(header), log_level_(log_level) {
    start_ = last_ = stdclock::now();
}

TimeRecorder::~TimeRecorder() = default;

std::string
TimeRecorder::GetTimeSpanStr(double span) {
    std::string str_sec = std::to_string(span * 0.000001) + ((span > 1000000) ? " seconds" : " second");
    std::string str_ms = std::to_string(span * 0.001) + " ms";

    return str_sec + " [" + str_ms + "]";
}

void
TimeRecorder::PrintTimeRecord(const std::string& msg, double span) {
    std::string str_log;
    if (!header_.empty())
        str_log += header_ + ": ";
    str_log += msg;
    str_log += " (";
    str_log += TimeRecorder::GetTimeSpanStr(span);
    str_log += ")";

    switch (log_level_) {
        case 0: {
            SERVER_LOG_TRACE << str_log;
            break;
        }
        case 1: {
            SERVER_LOG_DEBUG << str_log;
            break;
        }
        case 2: {
            SERVER_LOG_INFO << str_log;
            break;
        }
        case 3: {
            SERVER_LOG_WARNING << str_log;
            break;
        }
        case 4: {
            SERVER_LOG_ERROR << str_log;
            break;
        }
        case 5: {
            SERVER_LOG_FATAL << str_log;
            break;
        }
        default: {
            SERVER_LOG_INFO << str_log;
            break;
        }
    }
}

double
TimeRecorder::RecordSection(const std::string& msg) {
    stdclock::time_point curr = stdclock::now();
    double span = (std::chrono::duration<double, std::micro>(curr - last_)).count();
    last_ = curr;

    PrintTimeRecord(msg, span);
    return span;
}

double
TimeRecorder::ElapseFromBegin(const std::string& msg) {
    stdclock::time_point curr = stdclock::now();
    double span = (std::chrono::duration<double, std::micro>(curr - start_)).count();

    PrintTimeRecord(msg, span);
    return span;
}

TimeRecorderAuto::TimeRecorderAuto(const std::string& header, int64_t log_level) : TimeRecorder(header, log_level) {
}

TimeRecorderAuto::~TimeRecorderAuto() {
    ElapseFromBegin("totally cost");
}

}  // namespace milvus
