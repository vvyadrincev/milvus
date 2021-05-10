#pragma once


#include <zmq.hpp>

#include <opentracing/tracer.h>

#include "server/Config.h"
#include "server/delivery/RequestHandler.h"
#include "cbor_utils.hpp"
#include "utils/Json.h"

namespace milvus {
namespace server {
namespace zeromq {


class ServerOpts{
public:
    std::string endpoint;
    int         threads = 6;

};


class ZeroMQServer{

public:

    static ZeroMQServer&
    GetInstance() {

        Config& config = Config::GetInstance();
        std::string address, port;
        Status s;

        s = config.GetServerConfigAddress(address);
        if (!s.ok()) {throw std::runtime_error("Failed to get addr!");}
        s = config.GetServerConfigPort(port);
        if (!s.ok()) {throw std::runtime_error("Failed to get port!");}

        std::string server_address(address + ":" + port);
        static ZeroMQServer server(ServerOpts{"tcp://" + server_address, 6});
        return server;
    }

    ZeroMQServer(const ServerOpts& opts):opts_(opts),
                                         zmq_context_(1),
                                         tracer_(opentracing::Tracer::Global())
    {}


    void Start();
    void Stop();

protected:

    void main_loop();
    std::string recv_str(zmq::socket_t& socket);

    void try_process(zmq::socket_t& socket);

    void process();

    std::vector<uint8_t> handle_req(const zmq::message_t& msg);
    std::vector<uint8_t> handle_add(const std::shared_ptr<Context>& pctx, Unpacker& unpacker);
    std::vector<uint8_t> handle_create_index(const std::shared_ptr<Context>& pctx, Unpacker& unpacker);
    std::vector<uint8_t> handle_drop_table(const std::shared_ptr<Context>& pctx, Unpacker& unpacker);
    std::vector<uint8_t> handle_search(const std::shared_ptr<Context>& pctx, Unpacker& unpacker);
    std::vector<uint8_t> handle_search_by_id(const std::shared_ptr<Context>& pctx, Unpacker& unpacker);
    std::vector<uint8_t> search_impl(const std::shared_ptr<Context>& pctx,
                                     engine::VectorsData&& vectors,
                                     Unpacker& unpacker);
    std::vector<uint8_t> handle_compare_fragments_by_id(const std::shared_ptr<Context>& pctx,
                                                        Unpacker& unpacker);
    std::vector<uint8_t> handle_compare_fragments(const std::shared_ptr<Context>& pctx,
                                                  Unpacker& unpacker);
    std::vector<uint8_t> compare_fragments_impl(const std::shared_ptr<Context>& pctx,
                                                engine::CompareFragmentsReq&& req,
                                                Unpacker& unpacker);

    std::vector<uint8_t> handle_get_vectors(const std::shared_ptr<Context>& pctx, Unpacker& unpacker);
    std::vector<uint8_t> handle_clusterize(const std::shared_ptr<Context>& pctx, Unpacker& unpacker);


    Status create_table(const std::shared_ptr<Context>& pctx,
                        const std::string& table_name, const json& params);
    Status create_index(const std::shared_ptr<Context>& pctx,
                        const std::string& table_name, const json& params);

    ServerOpts                           opts_;
    zmq::context_t                       zmq_context_;

    RequestHandler                       request_handler_;
    std::shared_ptr<opentracing::Tracer> tracer_;

};


}}}
