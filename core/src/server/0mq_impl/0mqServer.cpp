#include "0mqServer.h"
#include "cbor_utils.hpp"

#include "db/Options.h"

#include <faiss/utils/distances.h>
#include <boost/format.hpp>

#include <thread>
#include <iostream>
#include <queue>

namespace milvus {
namespace server {
namespace zeromq {



void ZeroMQServer::Start(){
    std::thread(&ZeroMQServer::main_loop, this).detach();
}
void ZeroMQServer::main_loop(){

    zmq::socket_t clients (zmq_context_, zmq::socket_type::router);
    clients.bind(opts_.endpoint);

    zmq::socket_t workers (zmq_context_, zmq::socket_type::router);
    workers.bind ("inproc://workers");

    zmq::socket_t control (zmq_context_, zmq::socket_type::pair);
    control.bind ("inproc://control");

    for(int i = 0; i != opts_.threads; ++i)
        std::thread(&ZeroMQServer::process, this).detach();

    std::queue<std::string> worker_queue;

    while(true){
        zmq::pollitem_t items [] = {
            //control
            { control, 0, ZMQ_POLLIN, 0 },
            //  Always poll for worker activity on backend
            { workers, 0, ZMQ_POLLIN, 0 },
            //  Poll front-end only if we have available workers
            { clients, 0, ZMQ_POLLIN, 0 }
        };


        int items_check = 3;
        if (worker_queue.empty())
            items_check -= 1;

        try{
            zmq::poll(&items[0], items_check, -1);
            // std::cout<<"out of poll"<<std::endl;
        }catch(const zmq::error_t& e){
            std::cout<<"zmq error "<<e.what()<<std::endl;

            if (e.num() == EINTR)
                break;
            std::cerr<<"poll error: "<<"num - "<<e.num()<<" "<<e.what()<<std::endl;
        }

        if (items[0].revents & ZMQ_POLLIN)
            break;

        //  Handle worker activity on backend
        if (items[1].revents & ZMQ_POLLIN) {

            //  Queue worker address for LRU routing
            worker_queue.push(recv_str(workers));

            {
                //  Second frame is empty
                std::string empty = recv_str(workers);
                assert(empty.size() == 0);
            }

            //  Third frame is READY or else a client reply address
            std::string client_addr = recv_str(workers);

            //  If client reply, send rest back to frontend
            if (client_addr.compare("READY") != 0) {
                {
                    std::string empty = recv_str(workers);
                    assert(empty.size() == 0);
                }

                zmq::message_t resp;
                workers.recv(resp);
                clients.send(zmq::buffer(client_addr.data(), client_addr.size()),
                             zmq::send_flags::sndmore);
                clients.send(zmq::message_t(), zmq::send_flags::sndmore);
                clients.send(std::move(resp), zmq::send_flags::none);

            }

        }
        if (items[2].revents & ZMQ_POLLIN) {

            //  Now get next client request, route to LRU worker
            //  Client request is [address][empty][request]
            std::string client_addr = recv_str(clients);
            // std::cout<<"client addr "<<client_addr<<std::endl;

            {
                std::string empty = recv_str(clients);
                assert(empty.size() == 0);
            }

            zmq::message_t req;
            clients.recv(req);

            std::string worker_addr = worker_queue.front();
            // std::cout<<"worker addr: "<<worker_addr<<std::endl;
            worker_queue.pop();

            workers.send(zmq::buffer(worker_addr.data(), worker_addr.size()),
                         zmq::send_flags::sndmore);
            workers.send(zmq::message_t(), zmq::send_flags::sndmore);
            workers.send(zmq::buffer(client_addr.data(), client_addr.size()),
                         zmq::send_flags::sndmore);
            workers.send(zmq::message_t(), zmq::send_flags::sndmore);
            workers.send(std::move(req), zmq::send_flags::none);

        }
    }

    std::cout<<"Destroy everything!"<<std::endl;

}
void ZeroMQServer::Stop(){
    try{
        zmq::socket_t socket (zmq_context_, zmq::socket_type::pair);
        socket.connect("inproc://control");
        socket.send(zmq::message_t("TERMINATE", 9), zmq::send_flags::none);
    }catch(const std::exception& e){
        std::cerr<<"Excep "<<e.what()<<std::endl;
    }
}


std::string ZeroMQServer::recv_str(zmq::socket_t& socket){
    zmq::message_t resp;
    socket.recv(resp);
    return std::string(resp.data<const char>(), resp.size());
}

void ZeroMQServer::try_process(zmq::socket_t& socket){

    auto client_addr = recv_str(socket);

    {
        //  Second frame is empty
        std::string empty = recv_str(socket);
        assert(empty.size() == 0);
    }

    zmq::message_t request;
    socket.recv(request);


    std::vector<std::uint8_t> buf;
    try{
        buf = handle_req(request);
    }catch(const std::exception& e){
        json err_resp = {{"error", {{"msg", e.what()}, {"code", 255}}}};
        buf = json::to_cbor(err_resp);
    }



    socket.send(zmq::buffer(client_addr.data(), client_addr.size()),
                zmq::send_flags::sndmore);
    socket.send(zmq::message_t(), zmq::send_flags::sndmore);
    socket.send(zmq::buffer(buf.data(), buf.size()));
}

void ZeroMQServer::process(){

    zmq::socket_t socket (zmq_context_, zmq::socket_type::req);
    socket.connect ("inproc://workers");

    std::cout << "TID: "<<std::this_thread::get_id()<<" ON duty!!"<<std::endl;
    //  Tell backend we're ready for work
    socket.send(zmq::message_t("READY", 5), zmq::send_flags::none);
    while (true) {

        try{
            try_process(socket);
        }catch(const zmq::error_t& e){
            if (e.num() == ETERM)
                break;
            std::cerr<<"Error ("<<e.num()<<") in worker loop: "<<e.what()<<std::endl;
        }
    }
    std::cout<<"end of worker: "<<std::this_thread::get_id()<<std::endl;

}


std::vector<std::uint8_t>
ZeroMQServer::
handle_req(const zmq::message_t& msg){

    Unpacker unpacker(msg.data<uint8_t>(), msg.size());
    auto method = unpacker.unpack<string_decoder_t>().copy();
    auto req_id = unpacker.unpack<string_decoder_t>().copy();

    //TODO pass carrier from the client
    // auto span = tracer_->StartSpan(server_rpc_info->method(), {opentracing::ChildOf(span_context_maybe->get())});
    auto span = tracer_->StartSpan(method);
    auto trace_context = std::make_shared<tracing::TraceContext>(span);
    auto context = std::make_shared<Context>(req_id);
    context->SetTraceContext(trace_context);

    if (method == "add")
        return handle_add(context, unpacker);
    else if (method == "create_index")
        return handle_create_index(context, unpacker);
    else if (method == "search_by_id")
        return handle_search_by_id(context, unpacker);
    else if (method == "compare_fragments_by_id")
        return handle_compare_fragments_by_id(context, unpacker);
    else if (method == "compare_fragments")
        return handle_compare_fragments(context, unpacker);
    else if (method == "drop_table")
        return handle_drop_table(context, unpacker);
    else if (method == "get_vectors")
        return handle_get_vectors(context, unpacker);
    else if (method == "clusterize")
        return handle_clusterize(context, unpacker);

    else
        throw std::runtime_error("Unknown method!");

}
std::vector<int64_t> decode_ids(Unpacker& unpacker){
    auto ids_decoder = unpacker.unpack<typed_array_decoder_t<uint64_t>>();
    //Faiss ids are int64_t, why faiss... why!!!??????
    std::size_t ids_size;
    auto pids = ids_decoder.dec(ids_size);
    std::vector<int64_t> ids (reinterpret_cast<int64_t*>(pids),
                              reinterpret_cast<int64_t*>(pids+ids_size));
    return ids;
}

void encode_ids(const std::vector<int64_t>& ids, Packer& packer){

    using arr_t = std::vector<uint64_t>;
    packer.pack<typed_array_encoder_t<arr_t>>(reinterpret_cast<const arr_t&>(ids));

}


json create_json_err_obj(const Status& status, const std::string& msg){

    json resp = {{"error",
                  {{"msg", str(boost::format("%s (%d): %s") % msg
                               % status.code() % status.message())},
                   {"code", status.code()}}}};
    return resp;
}

std::vector<std::uint8_t>
ZeroMQServer::
handle_add(const std::shared_ptr<Context>& pctx, Unpacker& unpacker){

    engine::VectorsData vectors;
    auto float_decoder = unpacker.unpack<typed_array_decoder_t<float>>();
    vectors.float_data_ = float_decoder.copy();


    vectors.id_array_ = decode_ids(unpacker);
    vectors.vector_count_ = vectors.id_array_.size();

    auto params = json::from_cbor(unpacker.buffer<char>(),
                                  unpacker.buffer<char>() + unpacker.size());

    // std::cout<<"PARAMS: "<<params<<std::endl;

    auto table_name = params.at("table_name").get<std::string>();

    Status status = create_table(pctx, table_name, params);
    if (not status.ok())
        return json::to_cbor(create_json_err_obj(status, "Failed to create table"));

    bool norm = params.value("normalize_L2", false);

    if (norm)
        faiss::fvec_renorm_L2(vectors.float_data_.size() / vectors.vector_count_,
                              vectors.vector_count_,
                              vectors.float_data_.data());

    status = request_handler_.Insert(pctx, table_name, vectors, "");
    if (not status.ok())
        return json::to_cbor(create_json_err_obj(status, "Failed to insert vectors"));

    bool index_vectors = params.value("create_index", false);
    if (index_vectors){
        status = create_index(pctx, table_name, params);
        if (not status.ok())
            return json::to_cbor(create_json_err_obj(status, "Failed to create index"));
    }



    json resp = {{"result", {{"normalize_L2", norm},
                             {"inserted_vecs", vectors.vector_count_},
                             {"created_index", index_vectors}}}};
    return json::to_cbor(resp);
}

std::vector<uint8_t>
ZeroMQServer::
handle_create_index(const std::shared_ptr<Context>& pctx, Unpacker& unpacker){
    auto params = json::from_cbor(unpacker.buffer<char>(),
                                  unpacker.buffer<char>() + unpacker.size());
    auto table_name = params.at("table_name").get<std::string>();

    auto status = create_index(pctx, table_name, params);
    if (not status.ok())
        return json::to_cbor(create_json_err_obj(status, "Failed to create index"));

    json resp = {{"result", {{"created_index", true}}}};
    return json::to_cbor(resp);
}

std::vector<uint8_t>
ZeroMQServer::
handle_drop_table(const std::shared_ptr<Context>& pctx, Unpacker& unpacker){
    auto params = json::from_cbor(unpacker.buffer<char>(),
                                  unpacker.buffer<char>() + unpacker.size());
    auto table_name = params.at("table_name").get<std::string>();

    auto status = request_handler_.DropTable(pctx, table_name);
    if (not status.ok())
        return json::to_cbor(create_json_err_obj(status, "Failed to drop table"));

    json resp = {{"result", {{"drop_table", true}}}};
    return json::to_cbor(resp);
}

std::vector<uint8_t>
ZeroMQServer::
handle_search_by_id(const std::shared_ptr<Context>& pctx, Unpacker& unpacker){

    engine::VectorsData vectors;
    vectors.id_array_ = decode_ids(unpacker);
    vectors.vector_count_ = vectors.id_array_.size();

    auto params = json::from_cbor(unpacker.buffer<char>(),
                                  unpacker.buffer<char>() + unpacker.size());
    // std::cout<<"PARAMS: "<<params<<std::endl;

    auto table_names = params.at("table_names").get<std::vector<std::string>>();
    vectors.query_table_ids = params.value("query_id_table_names", vectors.query_table_ids);

    std::vector<Range> ranges;
    std::vector<std::string> partitions;
    std::vector<std::string> file_ids;
    TopKQueryResult result;

    int64_t topk = params.value("topk", 10);

    auto status = request_handler_.Search(pctx, table_names, vectors, ranges,
                                          topk,
                                          params.value("nprobe", 16),
                                          partitions, file_ids, result);
    if (not status.ok())
        return json::to_cbor(create_json_err_obj(status, "Failed to search by ids"));

    if (result.row_num_ != vectors.vector_count_ or
        result.id_list_.size() != vectors.vector_count_ * topk or
        result.distance_list_.size() != vectors.vector_count_ * topk)
        return json::to_cbor(create_json_err_obj(status, "Invalid search results"));

    uint64_t est_size = 500 +
        result.id_list_.size() * sizeof(engine::ResultIds::value_type) + 50 +
        result.distance_list_.size() * sizeof(engine::ResultDistances::value_type) + 50;

    json resp = {{"result", {{"found_cnt", result.row_num_}}}};
    auto out_buf = json::to_cbor(resp);

    Packer packer(std::move(out_buf), out_buf.size(), est_size);

    encode_ids(result.id_list_, packer);

    packer.pack<typed_array_encoder_t<engine::ResultDistances>>(result.distance_list_);
    return packer.move_buffer();
}

std::vector<uint8_t>
ZeroMQServer::
handle_compare_fragments(const std::shared_ptr<Context>& pctx, Unpacker& unpacker){
    engine::CompareFragmentsReq req;
    auto float_decoder = unpacker.unpack<typed_array_decoder_t<float>>();
    req.float_data = float_decoder.copy();

    req.ids = decode_ids(unpacker);

    return compare_fragments_impl(pctx, std::move(req), unpacker);
}
std::vector<uint8_t>
ZeroMQServer::
handle_compare_fragments_by_id(const std::shared_ptr<Context>& pctx, Unpacker& unpacker){
    engine::CompareFragmentsReq req;
    return compare_fragments_impl(pctx, std::move(req), unpacker);
}

std::vector<uint8_t>
ZeroMQServer::
compare_fragments_impl(const std::shared_ptr<Context>& pctx,
                       engine::CompareFragmentsReq&& req,
                       Unpacker& unpacker){

    auto params = json::from_cbor(unpacker.buffer<char>(),
                                  unpacker.buffer<char>() + unpacker.size());

    if (params.contains("query_id_table_name"))
        req.query_table = params.at("query_id_table_name").get<std::string>();

    req.gpu_id = params.value("gpu_id", req.gpu_id);
    req.min_sim = params.value("min_sim", req.min_sim);
    req.topk = params.value("topk", req.topk);
    req.fragments = params.at("fragments");

    bool norm = params.value("normalize_L2", false);
    if (norm and not req.float_data.empty())
        faiss::fvec_renorm_L2(req.float_data.size() / req.ids.size(),
                              req.ids.size(),
                              req.float_data.data());

    json resp;
    auto status = request_handler_.CompareFragments(pctx, req, resp);

    if (not status.ok())
        return json::to_cbor(create_json_err_obj(status,
                                                 "Failed to compare vectors:" + status.ToString()));

    auto out_buf = json::to_cbor(resp);
    return out_buf;
}

std::vector<uint8_t>
ZeroMQServer::
handle_get_vectors(const std::shared_ptr<Context>& pctx, Unpacker& unpacker){

    engine::VectorsData vectors;
    vectors.id_array_ = decode_ids(unpacker);
    vectors.vector_count_ = vectors.id_array_.size();

    auto params = json::from_cbor(unpacker.buffer<char>(),
                                  unpacker.buffer<char>() + unpacker.size());

    auto table_names = params.at("table_names").get<std::vector<std::string>>();


    auto status = request_handler_.GetVectors(pctx, table_names, vectors);
    if (not status.ok())
        return json::to_cbor(create_json_err_obj(status, "Failed to get vectors"));


    uint64_t est_size = 500 +
        vectors.float_data_.size() * sizeof(float) + 50 +
        vectors.id_array_.size() * sizeof(engine::IDNumber) + 50;

    json resp = {{"result", {{"found_cnt", vectors.vector_count_}}}};
    auto out_buf = json::to_cbor(resp);

    Packer packer(std::move(out_buf), out_buf.size(), est_size);

    encode_ids(vectors.id_array_, packer);

    packer.pack<typed_array_encoder_t<std::vector<float>>>(vectors.float_data_);
    return packer.move_buffer();
}

std::vector<std::uint8_t>
ZeroMQServer::
handle_clusterize(const std::shared_ptr<Context>& pctx, Unpacker& unpacker){

    engine::VectorsData vectors;
    auto float_decoder = unpacker.unpack<typed_array_decoder_t<float>>();
    vectors.float_data_ = float_decoder.copy();


    vectors.id_array_ = decode_ids(unpacker);
    vectors.vector_count_ = vectors.id_array_.size();

    auto params = json::from_cbor(unpacker.buffer<char>(),
                                  unpacker.buffer<char>() + unpacker.size());

    vectors.query_table_ids = params.value("query_id_table_names", vectors.query_table_ids);
    engine::ClusterizeOptions opts;
    opts.table_id = params.at("table_name").get<std::string>();
    opts.use_gpu = params.value("use_gpu", opts.use_gpu);
    opts.niter = params.value("niter", opts.niter);
    opts.nredo = params.value("nredo", opts.nredo);
    opts.verbose = params.value("verbose", opts.verbose);
    opts.number_of_clusters = params.value("number_of_clusters", opts.number_of_clusters);

    if (params.value("drop_table_if_exist", true)){

        bool has_table = false;
        auto status = request_handler_.HasTable(pctx, opts.table_id, has_table);

        if (not status.ok())
            return json::to_cbor(create_json_err_obj(status, "Failed to check table"));

        if (has_table){
            status = request_handler_.DropTable(pctx, opts.table_id);
            if (not status.ok())
                return json::to_cbor(create_json_err_obj(status, "Failed to drop table"));
        }
    }

    auto status = create_table(pctx, opts.table_id, params);
    if (not status.ok())
        return json::to_cbor(create_json_err_obj(status, "Failed to create table"));

    bool norm = params.value("normalize_L2", false);

    if (norm)
        faiss::fvec_renorm_L2(vectors.float_data_.size() / vectors.vector_count_,
                              vectors.vector_count_,
                              vectors.float_data_.data());

    status = request_handler_.Clusterize(pctx, opts, vectors);
    if (not status.ok())
        return json::to_cbor(create_json_err_obj(status, "Failed to insert vectors"));

    bool index_vectors = params.value("create_index", false);
    if (index_vectors){
        status = create_index(pctx, opts.table_id, params);
        if (not status.ok())
            return json::to_cbor(create_json_err_obj(status, "Failed to create index"));
    }



    json resp = {{"result", {{"normalize_L2", norm},
                             {"created_index", index_vectors}}}};
    return json::to_cbor(resp);
}


Status
ZeroMQServer::
create_table(const std::shared_ptr<Context>& pctx,
             const std::string& table_name, const json& params){

    auto index_type = params.at("index_type").get<int64_t>();

    bool has_table = false;
    Status status =
        request_handler_.HasTable(pctx, table_name, has_table);

    if (not status.ok())
        return status;

    if (has_table)
        return status;

    auto dim = params.at("dim").get<int64_t>();
    auto index_file_size = params.at("index_build_threshold").get<int64_t>();

    int64_t metric_type = params.value("metric_type", 2);
    std::string enc_type = params.value("enc_type", "Flat");

    status = request_handler_.CreateTable(pctx, table_name, dim, index_file_size, metric_type,
                                          enc_type);

    if (not status.ok())
        return status;

    status = create_index(pctx, table_name, params);

    return status;

}

Status
ZeroMQServer::
create_index(const std::shared_ptr<Context>& pctx,
             const std::string& table_name, const json& params){

    auto index_type = params.at("index_type").get<int64_t>();
    auto nlist = params.value("ivf_nlist", 11000);
    std::string enc_type = params.value("enc_type", "Flat");

    auto status = request_handler_.CreateIndex(pctx, table_name, index_type, nlist, enc_type);
    return status;

}

}}}
