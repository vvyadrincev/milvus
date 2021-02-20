#ifndef ZEROMQ_IMPL_CBOR_UTILS_HPP
#define ZEROMQ_IMPL_CBOR_UTILS_HPP

#include <boost/endian/conversion.hpp>

#include <cbor.h>

#include <string>
#include <vector>
#include <cstring>
#include <type_traits>
#include <stdexcept>
#include <cmath>


namespace milvus {
namespace server {
namespace zeromq {


struct cbor_item_holder_t{
    cbor_item_holder_t(cbor_item_t* item):item(item){}
    ~cbor_item_holder_t(){cbor_decref(&item);}


    cbor_item_t* item;
};


class string_decoder_t{
public:
    string_decoder_t(const std::vector<uint8_t>& buffer):m_buffer(buffer.data()),
                                                         m_size(buffer.size()){}
    string_decoder_t(const uint8_t* buffer, std::size_t size):m_buffer(buffer),
                                                              m_size(size){}

    ~string_decoder_t(){
        if (m_item)
            cbor_decref(&m_item);
    }

    void init(){
        struct cbor_load_result result;
        m_item = cbor_load(m_buffer, m_size, &result);

        if (not m_item)
            throw std::runtime_error("CBOR parse error, code: " + std::to_string(result.error.code));

        m_read_size = result.read;

        if (not cbor_isa_string(m_item))
            throw std::runtime_error("String should have string type!");
    }

    std::size_t read_size()const{
        return m_read_size;
    }

    const char* dec(std::size_t& length){
        if (not m_item)
            init();

        length = cbor_string_length(m_item);

        return reinterpret_cast<const char*>(cbor_string_handle(m_item));
    }

    std::string copy(){
        std::size_t size;
        auto p = dec(size);
        return std::string(p, p+size);
    }

private:

    const uint8_t*     m_buffer;
    std::size_t        m_size;


    cbor_item_t* m_item      = nullptr;
    std::size_t  m_read_size = 0;

};



template <typename T>
class typed_array_decoder_t{
    static_assert(boost::endian::order::native == boost::endian::order::little);
public:
    typed_array_decoder_t(const std::vector<uint8_t>& buffer):m_buffer(buffer.data()),
                                                              m_size(buffer.size()){}
    typed_array_decoder_t(const uint8_t* buffer, std::size_t size):m_buffer(buffer),
                                                                   m_size(size){}

    ~typed_array_decoder_t(){
        if(m_bin_arr_item)
            cbor_decref(&m_bin_arr_item);
        if (m_tag_item)
            cbor_decref(&m_tag_item);
    }

    void init(){
        struct cbor_load_result result;
        m_tag_item = cbor_load(m_buffer, m_size, &result);

        if (not m_tag_item)
            throw std::runtime_error("CBOR parse error, code: " + std::to_string(result.error.code));

        m_read_size = result.read;

        if (not cbor_isa_tag(m_tag_item))
            throw std::runtime_error("Typed array should have type tag!");

        auto tag_value = cbor_tag_value(m_tag_item);
        if (not (tag_value > 63 && tag_value < 87))
            throw std::runtime_error("Unknown tag value " + std::to_string(tag_value));

        m_bin_arr_item = cbor_tag_item(m_tag_item);

        if (not cbor_isa_bytestring(m_bin_arr_item))
            throw std::runtime_error("Typed array should include bytestring!");

        assert(m_bin_arr_item->type == 2);

    }

    std::size_t read_size()const{
        return m_read_size;
    }

    T* dec(std::size_t& size){
        if (not m_bin_arr_item)
            init();

        auto binary_size = cbor_bytestring_length(m_bin_arr_item);

        auto tag_value = cbor_tag_value(m_tag_item);
        //see https://tools.ietf.org/html/draft-ietf-cbor-array-tags-00
        auto ll = tag_value & 3;
        auto f = tag_value >> 4 & 1;
        auto elem_sz = 1 << (f + ll);
        size = binary_size / elem_sz;

        auto little_endian = tag_value >> 2 & 1;
        if (not f && ll && not little_endian)
            //TODO swap bytes
            throw std::runtime_error("Big endian is not supported!");


        auto raw_data = cbor_bytestring_handle(m_bin_arr_item);
        switch(tag_value){
        case 64: check<std::uint8_t>(); break;
        case 69: check<std::uint16_t>(); break;
        case 70: check<std::uint32_t>(); break;
        case 71: check<std::uint64_t>(); break;
        case 72: check<std::int8_t>(); break;
        case 77: check<std::int16_t>(); break;
        case 78: check<std::int32_t>(); break;
        case 79: check<std::int64_t>(); break;
        case 85: check<float>(); break;
        case 86: check<double>(); break;
        default:
            std::runtime_error("Unknown tag_value: " + std::to_string(tag_value));
        }

        return reinterpret_cast<T*>(raw_data);

    }

    std::vector<T> copy(){
        std::size_t size;
        auto p = dec(size);
        std::vector<T> v(p, p+size);
        return v;
    }

private:

    template<class U>
    void check(){
        if (not std::is_same_v<T, U>)
            throw std::runtime_error("Different types");
    }

    const uint8_t*     m_buffer;
    std::size_t        m_size;


    cbor_item_t* m_tag_item     = nullptr;
    cbor_item_t* m_bin_arr_item = nullptr;
    std::size_t  m_read_size    = 0;

};

class Unpacker{
public:
    Unpacker(const std::vector<uint8_t>& buffer):m_buffer(buffer.data()),
                                                 m_size(buffer.size()){}

    Unpacker(const uint8_t* buffer, std::size_t size):m_buffer(buffer),
                                                      m_size(size){}


    template <class Decoder>
    Decoder unpack(){
        Decoder decoder(m_buffer + m_offset, m_size - m_offset);
        decoder.init();
        m_offset += decoder.read_size();
        return decoder;

    }

    template <class T = uint8_t>
    const T* buffer()const{
        return reinterpret_cast<const T*>(m_buffer) + m_offset;

    }

    std::size_t offset()const{
        return m_offset;
    }

    std::size_t size()const{
        return m_size - m_offset;
    }

private:


    const uint8_t*     m_buffer;
    std::size_t        m_size;
    std::size_t        m_offset = 0;

};




template <typename Cont>
class typed_array_encoder_t{
    // static_assert(boost::endian::order::native == boost::endian::order::little);
public:
    using container_t = Cont;
    using value_t = typename Cont::value_type;

    typed_array_encoder_t(const Cont& cont):m_cont(cont) {}

    ~typed_array_encoder_t(){
        //dont use cbor_decref
        //because we own binary data in m_cont
        _CBOR_FREE(m_bin_arr_item);
        _CBOR_FREE(m_tag_item);
    }

    cbor_item_t* enc(){
        m_bin_arr_item = cbor_new_definite_bytestring();
        cbor_bytestring_set_handle(m_bin_arr_item,
                                   reinterpret_cast<uint8_t*>(const_cast<value_t*>(m_cont.data())),
                                   m_cont.size() * sizeof(value_t));

        // std::cout<<"tag value: "<<tag_value()<<std::endl;
        m_tag_item = cbor_new_tag(tag_value());
        cbor_tag_set_item(m_tag_item, m_bin_arr_item);

        return m_tag_item;
    }

    uint64_t estimate_size()const{
        return sizeof(value_t) * m_cont.size() + 32;
    }
private:

    uint64_t tag_value()const{
        //see https://tools.ietf.org/html/draft-ietf-cbor-array-tags-00
        int f = std::is_floating_point_v<value_t> ? 1 : 0;
        int ll = std::log2(sizeof(value_t)) - f;
        int s = std::is_signed_v<value_t> && std::is_integral_v<value_t> ? 1 : 0 ;
        int e = boost::endian::order::native == boost::endian::order::little ? 1 : 0;

        int tag_value = 1;
        tag_value <<= 6;
        tag_value |= f << 4;
        tag_value |= s << 3;
        tag_value |= e << 2;
        tag_value |= ll;

        return tag_value;
    }

    const Cont& m_cont;


    cbor_item_t* m_tag_item     = nullptr;
    cbor_item_t* m_bin_arr_item = nullptr;

};

class Packer{
public:
    Packer(uint64_t estimated_buf_size):m_buffer(estimated_buf_size),
                                        m_written(0)
    {}
    Packer(std::vector<uint8_t>&& buf, uint64_t offset,
           uint64_t estimated_buf_size):m_buffer(std::move(buf)),
                                        m_written(offset)
    {
        m_buffer.resize(estimated_buf_size);
    }


    template <class Encoder, class T>
    void pack(const T& v){
        Encoder encoder(v);

        auto item = encoder.enc();

        if (encoder.estimate_size() > m_buffer.size() - m_written)
            //TODO resize
            throw std::runtime_error("Buffer is too small!");

        auto written = cbor_serialize(item, m_buffer.data() + m_written,
                                      m_buffer.size() - m_written);
        m_written += written;

    }

    template <class T>
    const T* buffer()const{
        return reinterpret_cast<const T*>(m_buffer.data());
    }

    uint64_t size()const{
        return m_written;
    }

    std::vector<uint8_t>&& move_buffer(){
        m_buffer.resize(m_written);
        return std::move(m_buffer);
    }


private:


    std::vector<uint8_t> m_buffer;
    uint64_t             m_written;

};








}}}



#endif // ZEROMQ_IMPL_CBOR_UTILS_HPP
