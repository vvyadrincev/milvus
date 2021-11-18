#define BOOST_TEST_MODULE simple_test
#include <boost/test/unit_test.hpp>


namespace bunt = boost::unit_test;

struct Fixture{
    Fixture() {}
};


BOOST_FIXTURE_TEST_CASE(dummy_test, Fixture){
    BOOST_TEST(true);
}
