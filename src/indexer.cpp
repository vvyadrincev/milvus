#include <faiss/IndexFlat.h>

int main(){

    faiss::IndexFlat f(300, faiss::METRIC_INNER_PRODUCT);
    std::vector<float> data(300);
    f.add(1, data.data());
    return 0;
}
