#ifndef BLOB_BY_FILTER_H
#define BLOB_BY_FILTER_H

#include <cpu_only.h>
#include <myblob.h>

namespace mytest {

class BlobByFilter : public MyBlob {
public:
    BlobByFilter(Blob<float> *blob) : MyBlob(blob) { rankStr = "Filter"; }
    ~BlobByFilter() = default;

    void rankByL1() override;
    void displayRank() override;
    int setZeroRate(float rate) override;
    void releaseRankBuf() override {
        v_filters.shrink_to_fit();
    }

    vector<pair<float, int> > v_asum; // filter 权重L1 和 序号
    vector<int> v_filters;
};


} // end of namespace mytest

#endif // BLOB_BY_FILTER_H
