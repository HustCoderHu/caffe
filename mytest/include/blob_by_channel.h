#ifndef BLOB_BY_CHANNEL_H
#define BLOB_BY_CHANNEL_H

#include <cpu_only.h>
#include <myblob.h>

namespace mytest {

class BlobByChannel : public MyBlob {
public:
    BlobByChannel(Blob<float> *blob) : MyBlob(blob) { rankStr = "Channel"; }
    ~BlobByChannel() = default;

    void rankByL1() override;
    void displayRank() override;
    int setZeroRate(float rate) override;
    void releaseRankBuf() override {
        v_asum.shrink_to_fit();
        v_channels.shrink_to_fit();
    }

    vector<pair<float, int> > v_asum; // channel 权重L1 和 序号
    vector<int> v_channels;
};


} // end of namespace mytest

#endif // BLOB_BY_CHANNEL_H
