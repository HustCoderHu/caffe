#ifndef BLOBBYKERNEL_H
#define BLOBBYKERNEL_H

#include <cpu_only.h>
#include <myblob.h>

namespace mytest {

class BlobByKernel : public MyBlob {
public:
    BlobByKernel(Blob<float> *blob) : MyBlob(blob) { rankStr = "Kernel"; }
    ~BlobByKernel() = default;

    void rankByL1() override;
    int setZeroRate(float rate) override;
    void releaseRankBuf() override {
        v_kernelsRank.clear();
        v_kernelsRank.shrink_to_fit();
    }

    vector<shared_ptr<vector<int> > > v_kernelsRank;
};

} // end of namespace mytest

#endif // BLOBBYKERNEL_H
