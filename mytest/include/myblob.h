#ifndef BASE_BLOB_H
#define BASE_BLOB_H

#include <cpu_only.h>

namespace mytest {

class MyBlob {
public:
    MyBlob(Blob<float> *blob);
    virtual ~MyBlob() = default;

    virtual void rankByL1() { }
    virtual void displayRank() { }
    virtual int setZeroRate(float rate) { return 0; }
    virtual void releaseRankBuf() { }
    int zeroByRate();
    void zeroAll();

    string rankStr;
    bool ranked_;
    float zeroRate_;
    vector<int> v_blockOffset_;
    int blkSize;

    Blob<float> *blob_;
    int output;
    int channels;
    int height;
    int width;
    int count;
};

} // end of namespace mytest

#endif // BASE_BLOB_H
