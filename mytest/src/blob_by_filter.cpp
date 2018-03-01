#include <cpu_only.h>
#include "blob_by_filter.h"

using std::pair;

using caffe::SyncedMemory;

namespace mytest {

void BlobByFilter::rankByL1()
{
    if (ranked_) {
        LOG(INFO) << "ranked already";
        LOG(INFO) << "rank again";
//        return ;
    }
    my_asum _asum;
    float *blobStart;

    switch (blob_->data()->head()) {
    case SyncedMemory::HEAD_AT_CPU:
        // perform computation on CPU
        _asum = caffe_cpu_asum;
        blobStart = blob_->mutable_cpu_data();
        break;
    case SyncedMemory::HEAD_AT_GPU:
    case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
        // perform computation on GPU
        _asum = my_gpu_asum;
        blobStart = blob_->mutable_gpu_data();
#else
        NO_GPU;
#endif
        break;
    default:
        LOG(FATAL) << "Syncedmem not initialized.";
    }

    int filterSize = channels * height * width;


    v_asum.reserve(output);
    v_asum.clear();
    blkSize = filterSize;
    for (int i = 0; i != output; ++i) {
        int destFilter = filterSize * i;
        float filterL1 = _asum(blkSize, blobStart + destFilter);
        v_asum.push_back({filterL1, i});
    }
    // filter L1 范数排序
    sort(v_asum.begin(), v_asum.end());
    v_filters.clear();
    v_filters.reserve(output);
    // 保存 filter index
    for (const pair<float, int> &item : v_asum)
        v_filters.push_back(item.second);
    ranked_ = true;
}

void BlobByFilter::displayRank()
{
    cout << rankStr << " -- size: " << v_asum.size() << endl;
    for (size_t i = 0; i != v_asum.size(); ++i) {
        if ((i & 7) == 0)
            cout << endl;
        printf("%5d: %f", v_asum.at(i).second, v_asum.at(i).first);
    }
    cout << endl;
}

int BlobByFilter::setZeroRate(float rate)
{
    int nZerod = output * rate;
    if (nZerod == 0)
        return nZerod;
    zeroRate_ = rate;

    int filterSize = channels * height * width;
    v_blockOffset_.clear();
    v_blockOffset_.reserve(nZerod);
    for (int i = 0; i != nZerod; ++i) {
//        LOG(INFO) << "i: " << i;
        int idx = v_filters.at(i);
        int destFilter = filterSize * idx;
        v_blockOffset_.push_back(destFilter);
    }
    sort(v_blockOffset_.begin(), v_blockOffset_.end());

    return nZerod;
}


} // end of namespace mytest
