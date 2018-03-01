#include <cpu_only.h>
#include "blob_by_channel.h"

using std::pair;

using caffe::SyncedMemory;

namespace mytest {

void BlobByChannel::rankByL1()
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

    int kernelSize = height * width;
    int filterSize = channels * kernelSize;
    blkSize = kernelSize;

    v_asum.reserve(channels);
    v_asum.clear();
    float channelL1;
    for (int j = 0; j != channels; ++j) {
        channelL1 = 0;
        int destKernel = kernelSize * j;
        for (int i = 0; i != output; ++i) {
            channelL1 += _asum(blkSize, blobStart + filterSize * i + destKernel);
        }
        v_asum.push_back({channelL1, j});
    }
    // channel L1 范数排序
    sort(v_asum.begin(), v_asum.end());
    v_channels.reserve(output);
    v_channels.clear();
    // 保存 channel index
    for (const pair<float, int> &item : v_asum)
        v_channels.push_back(item.second);
    ranked_ = true;
}

void BlobByChannel::displayRank()
{
    cout << rankStr << " -- size: " << v_asum.size() << endl;
    for (size_t i = 0; i != v_asum.size(); ++i) {
        if ((i & 7) == 0)
            cout << endl;
        printf("%5d: %9f", v_asum.at(i).second, v_asum.at(i).first);
    }
    cout << endl;
}

int BlobByChannel::setZeroRate(float rate)
{
    int nZerod = channels * rate;
//    LOG(INFO) << "channels: " << channels;
//    LOG(INFO) << "nZerod: " << nZerod;
    if (nZerod == 0)
        return nZerod;
    zeroRate_ = rate;

    int kernelSize = height * width;
    int filterSize = channels * kernelSize;
    v_blockOffset_.clear();
    v_blockOffset_.reserve(nZerod);
    for (int i = 0; i != output; ++i) {
        int destFilter = filterSize * i;
        for (int j = 0; j != nZerod; ++j) {
            int _ch = v_channels.at(j);
            int destKernel = destFilter + kernelSize * _ch;
            v_blockOffset_.push_back(destKernel);
        }
    }
    sort(v_blockOffset_.begin(), v_blockOffset_.end());
    return nZerod;
}

} // end of namespace mytest
