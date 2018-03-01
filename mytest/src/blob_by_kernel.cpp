#include <cpu_only.h>
#include "blob_by_kernel.h"

using std::pair;

using caffe::SyncedMemory;

using my_asum = float(*)(const int, const float*);
using caffe::caffe_cpu_asum;
#ifndef CPU_ONLY
using caffe::caffe_gpu_asum;
static inline float my_gpu_asum(const int n, const float* x)
{
    float y;
    caffe_gpu_asum(n, x, &y);
    return y;
}
#endif

namespace mytest {

void BlobByKernel::rankByL1()
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
    shared_ptr<vector<int> > sp_kernelsRank;
    vector<int> *_kernelsRank;
    // 分配空间
    v_kernelsRank.clear();
    v_kernelsRank.reserve(output);
    for (int i = 0; i != output; ++i) {
        _kernelsRank = new vector<int>;
        _kernelsRank->reserve(channels);
        sp_kernelsRank.reset(_kernelsRank);
        v_kernelsRank.push_back(sp_kernelsRank);
    }

    vector<pair<float, int> > v_asum; // kernel 权重L1 和 序号
    v_asum.reserve(channels);
    blkSize = kernelSize;
    for (int i = 0; i != output; ++i) {
        int destFilter = filterSize * i;
        v_asum.clear();
        for (int j = 0; j != channels; ++j) {
            int destKernel = destFilter + kernelSize * j;
            float kernelL1 = _asum(blkSize, blobStart + destKernel);
            v_asum.push_back({kernelL1, j});
        }
        // kernel L1范数 排序
        sort(v_asum.begin(), v_asum.end());
        _kernelsRank = v_kernelsRank[i].get();
        // 保存 kernel index
        for (const pair<float, int> &item : v_asum)
            _kernelsRank->push_back(item.second);
    }
    ranked_ = true;
    //    LOG(INFO) << __FUNCTION__ << " finish";
}

int BlobByKernel::setZeroRate(float rate)
{
    int nZerod = channels * rate;
    if (nZerod == 0)
        return nZerod;
    zeroRate_ = rate;

    int kernelSize = height * width;
    int filterSize = channels * kernelSize;
    v_blockOffset_.clear();
    v_blockOffset_.reserve(output * nZerod);
//    float *blobStart = blob_->mutable_cpu_data();
    for (int i = 0; i < output; ++i) {
        int destFilter = filterSize * i;
        const vector<int> *_kernelsRank = v_kernelsRank.at(i).get();
        for (int j = 0; j < nZerod; ++j) {
//            LOG(INFO);
            int idx = _kernelsRank->at(j);
//            LOG(INFO);
            int destKernel = destFilter + kernelSize * idx;
            v_blockOffset_.push_back(destKernel);
        }
    }
    sort(v_blockOffset_.begin(), v_blockOffset_.end());
//    LOG(INFO) << "v_blocks_.size(): " << v_blockIdx_.size();
    return nZerod;
}


} // end of namespace mytest
