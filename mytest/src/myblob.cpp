#include <cpu_only.h>
#include <myblob.h>

using std::vector;

namespace mytest {

MyBlob::MyBlob(Blob<float> *blob)
    : ranked_(false), blob_(blob)
{
    const vector<int>& shape = blob_->shape();

    output = 1;
    channels = 1;
    height = 1;
    width = 1;

    int shapeSize = shape.size();

    switch (shapeSize) {
    case 4: width = shape.at(3);
    case 3: height = shape.at(2);
    case 2: channels = shape.at(1);
    case 1: output = shape.at(0);
        break;
    default:
        LOG(INFO) << "shapeSize: " << shapeSize;
        break;
    }
    count = output * channels * height * width;
}

int MyBlob::zeroByRate()
{
    my_memset _memset = caffe_memset;
    float *blobStart = blob_->mutable_cpu_data();

    for (int _offset : v_blockOffset_) {
        _memset(blkSize * sizeof(float), float(0),
                blobStart + _offset);
    }
    return 0;
}

void MyBlob::zeroAll()
{
    my_memset _memset = caffe_memset;
    float *blobStart = blob_->mutable_cpu_data();

    _memset(count * sizeof(float), float(0), blobStart);
}


} // end of namespace mytest
