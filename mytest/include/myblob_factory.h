#ifndef MYBLOBFACTORY_H
#define MYBLOBFACTORY_H

#include <cpu_only.h>

#include <myblob.h>
#include <blob_by_filter.h>
#include <blob_by_channel.h>
#include <blob_by_kernel.h>

namespace mytest {

class BlobFactory {
public:
    static MyBlob* getBlob(Blob<float> *blob, RANK_STRATEGE rank)
    {
        MyBlob* my_blob = nullptr;
        switch (rank) {
        case FILTER:
            my_blob = new BlobByFilter(blob);
            break;
        case CHANNEL:
            my_blob = new BlobByChannel(blob);
            break;
        case KERNEL:
            my_blob = new BlobByKernel(blob);
        default:
            break;
        }
        return my_blob;
    }
private:
    BlobFactory() = default;
};


} // end of namespace mytest

#endif // MYBLOBFACTORY_H
