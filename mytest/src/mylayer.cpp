#include <cpu_only.h>

#include <mylayer.h>
#include <myblob_factory.h>

//using caffe::BlobProto;

namespace mytest {

//MyLayer::MyLayer(LayerParameter *layerParam)
//{
//    if (nullptr == layerParam) {
//        LOG(FATAL) << layerParam << " nullptr !!!";
//        return ;
//    }
//    name_ = layerParam->name();
//    rank_bitmap = 0;
//    BlobProto *weightsProto = layerParam->mutable_blobs(0);
//    blobProto_.reset(new MyBlobProto(weightsProto));

////    for (int i = 0; i != layerParam->blobs_size(); ++i) {
////        BlobProto *weightsProto = layerParam->mutable_blobs(i);
//    //    }
//}

MyLayer::MyLayer(Layer<float> *layerPtr)
    : layer_ptr_(layerPtr), rank_(RANK_NOT_SET)
{
    name_ = layerPtr->layer_param().name();
    rank_bitmap = 0;
//    blobProto_ = nullptr;
}
/**
 * @brief MyLayer::mapBlob
 * @param layerPtr
 * 必须在 Net 类初始化完后才能执行
 */
void MyLayer::mapBlob(Layer<float> *layerPtr)
{
    shared_ptr<MyBlob> sp_myblob;
//    blobs_.clear();
    v_spBlobs.reserve(layerPtr->blobs().size());
    for (shared_ptr<Blob<float>> _blob : layerPtr->blobs()) {
        sp_myblob.reset(new MyBlob(_blob.get()));
//        sp_myblob.reset(new MyBlob(_blob.get(), blobProto_));
        v_spBlobs.push_back(sp_myblob);
    }
}

bool MyLayer::rankByL1()
{
    bool ret = true;

    shared_ptr<MyBlob> sp_myblob = v_spBlobs[0];
    sp_myblob->rankByL1();
//    rank_ = rank;

//    if (isRanked(rank)) {
//        LOG(INFO) << name_ << ": #" << rank << "# ranked already";
//        ret = false;
//    } else {
//        shared_ptr<MyBlob> sp_myblob = v_spBlobs[0];
//        sp_myblob->rankByL1();
//        setRanked(rank);
//    }
    return ret;
//    if (isRanked(rank))
//        LOG(INFO) << name_ << "#" << rank << "# ranked already";
//    else {
//        blobProto_->rankByL1(rank);
//        setRanked(rank);
    //    }
}

void MyLayer::displayRank()
{
    shared_ptr<MyBlob> sp_myblob = v_spBlobs[0];
    cout << "----  " << name_ << "  ----" << endl;
    sp_myblob->displayRank();
}

int MyLayer::setZeroRate(float rate)
{
    int nZerod = 0;
    for (shared_ptr<MyBlob> _blob : v_spBlobs) {
        MyBlob *__blob = _blob.get();
        nZerod = __blob->setZeroRate(rate);
//        nZerod = __blob->zeroByRate(rate, rank);
//        nZerod = _blob->zeroByRate(rate, rank);
    }
    return nZerod;
}

int MyLayer::zeroByRate()
{
    int nZerod = 0;
    for (shared_ptr<MyBlob> _blob : v_spBlobs) {
        MyBlob *__blob = _blob.get();
        nZerod = __blob->zeroByRate();
//        nZerod = __blob->zeroByRate(rate, rank);
//        nZerod = _blob->zeroByRate(rate, rank);
    }
    return nZerod;
}

void MyLayer::filters2vec(vector<float> &v_filtersAsum)
{
//    blobProto_->filtersAsum2vec(v_filtersAsum);
}

void MyLayer::allFilters2file(const std::string &delim, const std::string &file)
{
//    shared_ptr<Net<float> > _net(new Net<float>(*modelParam_.get()));
////    Net<float> *_net;
//    _net->CopyTrainedLayersFrom(origWeights);

//    int nLayer = _net->layers().size();
//    Layer<float> *layer_ptr;
//    string layerName;
//    string layerType;
//    Blob<float> *weights;
//    int output;
//    std::stringstream formatted;
//    float *x;
//    string outputFile;
//    int fd;

//    for (int i = 0; i != nLayer; ++i) {
//        layer_ptr = _net->layers()[i].get();
//        layerName = layer_ptr->layer_param().name();
//        layerType = layer_ptr->layer_param().type();
//        LOG(INFO) << layerName << ": " << layerType;
//        if (layer_ptr->layer_param().has_convolution_param()) {
//            LOG(INFO) << "write -> file ";
//        } else {
//            LOG(INFO) << "not has_convolution_param v continue";
//            continue;
//        }
////        if (layerType != "CONVOLUTION" && layerType != "Convolution") {
////            LOG(INFO) << "not CONVOLUTION v continue";
////            continue;
////        } else
////            LOG(INFO) << "write -> file ";

//        // save filters to ${layerName}.txt
//        weights = layer_ptr->blobs()[0].get();
//        output = weights->shape(0);
//        LOG(INFO) << "output: " << output;

//        // asum of each filter per line
////        formatted.clear();
//        char dest[16];
//        string out;
//        out.clear();
//        for (int i = 0; i != output; ++i) {
//            sprintf(dest, "%f", filter3d_asum(weights, i, x));
////            LOG(INFO) << "out " << out << "error !!!";
////            out += filter3d_asum(weights, i, x);
//            out += dest;
//            out += "\n";
////            formatted << filter3d_asum(weights, i, x) << "\n";
//        }
//        outputFile = layerName + ".txt";
//        fd = open(outputFile.c_str(), O_WRONLY | O_CREAT | O_TRUNC,
//                      0666); // -rwxrwxrwx
//        if (-1 == fd) {
//            LOG(INFO) << "open " << outputFile << "error !!!";
//            continue ;
//        }
//        LOG(INFO) << "out.size(): " << out.size();
//        LOG(INFO) << "write: " << write(fd, out.c_str(), out.size());
////        write(fd, formatted.str().c_str(), formatted.str().size());
//        close(fd);
//    }
    //    return ;
}

/**
 * @brief MyLayer::setRanked
 * @param rank
 * 根据排序类型生成不同的 Blob
 */
void MyLayer::setRanked(RANK_STRATEGE rank)
{
    shared_ptr<MyBlob> sp_myblob;
    MyBlob* myblob = nullptr;

    v_spBlobs.clear();
    v_spBlobs.reserve(layer_ptr_->blobs().size());
    for (shared_ptr<Blob<float>> _blob : layer_ptr_->blobs()) {
        myblob = BlobFactory::getBlob(_blob.get(), rank);
        sp_myblob.reset(myblob);
//        sp_myblob.reset(new MyBlob(_blob.get(), blobProto_));
        v_spBlobs.push_back(sp_myblob);
    }
    rank_ = rank;
}


} // end of namespace mytest
