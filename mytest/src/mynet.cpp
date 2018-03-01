#include <cpu_only.h>

#include <myblob_factory.h>

#include <mynet.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <iosfwd>
#include <utility>
#include <omp.h>

namespace mytest {

static string float2str(float f)
{
    ostringstream outf;
    outf << f;
    return outf.str();
}

static void testNet(Net<float> *_net, uint32 iterations, const string &outputFile);


// general functions  =========================================================

static void testNet(Net<float> *_net, uint32 iterations, const string& outputFile)
{
    // Set device id and mode
#ifdef CPU_ONLY
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
#else
//    cudaDeviceProp device_prop;
//    cudaGetDeviceProperties(&device_prop, 0);
//    LOG(INFO) << "GPU device name: " << device_prop.name;
    LOG(INFO) << "Use GPU";
    Caffe::SetDevice(0);
    Caffe::set_mode(Caffe::GPU);
#endif

    vector<int> test_score_output_id;
    vector<float> test_score;
    float loss = 0;
    for (uint32 i = 0; i < iterations; ++i) {
        float iter_loss;
        const vector<Blob<float>*>& result =
                _net->Forward(&iter_loss);
        loss += iter_loss;
        int idx = 0;
        for (uint32 j = 0; j < result.size(); ++j) {
            const float* result_vec = result[j]->cpu_data();
            for (int k = 0; k < result[j]->count(); ++k, ++idx) {
                const float score = result_vec[k];
                if (i == 0) {
                    test_score.push_back(score);
                    test_score_output_id.push_back(j);
                } else {
                    test_score[idx] += score;
                }
                const std::string& output_name = _net->blob_names()[
                        _net->output_blob_indices()[j]];
                LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
            }
        }
    }
    loss /= iterations;
    LOG(INFO) << "Loss: " << loss;

    string toFile;
    for (uint32 i = 0; i < test_score.size(); ++i) {
        const std::string& output_name = _net->blob_names()[
                _net->output_blob_indices()[test_score_output_id[i]]];
        const float loss_weight = _net->blob_loss_weights()[
                _net->output_blob_indices()[test_score_output_id[i]]];
        std::ostringstream loss_msg_stream;
        const float mean_score = test_score[i] / iterations;
        if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight
                            << " = " << loss_weight * mean_score << " loss)";
        }
        toFile = toFile + output_name + " = " + float2str(mean_score)
                + loss_msg_stream.str() + "\n";
    }
    int fd = open(outputFile.c_str(), O_WRONLY | O_CREAT | O_TRUNC,
                       0666); // -rwxrwxrwx
    if (-1 == fd) {
        LOG(INFO) << "open " << outputFile << "error !!!";
        return ;
    }
    LOG(INFO) << toFile;
    LOG(INFO) << "write to file: " << outputFile;
    LOG(INFO) << "bytes: " << write(fd, toFile.c_str(), toFile.size());
//    LOG(INFO) << "write to file: outputFile, bytes: " << write(fd, toFile.c_str(), toFile.size());
    close(fd);
}

// class functions  =========================================================

MyNet::MyNet(const string& modelFile) : origModel(modelFile)
{
    sp_net_ = nullptr;
    // 读模型结构
    modelParam_.reset(new NetParameter);
    ReadNetParamsFromTextFileOrDie(origModel, modelParam_.get());
}

MyNet::MyNet(const string& modelFile, const string& weightsFile)
    : origModel(modelFile), origWeights(weightsFile)
{
    sp_net_ = nullptr;
    // 读模型结构
    modelParam_.reset(new NetParameter);
    ReadNetParamsFromTextFileOrDie(origModel, modelParam_.get());
    // 读参数到内存 后面要使用的话 直接复制一份 避免反复读文件
    weightsParam_.reset(new NetParameter);
    ReadNetParamsFromBinaryFileOrDie(origWeights, weightsParam_.get());

    modelParam_->mutable_state()->set_phase(TEST);
    modelParam_->mutable_state()->set_level(0);

    // contruct the layer map bottom -> top
    constructMap();
//    NetParameter &param = *modelParam_;
    //    LOG(INFO) << "modelParam_->layer_size(): " << modelParam_->layer_size();
}

MyNet::MyNet(shared_ptr<Net<float> > net)
    :sp_net_(net), modelParam_(nullptr), weightsParam_(nullptr)
{

}

void MyNet::add_notrainLayers(const string &layerName)
{
    auto iter = m_mylayer.find(layerName);
    if (iter != m_mylayer.end()) {
        LOG(INFO) << iter->first << " already added ";
//        LOG(INFO) << layerName << " already added ";
        return ;
    }
//    LOG(INFO) << layerName << " before add ";
    Net<float>* _net = sp_net_.get();
    const shared_ptr<Layer<float> > sp_layer =
            _net->layer_by_name(layerName);
    if (sp_layer != nullptr) {
        shared_ptr<MyLayer> sp_mylayer(new MyLayer(sp_layer.get()));
        m_mylayer.insert({layerName, sp_mylayer});
        LOG(INFO) << layerName << " add it ";
    } else {
        LOG(WARNING) << layerName << " not exist";
    }
}

/**
 * @brief myNet::mapBlob
 * layer -> blob 的映射
 * layer: 需要修改的那些 (通过 add_testLayers 加入的)
 *
 */
void MyNet::mapBlob()
{
    if (nullptr == sp_net_) {
        LOG(FATAL) << "!!! reloadModel before " << __FUNCTION__;
        return ;
    }
    Net<float> &_net = *sp_net_;
    for (auto item : m_mylayer) {
        string layerName = item.first;
        MyLayer *_mylayer = item.second.get();
        Layer<float> *layerPtr = _net.layer_by_name(layerName).get();
        _mylayer->mapBlob(layerPtr);
        LOG(INFO) << layerName << " mapBlob";
    }
    return;
}

void MyNet::set_use_global_stats(bool stat)
{
    NetParameter &param = *modelParam_;
    for (int i = 0; i != param.layer_size(); ++i) {
        LayerParameter *layerPtr = param.mutable_layer(i);
        if (layerPtr->type() == "BatchNorm") {
//            LOG(INFO) << "set_use_global_stats: " << layerPtr->name()
//                      << " " << stat;
            if (layerPtr->has_batch_norm_param()) {
                layerPtr->mutable_batch_norm_param()->set_use_global_stats(stat);
            }
        }
    }
}

void MyNet::rankByL1(RANK_STRATEGE rank)
{
    rank_ = rank;
//#pragma omp parallel for
    for (auto item : m_mylayer) {
        string layerName = item.first;
        MyLayer *_mylayer = item.second.get();
        LOG(INFO) << layerName << " rankByL1: " << rank;
        _mylayer->setRanked(rank);
//        LOG(INFO) << "  before rankByL1" ;
        _mylayer->rankByL1();
        LOG(INFO) << "rank finished";
        //    MyBlobProto::rankString[stg];
    }
    learnable_params_.clear();
    for (auto item : m_mylayer) {
        MyLayer *_mylayer = item.second.get();
        vector<shared_ptr<MyBlob> > &v_spBlobs = _mylayer->v_spBlobs;
        for (shared_ptr<MyBlob> _blob : v_spBlobs)
            learnable_params_.push_back(_blob);
    }
    LOG(INFO) << "learnable_params_.size(): " << learnable_params_.size();
    return ;

}

void MyNet::displayRank()
{
    for (auto item : m_mylayer) {
        MyLayer *_mylayer = item.second.get();
        _mylayer->displayRank();
    }
}

int MyNet::setZeroRate(float rate)
{
    for (shared_ptr<MyBlob> _blob : learnable_params_) {
//        LOG(INFO);
//        MyBlob *__blob = _blob.get();
        _blob->setZeroRate(rate);
//        nZerod = __blob->zeroByRate(rate, rank);
//        nZerod = _blob->zeroByRate(rate, rank);
    }
    return 0;
}

/**
 * @brief myNet::zeroByRate
 * @param rate
 * 对 Net 中较小权重置零
 */
void MyNet::zeroByRate()
{
    if (nullptr == sp_net_.get()) {
        LOG(FATAL) << "mapBlob not executed !!!";
        return ;
//        net_.reset(new Net<float>(*modelParam_));
//        net_->CopyTrainedLayersFrom(*weightsParam_);
//        reloadModel();
//        reloadWeights();
    }
    size_t paramSize = learnable_params_.size();
    #pragma omp parallel for num_threads(2)
    for (size_t i = 0; i < paramSize; ++i) {
        learnable_params_[i]->zeroByRate();
//        MyBlob *__blob = _blob.get();
//        _blob->zeroKernel();
//        nZerod = __blob->zeroByRate(rate, rank);
//        nZerod = _blob->zeroByRate(rate, rank);
    }
    return ;

    for (shared_ptr<MyBlob> _blob : learnable_params_) {
//        MyBlob *__blob = _blob.get();
        _blob->zeroByRate();
//        _blob->zeroKernel();
//        nZerod = __blob->zeroByRate(rate, rank);
//        nZerod = _blob->zeroByRate(rate, rank);
    }
    return ;

//    LOG(INFO) << "before zeroByRate";
    for (auto item : m_mylayer) {
//        string layerName = item.first;
        MyLayer *_mylayer = item.second.get();
//        LOG(INFO) << layerName;
        _mylayer->zeroByRate();
//        LOG(INFO) << layerName << " zerod: " << rate;
        //    MyBlobProto::rankString[stg];
    }
//    LOG(INFO) << "net zerod: " << rate;
}

void MyNet::test(uint32 testSetSize, const string& outputFile)
{
//    shared_ptr<NetParameter> net_param(new NetParameter);
//    sp_net_->ToProto(net_param.get());

//    uint32 batch_size = getTestBatchSize(*net_param);
//    uint32 batch_size = 50;
//    setTestBatchSize(*modelParam_, batch_size);
    if (origModel.empty()) {
        LOG(INFO) << "model not provided, unable to set batch size";
        return ;
    }
//    modelParam_.reset(new NetParameter);
//    ReadNetParamsFromTextFileOrDie(origModel, modelParam_.get());
    uint32 batch_size = getTestBatchSize(*modelParam_);
    if (0 == batch_size) {
        LOG(WARNING) << "batch_size 0";
        return ;
    }
    LOG(INFO) << "batch_size: " << batch_size;
    batch_size = 50;
    setTestBatchSize(*modelParam_, 50);
    batch_size = getTestBatchSize(*modelParam_);
    uint32 iterations = testSetSize / batch_size;
    LOG(INFO) << "batch_size: " << batch_size;
    LOG(INFO) << "iterations: " << iterations;

    reloadModel();
    reloadWeights();
    testNet(sp_net_.get(), iterations, outputFile);
}

void MyNet::toCaffeModel(const std::string &weightsFile) {
    NetParameter netParam;
    sp_net_->ToProto(&netParam);
    WriteProtoToBinaryFile(netParam, weightsFile);
    return ;
}

// private
void MyNet::constructMap()
{
    NetParameter *param = weightsParam_.get();
    //    NetParameter *param = modelParam.get();
    // scan the all layers
    for (int i = 0; i != param->layer_size(); ++i) {
        const LayerParameter& Lparam = param->layer(i);
        // construct map layer -> index
        mapLayerIdx.insert({Lparam.name(), i});
    }
    LOG(INFO) << __FUNCTION__ << " end";
}

uint32 MyNet::getTestBatchSize(Net<float> *_net)
{
    const vector<shared_ptr<Layer<float> > >& v_splayer = _net->layers();

    shared_ptr<Layer<float>> splayer = nullptr;
    for ( shared_ptr<Layer<float>> _splayer : v_splayer) {
        string layer_type = _splayer->type();
        if (layer_type == "Data") {
            splayer = _splayer;
            break;
        }
    }
    if (splayer == nullptr) {
        LOG(WARNING) << "data layer not found !!!";
        return 0;
    }
    const LayerParameter& layer_param = splayer->layer_param();
    return layer_param.data_param().batch_size();
}

uint32 MyNet::getTestBatchSize(NetParameter &param)
{
    uint32 batch_size = 0;
    int i = 0;
//    LOG(INFO) << "param.layer_size(): " << param.layer_size();
    for (; i != param.layer_size(); ++i) {
        const LayerParameter &layer = param.layer(i);
//        LOG(INFO) << layer.name() << ": " << layer.type()
//                  << ", " << layer.phase();
        if (layer.has_data_param()) {
            LOG(INFO) << layer.name() << ": " << layer.type()
                              << ", " << layer.phase();
            LOG(INFO) << "source: " << layer.data_param().source();
            batch_size = layer.data_param().batch_size();
            break;
        }
        if (layer.has_data_param() && layer.phase() == TEST) {
            // # data layer test
            batch_size = layer.data_param().batch_size();
            break;
        }
    }
    if (i == param.layer_size()) {
        LOG(WARNING) << "test phase not exist ?";
    }
//    LOG(INFO) << "batch_size: " << batch_size;
    return batch_size;
}

void MyNet::setTestBatchSize(NetParameter& param, uint32 batch_size)
{
//    NetParameter &param = *modelParam_;
//    LOG(INFO) << "param.layers_size(): " << param.layer_size();
    for (int i = 0; i != param.layer_size(); ++i) {
        LayerParameter *layerPtr = param.mutable_layer(i);
        if (layerPtr->has_data_param()) {
            LOG(INFO) << layerPtr->name() << ": " << layerPtr->type()
                              << ", " << layerPtr->phase();
            LOG(INFO) << "source: " << layerPtr->data_param().source();
            layerPtr->mutable_data_param()->set_batch_size(batch_size);
            break;
        }
    }
}

LayerParameter* MyNet::Lparam_by_name(const string& lName)
{
    NetParameter &param = *weightsParam_;
    auto iter = mapLayerIdx.find(lName);
    if (iter != mapLayerIdx.cend()) {
        return param.mutable_layer(iter->second);
    } else {
        LOG(WARNING) << "Unknown layer name " << lName;
        return nullptr;
    }
}

/**
 * @brief allFilters2file - write every filter3d of layerName to file
 * @param layerName
 * @param f
 */
void MyNet::allFilters2file()
{

}


} // end of namespace mytest

    // write to model
//    WriteProtoToTextFile(*modelParam_.get(), newModel);
//    LOG(INFO) << "written to " << newModel;


// 参考 https://github.com/shicai/Caffe_Manual
// http://blog.csdn.net/seven_first/article/category/5721883
// http://www.cnblogs.com/louyihang-loves-baiyan/p/5149628.html
// http://blog.csdn.net/u014114990/article/details/50315783
