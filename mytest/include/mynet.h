#ifndef MYNET_N
#define MYNET_N

//#include <myblobproto.h>
//#include <myblob.h>

#include <cpu_only.h>

#include <mylayer.h>

namespace mytest {

class MyNet {
public:
    enum BlobType {
        WEIGHTS = 0,
        BIAS
    };
    MyNet(const string& modelFile);
    MyNet(const string& modelFile, const string& weightsFile);
    MyNet(shared_ptr<Net<float> > net);
    ~MyNet() = default;

    inline shared_ptr<NetParameter> modelParam() {
        return modelParam_;
    }
    inline shared_ptr<NetParameter> weightsParam() {
        return weightsParam_;
    }
    inline void reloadModel() {
        sp_net_.reset(new Net<float>(*modelParam_));
    }
    inline void reloadWeights() {
        sp_net_->CopyTrainedLayersFrom(*weightsParam_);
    }

    void add_notrainLayers(const string& layerName);
    inline void clear_testLayers() {
        m_mylayer.clear();
//        m_blob.clear();
//        m_blobRank.clear();
    }
    void mapBlob();

    void setTestBatchSize(caffe::NetParameter &param, uint32 batch_size);
    void set_use_global_stats(bool stat);
//    void setZeroStratege(enum RANK_STRATEGE stg) { stratege = stg; }

    void rankByL1(RANK_STRATEGE rank);
    void displayRank();
//    void rankFiltersByL1();
//    void rankChannelsByL1();
//    void rankKernelsByL1();

    int setZeroRate(float rate);
    void zeroByRate();

    void test(uint32 testSetSize, const string &outputFile);

    void toCaffeModel(const string& weightsFile);

//private:
    void constructMap();
    uint32 getTestBatchSize(Net<float> *_net);
    static uint32 getTestBatchSize(NetParameter &param);

    LayerParameter* Lparam_by_name(const string& lName);

    void allFilters2file();

    string origModel;
    string origWeights;

    shared_ptr<Net<float> > sp_net_;
    shared_ptr<NetParameter> modelParam_;
    shared_ptr<NetParameter> weightsParam_;
    map<string, int> mapLayerIdx;

    RANK_STRATEGE rank_;
    map<string, shared_ptr<MyLayer> > m_mylayer;
    vector<shared_ptr<MyBlob> > learnable_params_;
}; // end of class myNet

} // end of namespace mytest

#endif // MYNET_N

