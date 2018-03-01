#ifndef MYLAYER_H
#define MYLAYER_H

#include <cpu_only.h>
#include <myblob.h>

namespace mytest {

class MyLayer
{
public:
    MyLayer(LayerParameter *layerParam);
    MyLayer(Layer<float> *layerPtr);
    void mapBlob(Layer<float> *layerPtr);
    void clearBlob() {
        v_spBlobs.clear();
    }

    bool rankByL1();
    void displayRank();
    int setZeroRate(float rate);
    int zeroByRate();

//    float partAsum(float rate, RANK_STRATEGE rank) {
//        return blobProto_->asumOfRanked(rate, rank);
//    }

    void filters2vec(vector<float>& v_filtersAsum);

    void allFilters2file(const string& delim, const string& file);

    inline bool isRanked(RANK_STRATEGE rank) {
        return rank_bitmap & (1 << rank);
    }
    void setRanked(RANK_STRATEGE rank);

    string name_;
    Layer<float> *layer_ptr_;

    uint16_t rank_bitmap;

    RANK_STRATEGE rank_;
    //    LayerParameter *lParam;
//    shared_ptr<MyBlobProto> blobProto_;
    vector<shared_ptr<MyBlob> > v_spBlobs;
};

} // end of namespace mytest

#endif // MYLAYER_H
