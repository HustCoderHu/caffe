#ifndef CPU_ONLY_H
#define CPU_ONLY_H

//#define QT_HIGHLIGHT
//#define CPU_ONLY

#include <string>
#include <vector>
#include <set>
#include <map>
#include <memory>
#include <sstream>
#include <algorithm>

#include <caffe/caffe.hpp>
#include <caffe/util/signal_handler.h>
//#define TRAIN_EXTREM_SPEED

namespace mytest {
enum RANK_STRATEGE {
    FILTER = 0,
    CHANNEL,
    KERNEL,
    RANK_NOT_SET
};
using uint32 = unsigned int;
using std::string;
using std::vector;
using std::set;
using std::map;
using std::pair;
using std::sort;
using std::stringstream;
using std::istringstream;
using std::ostringstream;
using std::cout;
using std::endl;

//using namespace caffe;  // NOLINT(build/namespaces)
using caffe::Caffe;
using caffe::TEST;
using caffe::Timer;
using caffe::Net;
using caffe::Layer;
using caffe::Blob;
using caffe::Solver;
using caffe::NetParameter;
using caffe::LayerParameter;
using caffe::BlobProto;
using caffe::BlobShape;
using caffe::SolverParameter;
using caffe::ReadNetParamsFromTextFileOrDie;
using caffe::ReadNetParamsFromBinaryFileOrDie;
using caffe::WriteProtoToTextFile; // (*modelParam_.get(), newModel);



#ifdef QT_HIGHLIGHT
using std::shared_ptr;
#else
using caffe::shared_ptr;
#endif

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

using my_memset = void(*)(const size_t, const int, void*);
using caffe::caffe_memset;
#ifndef CPU_ONLY
using caffe::caffe_gpu_memset;
#endif

} // end of namespace mytest

#endif // CPU_ONLY_H
