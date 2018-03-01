#include <iostream>
#include <iosfwd>
#include <utility>

#include <cpu_only.h>
#include <mynet.h>
#include <mysolver.h>
#include <helper_functions.hpp>

#include <caffe/util/signal_handler.h>

using namespace mytest;

using std::string;
using std::vector;
//using std::set;
//using std::pair;
using std::map;
//using std::make_pair;
using std::cout;
using std::endl;
//using std::stringstream;
using std::istringstream;
using std::ostringstream;

using caffe::Caffe;
using caffe::Net;

static float str2f(const string& s)
{
    istringstream istream(s);
    float f;
    istream >> f;
    return f;
}

//int test_mynet(string& weightsFile, string& solverFile);

static void testTrain(int argc, char *argv[]);
void testResNet(int argc, char *argv[]);

/**
 * @brief main
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char *argv[])
{
//    Timer total_timer;
//    auto &cout = std::cout;
//    std::cout << argv[0] << "\n";
    testResNet(argc, argv);

    return 0;
}

static void testTrain(int argc, char *argv[])
{
    if (argc != 3) {
        cout << argc << "\n";
    }

    int param_idx = 1;
    string weightsFile(argv[param_idx++]);
    string modelFile(argv[param_idx++]);
//    string solverFile(argv[param_idx++]);
    float zeroRate = str2f(argv[param_idx++]);

    shared_ptr<Net<float>> _net(new Net<float>(modelFile, caffe::TRAIN));
    _net->CopyTrainedLayersFrom(weightsFile);

//    zeroResNet(modelFile, weightsFile);
//    filters2file(modelFile, weightsFile);

    shared_ptr<MyNet> sp_mynet(new MyNet(_net));
    vector<string> v_zerodLayers;
    resnet50_zerodLayers(v_zerodLayers);
    for (const string& layer : v_zerodLayers)
        sp_mynet->add_notrainLayers(layer);

    sp_mynet->rankByL1(KERNEL);
    sp_mynet->setZeroRate(zeroRate);
    sp_mynet->zeroByRate();
//    sp_mynet->rankByL1(MyBlobProto::FILTER);
//    sp_mynet->zeroByRate(0.1, MyBlobProto::FILTER);
}

void testResNet(int argc, char *argv[])
{
    int param_idx = 1;
    string weightsFile(argv[param_idx++]);
    string modelFile(argv[param_idx++]);
//    float zeroRate = str2f(argv[param_idx++]);

    shared_ptr<MyNet> sp_mynet(new MyNet(modelFile, weightsFile));

//    vector<string> v_zerodLayers;
//    resnet50_zerodLayers(v_zerodLayers);
//    for (const string& layer : v_zerodLayers)
//        sp_mynet->add_notrainLayers(layer);
//    LOG(INFO) << "after for add_notrainLayers";

//    sp_mynet->rankByL1(KERNEL);
//    sp_mynet->setZeroRate(zeroRate);
//    sp_mynet->zeroByRate();

    sp_mynet->test(50000, "sss.txt");
    LOG(INFO) << "before";
    sp_mynet = nullptr;
    LOG(INFO) << "main end";
}
