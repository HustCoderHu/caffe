#include <iostream>
#include <iosfwd>
#include <utility>

#include <cpu_only.h>
#include <mynet.h>
#include <mysolver.h>
#include <helper_functions.hpp>

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

static void dispW(int argc, char *argv[]);

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
    dispW(argc, argv);

    return 0;
}

static void dispW(int argc, char *argv[])
{
    if (argc != 3) {
        cout << argc << "\n";
    }

    int param_idx = 1;
    string weightsFile(argv[param_idx++]);
    string modelFile(argv[param_idx++]);
//    string solverFile(argv[param_idx++]);
    string layerName(argv[param_idx++]);

    shared_ptr<Net<float>> _net(new Net<float>(modelFile, caffe::TEST));
    _net->CopyTrainedLayersFrom(weightsFile);

    shared_ptr<MyNet> sp_mynet(new MyNet(_net));
//    vector<string> v_zerodLayers;
    sp_mynet->add_notrainLayers(layerName);
    LOG(INFO);
    sp_mynet->rankByL1(CHANNEL);
    sp_mynet->setZeroRate(0.9);
    sp_mynet->zeroByRate();
    sp_mynet->rankByL1(CHANNEL);
    LOG(INFO);
    sp_mynet->displayRank();
//    LOG(INFO);
}
