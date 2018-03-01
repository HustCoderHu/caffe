#include <iostream>

#include <cpu_only.h>
#include <mynet.h>
#include <mysolver.h>
#include <helper_functions.hpp>

using namespace mytest;
using std::string;
using std::vector;
using std::ostringstream;

#ifdef QT_HIGHLIGHT
using std::shared_ptr;
#endif

static float str2f(const string& s)
{
    istringstream istream(s);
    float f;
    istream >> f;
    return f;
}

static void retrain(int argc, char *argv[]);
static int _retrainZerod(const string &solverFile, const string &weightsFile,
                 float zeroRate);

int main(int argc, char *argv[])
{
    retrain(argc, argv);
    return 0;
}

void retrain(int argc, char *argv[])
{
    int param_idx = 1;
    string weightsFile(argv[param_idx++]);
    string solverFile(argv[param_idx++]);
    float zeroRate = str2f(argv[param_idx++]);
    _retrainZerod(solverFile, weightsFile, zeroRate);
}

int _retrainZerod(const string &solverFile, const string &weightsFile,
                 float zeroRate)
{
//    CHECK_GT(solverFile.size(), 0) << "Need a solver definition to train.";
//    CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
//            << "Give a snapshot to resume training or weights to finetune "
//               "but not both.";
//    vector<string> stages = get_stages_from_flags();

    caffe::SolverParameter solver_param;
    caffe::ReadSolverParamsFromTextFileOrDie(solverFile, &solver_param);

//    solver_param.mutable_train_state()->set_level(FLAGS_level);
//    for (int i = 0; i < stages.size(); i++) {
//        solver_param.mutable_train_state()->add_stage(stages[i]);
//    }

    // If the gpus flag is not provided, allow the mode and device to be set
    // in the solver prototxt.
//    if (FLAGS_gpu.size() == 0
//            && solver_param.has_solver_mode()
//            && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
//        if (solver_param.has_device_id()) {
//            FLAGS_gpu = "" +
//                    boost::lexical_cast<string>(solver_param.device_id());
//        } else {  // Set default GPU if unspecified
//            FLAGS_gpu = "" + boost::lexical_cast<string>(0);
//        }
//    }

    vector<int> gpus;
#ifndef CPU_ONLY
    gpus.push_back(0);
#endif
//    get_gpus(&gpus);
    if (gpus.size() == 0) {
        LOG(INFO) << "Use CPU.";
        Caffe::set_mode(Caffe::CPU);
    } else {
        ostringstream s;
        for (size_t i = 0; i < gpus.size(); ++i) {
            s << (i ? ", " : "") << gpus[i];
        }
        LOG(INFO) << "Using GPUs " << s.str();
#ifndef CPU_ONLY
        cudaDeviceProp device_prop;
        for (size_t i = 0; i < gpus.size(); ++i) {
            cudaGetDeviceProperties(&device_prop, gpus[i]);
            LOG(INFO) << "GPU " << gpus[i] << ": " << device_prop.name;
        }
#endif
        solver_param.set_device_id(gpus[0]);
        Caffe::SetDevice(gpus[0]);
        Caffe::set_mode(Caffe::GPU);
        Caffe::set_solver_count(gpus.size());
    }

    string FLAGS_sigint_effect("stop");
    string FLAGS_sighup_effect("snapshot");
    caffe::SignalHandler signal_handler(
                GetRequestedAction(FLAGS_sigint_effect),
                GetRequestedAction(FLAGS_sighup_effect));

    shared_ptr<caffe::Solver<float> >
            solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

    solver->SetActionFunction(signal_handler.GetActionFunction());

    if (weightsFile.find("solverstate") != string::npos) {
        LOG(INFO) << "Resuming from " << weightsFile;
        solver->Restore(weightsFile.c_str());
    } else if (solverFile.size()) {
        CopyLayers(solver.get(), weightsFile);
    }

    vector<string> v_zerodLayers;
    resnet50_zerodLayers(v_zerodLayers);
    shared_ptr<MySolver> sp_mysolver(new MySolver(solver, v_zerodLayers));
//    sp_mysolver->top5acc = 0.9;
    shared_ptr<MyNet> sp_mynet = sp_mysolver->sp_mynet_;
    LOG(INFO) << "zeroRate: " << zeroRate;
    sp_mysolver->zeroRate_ = zeroRate;
    sp_mynet->rankByL1(CHANNEL);
    LOG(INFO) << "before setZeroRate";
    sp_mynet->setZeroRate(zeroRate);
    LOG(INFO) << "before zeroByRate";
    sp_mynet->zeroByRate();
    LOG(INFO) << "net zerod: " << zeroRate;

    LOG(INFO) << "Starting Optimization";
    if (gpus.size() > 1) {
#ifdef USE_NCCL
        caffe::NCCL<float> nccl(solver);
        nccl.Run(gpus, FLAGS_snapshot.size() > 0 ? FLAGS_snapshot.c_str() : NULL);
#else
        LOG(FATAL) << "Multi-GPU execution not available - rebuild with USE_NCCL";
#endif
    } else {
        sp_mysolver->Solve();
        LOG(INFO) << __FUNCTION__;
//        solver->Solve();
    }
    LOG(INFO) << "Optimization Done.";
    return 0;
}
