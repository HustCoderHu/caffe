#include <cpu_only.h>
#include <mysolver.h>

//using caffe::Solver<float>::Callback;
//using caffe::SolverAction;
// error: namespace ‘caffe::SolverAction’ not allowed in using-declaration

namespace mytest {

MySolver::MySolver(shared_ptr<Solver<float> > solver,
                   const vector<std::string> &v_zerodLayers)
    :solver_(solver)
{
    sp_mynet_.reset(new MyNet(solver->net()));
    for (const string& layer : v_zerodLayers)
        sp_mynet_->add_notrainLayers(layer);
//    sp_mynet_->rankByL1(KERNEL);
}

void MySolver::Solve(const char* resume_file)
{
    shared_ptr<Net<float> > net_ = solver_->net();
    SolverParameter& param_ = solver_->param_;
    int& iter_ = solver_->iter_;
    bool& requested_early_exit_ = solver_->requested_early_exit_;

    CHECK(Caffe::root_solver());
    LOG(INFO) << "Solving " << net_->name();
    LOG(INFO) << "Learning Rate Policy: " << param_.lr_policy();

    // Initialize to false every time we start solving.
    requested_early_exit_ = false;
    if (resume_file) {
        LOG(INFO) << "Restoring previous solver status from " << resume_file;
        solver_->Restore(resume_file);
    }
    // For a network that is trained by the solver, no bottom or top vecs
    // should be given, and we will just provide dummy vecs.
    int start_iter = iter_;
    Step(param_.max_iter() - iter_);
    // If we haven't already, save a snapshot after optimization, unless
    // overridden by setting snapshot_after_train := false
    param_.set_snapshot_after_train(false);
    if (param_.snapshot_after_train()
            && (!param_.snapshot() || iter_ % param_.snapshot() != 0)) {
        solver_->Snapshot();
    }
    if (requested_early_exit_) {
        LOG(INFO) << "Optimization stopped early.";
        return;
    }
    // After the optimization is done, run an additional train and test pass to
    // display the train and test loss/outputs if appropriate (based on the
    // display and test_interval settings, respectively).  Unlike in the rest of
    // training, for the train net we only run a forward pass as we've already
    // updated the parameters "max_iter" times -- this final pass is only done to
    // display the loss, which is computed in the forward pass.
    if (param_.display() && iter_ % param_.display() == 0) {
        int average_loss = param_.average_loss();
        float loss;
        net_->Forward(&loss);

        solver_->UpdateSmoothedLoss(loss, start_iter, average_loss);

        LOG(INFO) << "Iteration " << iter_ << ", loss = "
                  << solver_->smoothed_loss_;
    }
    if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
        solver_->TestAll();
    }
    LOG(INFO) << "Optimization Done.";
}

void MySolver::Step(int iters)
{
    SolverParameter& param_ = solver_->param_;
    float& smoothed_loss_ = solver_->smoothed_loss_;
    Timer& iteration_timer_ = solver_->iteration_timer_;
    int& iter_ = solver_->iter_;
    bool& requested_early_exit_ = solver_->requested_early_exit_;
    float& iterations_last_ = solver_->iterations_last_;

    shared_ptr<Net<float> > net_ = solver_->net();
    const int start_iter = solver_->iter_;
    const int stop_iter = start_iter + iters;
    int average_loss = param_.average_loss();
    solver_->losses_.clear();
    smoothed_loss_ = 0;
    iteration_timer_.Start();

    while (iter_ < stop_iter) {
        // zero-init the params
        net_->ClearParamDiffs();
        if (param_.test_interval() && iter_ % param_.test_interval() == 0
                && (iter_ > 0 || param_.test_initialization())) {
            if (Caffe::root_solver()) {
                solver_->TestAll();
            }
            if (requested_early_exit_) {
                // Break out of the while loop because stop was requested while testing.
                break;
            }
        }

        solver_->v_callbacks_start();
        const bool display = param_.display() && iter_ % param_.display() == 0;
        net_->set_debug_info(display && param_.debug_info());
        // accumulate the loss and gradient
        float loss = 0;
        for (int i = 0; i < param_.iter_size(); ++i) {
            loss += net_->ForwardBackward();
        }
        loss /= param_.iter_size();
        // average the loss across iterations for smoothed reporting
        solver_->UpdateSmoothedLoss(loss, start_iter, average_loss);
        if (display) {
            float lapse = iteration_timer_.Seconds();
            float per_s = (iter_ - iterations_last_) / (lapse ? lapse : 1);
            LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_
                                               << " (" << per_s << " iter/s, " << lapse << "s/"
                                               << param_.display() << " iters), loss = " << smoothed_loss_;
            iteration_timer_.Start();
            iterations_last_ = iter_;
            const vector<Blob<float>*>& result = net_->output_blobs();
            int score_index = 0;
            for (size_t j = 0; j < result.size(); ++j) {
                const float* result_vec = result[j]->cpu_data();
                const string& output_name =
                        net_->blob_names()[net_->output_blob_indices()[j]];
                const float loss_weight =
                        net_->blob_loss_weights()[net_->output_blob_indices()[j]];
                for (int k = 0; k < result[j]->count(); ++k) {
                    ostringstream loss_msg_stream;
                    if (loss_weight) {
                        loss_msg_stream << " (* " << loss_weight
                                        << " = " << loss_weight * result_vec[k] << " loss)";
                    }
                    LOG_IF(INFO, Caffe::root_solver()) << "    Train net output #"
                                                       << score_index++ << ": " << output_name << " = "
                                                       << result_vec[k] << loss_msg_stream.str();
                }
            }
        }
        solver_->v_callbacks_ready();
        solver_->ApplyUpdate();
        // 对指定层的权重部分 kernel 置零
//        LOG(INFO) << "before zerod";
        sp_mynet_->zeroByRate();
//        LOG(INFO) << "after zerod";

        // Increment the internal iter_ counter -- its value should always indicate
        // the number of times the weights have been updated.
        ++iter_;

        caffe::SolverAction::Enum request = solver_->GetRequestedAction();

        // Save a snapshot if needed.
        if ((param_.snapshot()
             && iter_ % param_.snapshot() == 0
             && Caffe::root_solver()) ||
                (request == caffe::SolverAction::SNAPSHOT)) {
            solver_->Snapshot();
        }
        if (caffe::SolverAction::STOP == request) {
            requested_early_exit_ = true;
            // Break out of training loop.
            break;
        }
    }
}

} // end of namespace mytest
