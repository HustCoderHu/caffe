#ifndef HELPER_FUNCTIONS_HPP
#define HELPER_FUNCTIONS_HPP

#include <cpu_only.h>
#include <boost/algorithm/string.hpp>

namespace mytest {

static inline string getLayerName(int group, char block,
                                  const string& branch) {
//    2, b, 2c
    ostringstream lname;
    lname << "res" << group << block << "_branch" << branch;
    return lname.str(); // res2b_branch2c
}

static string resnet50_zerodLayers(vector<string>& v_zerodLayers)
{
    vector<int> v_group = {2, 3, 4, 5};
    vector<vector<char>> v_v_block = { {'a', 'b', 'c'},
                                       {'a', 'b', 'c', 'd'},
                                       {'a', 'b', 'c', 'd', 'e', 'f'},
                                       {'a', 'b', 'c'}
                                     };
    vector<string> v_branch = {"2a", "2b"};

    v_zerodLayers.reserve(v_group.size() * v_v_block.size()
                          * v_branch.size());
    ostringstream format;
    for (size_t i = 0; i != v_group.size(); ++i) {
        vector<char>& v_block = v_v_block[i];
        for (size_t j = 0; j != v_block.size(); ++j) {
//            format.str("");
            for (size_t k = 0; k != v_branch.size(); ++k) {
                string lname = getLayerName(v_group.at(i), v_block.at(j),
                                            v_branch.at(k));
                v_zerodLayers.push_back(lname);
                format << lname << ", ";
            }
            format << endl;
        }
        format << endl;
    }
    return format.str();
}


// Parse stages from flags
//vector<string> get_stages_from_flags(const string& FLAGS_stage) {
//  vector<string> stages;
//  boost::split(stages, FLAGS_stage, boost::is_any_of(","));
//  return stages;
//}

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
    std::vector<std::string> model_names;
    boost::split(model_names, model_list, boost::is_any_of(",") );
    for (size_t i = 0; i < model_names.size(); ++i) {
        LOG(INFO) << "Finetuning from " << model_names[i];
        solver->net()->CopyTrainedLayersFrom(model_names[i]);
        for (size_t j = 0; j < solver->test_nets().size(); ++j) {
            solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
        }
    }
}

// Translate the signal effect the user specified on the command-line to the
// corresponding enumeration.
static caffe::SolverAction::Enum GetRequestedAction(
    const std::string& flag_value) {
  if (flag_value == "stop") {
    return caffe::SolverAction::STOP;
  }
  if (flag_value == "snapshot") {
    return caffe::SolverAction::SNAPSHOT;
  }
  if (flag_value == "none") {
    return caffe::SolverAction::NONE;
  }
  LOG(FATAL) << "Invalid signal effect \""<< flag_value << "\" was specified";
}

} // end of namespace mytest

#endif // HELPER_FUNCTIONS_HPP
