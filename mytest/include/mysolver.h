#ifndef MYSOLVER_H
#define MYSOLVER_H

#include <cpu_only.h>
#include <mynet.h>

//namespace SolverAction {
//  enum Enum {
//    NONE = 0,  // Take no special action.
//    STOP = 1,  // Stop training. snapshot_after_train controls whether a
//               // snapshot is created.
//    SNAPSHOT = 2  // Take a snapshot, and keep training.
//  };
//}

namespace mytest {

class MySolver
{
public:
    // Invoked at specific points during an iteration
    class Callback {
    protected:
        virtual void on_start() = 0;
        virtual void on_gradients_ready() = 0;

        template <typename T>
        friend class Solver;
    };

    //    friend class Solver<float>;
    MySolver(shared_ptr<Solver<float> > Solve,
             const vector<string>& v_zerodLayers);

    void Solve(const char* resume_file = NULL);
    void Step(int iters);

    shared_ptr<Solver<float> > solver_;
    shared_ptr<MyNet> sp_mynet_;
    float zeroRate_;

    float top5acc;
};

} // end of namespace mytest

#endif // MYSOLVER_H
