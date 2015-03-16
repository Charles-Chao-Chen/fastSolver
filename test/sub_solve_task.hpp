#ifndef sub_solve_task_hpp
#define sub_solve_task_hpp

#include "range.h"
#include "fast_solver.h"
#include "direct_solve.h"
#include "matrix_array.hpp"
#include "legion.h"

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;

class SubSolveTask : public TaskLauncher {
 public:
  struct TaskArgs {
    TaskArgs
    (int sl, int gl, long s, const std::string& str, const Range& p,
     int rhs, int r, int t, int ls, double d);

    // sub-problem configuration
    int subLvl;
    int gloLvl;
    long seed;
    char name[50];
    Range procs;

    // h-matrix configuration
    int nRHS;
    int rank;
    int threshold;
    int leafSize;
    double diagonal;
  };
  
  SubSolveTask(TaskArgument arg,
	      Predicate pred = Predicate::TRUE_PRED,
	      MapperID id = 0,
	      MappingTagID tag = 0);
  
  static int TASKID;
  static void register_tasks(void);
 public:
  static LMatrixArray
  cpu_task (const Task *task,
	    const std::vector<PhysicalRegion> &regions,
	    Context ctx, HighLevelRuntime *runtime);
};


#endif // sub_solve_task_hpp
