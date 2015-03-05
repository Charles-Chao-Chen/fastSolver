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
    TaskArgs(int sl, int gl, long s,
	     const std::string& str,
	     const Range& p) :
      subLvl(sl), gloLvl(gl),
      seed(s), procs(p)
    { strcpy(name, str.c_str()); }

    int subLvl;
    int gloLvl;
    int nRHS;
    int nRow;
    long seed;
    char name[50];
    Range procs;
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
