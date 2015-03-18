#ifndef INIT_MATRIX_TASKS_H
#define INIT_MATRIX_TASKS_H

#include "legion.h"
#include "hodlr_matrix.h"

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;


void register_init_tasks();


class RandomMatrixTask : public TaskLauncher {
 public:
  struct TaskArgs {
    long int seed;
    int ncol;
  };
  
  RandomMatrixTask(TaskArgument arg,
	      Predicate pred = Predicate::TRUE_PRED,
	      MapperID id = 0,
	      MappingTagID tag = 0);
  
  static int TASKID;
  static void register_tasks(void);

 public:
  static void cpu_task(const Task *task,
		       const std::vector<PhysicalRegion> &regions,
		       Context ctx, HighLevelRuntime *runtime);
};

class InitCirculantKmatTask : public TaskLauncher {
 public:
  template <int N>
    struct TaskArgs {
      //int treeSize;
      int row_beg_global;
      int rank;
      //int LD; // leading dimension
      double diag;
      Node treeArray[N];
    };
  
  InitCirculantKmatTask(TaskArgument arg,
			Predicate pred = Predicate::TRUE_PRED,
			MapperID id = 0,
			MappingTagID tag = 0);
  
  static int TASKID;
  static void register_tasks(void);

 public:
  static void cpu_task(const Task *task,
		       const std::vector<PhysicalRegion> &regions,
		       Context ctx, HighLevelRuntime *runtime);
};

class InitCirculantMatrixTask : public TaskLauncher {
 public:
  struct TaskArgs {
    int col_beg;
    int row_beg;
    int rank;
  };
    
  InitCirculantMatrixTask(TaskArgument arg,
			  Predicate pred = Predicate::TRUE_PRED,
			  MapperID id = 0,
			  MappingTagID tag = 0);
  
  static int TASKID;
  static void register_tasks(void);

 public:
  static void cpu_task(const Task *task,
		       const std::vector<PhysicalRegion> &regions,
		       Context ctx, HighLevelRuntime *runtime);
};





#endif // INIT_MATRIX_TASKS_H
