#ifndef ZERO_MATRIX_TASK_H
#define ZERO_MATRIX_TASK_H

#include "legion.h"
#include "macros.h"
using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;


void register_zero_matrix_task();


class ZeroMatrixTask : public TaskLauncher {
 public:
  ZeroMatrixTask(TaskArgument arg,
		 Predicate pred = Predicate::TRUE_PRED,
		 MapperID id = 0,
		 MappingTagID tag = 0);
  
  static void register_task(void);

  static void cpu_task(const Task *task,
		       const std::vector<PhysicalRegion> &regions,
		       Context ctx, HighLevelRuntime *runtime);
  // public:
 private:
  static int TASKID;
};


#endif // ZERO_MATRIX_TASK_H
