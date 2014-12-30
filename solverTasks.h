#ifndef _SOLVER_TASKS_H
#define _SOLVER_TASKS_H

#include "legion.h"

using namespace LegionRuntime::HighLevel;


void register_solver_operators();


class LUSolveTask : public TaskLauncher {
public:

  LUSolveTask(TaskArgument arg,
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


class LeafSolveTask : public TaskLauncher {
public:

  LeafSolveTask(TaskArgument arg,
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


#endif // _SOLVER_TASKS_H
