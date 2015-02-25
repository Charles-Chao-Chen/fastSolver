#ifndef LAUNCH_NODE_TASK
#define LAUNCH_NODE_TASK

#include "range.h"


void register_launch_node_task();


class LaunchNodeTask : public TaskLauncher {
 public:
  template <int N>
  struct TaskArgs {
    Range taskTag;
    int   treeSize;
    FSTreeNode treeArray[N];
  };
  
  LaunchNodeTask(TaskArgument arg,
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


#endif // LAUNCH_NODE_TASK
