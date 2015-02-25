#ifndef _SAVE_TASK_H
#define _SAVE_TASK_H

#include "string"
#include "legion.h"
#include "range.h"

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;


void register_save_region_task();


class SaveRegionTask : public TaskLauncher {
    
 public:
  struct TaskArgs {
    Range col_range;
    std::string filename;
  };

  SaveRegionTask(TaskArgument arg,
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
  

#endif //_SAVE_TASK_H
