#include <iostream>

#include "test.h"
#include "custom_mapper.h"

enum {
  MASTER_TASK_ID = 0,
};

void top_level_task(const Task *task,
		    const std::vector<PhysicalRegion> &regions,
		    Context ctx, HighLevelRuntime *runtime) {  

  int nproc = 1; // number of processes, which equals to
                 // the number of nodes
  
  // Check for any command line arguments
  {
    const InputArgs
      &command_args = HighLevelRuntime::get_input_args();
    for (int i = 1; i < command_args.argc; i++) {
      if (!strcmp(command_args.argv[i],"-np"))
	nproc = atoi(command_args.argv[++i]);
    }
  }
  printf("Running fast solver for %d processes...\n", nproc);
  

  //test_accuracy(ctx, runtime);
  //test_performance(ctx, runtime);
  //test2(ctx, runtime);
  test1(nproc, ctx, runtime);

  return;
}


int main(int argc, char *argv[]) {

  // register top level task
  HighLevelRuntime::set_top_level_task_id(MASTER_TASK_ID);
  HighLevelRuntime::register_legion_task<top_level_task>(
    MASTER_TASK_ID, /* task id */
    Processor::LOC_PROC, /* proc kind */
    true,  /* single */
    false, /* index  */
    AUTO_GENERATE_ID,
    TaskConfigOptions(false /*leaf task*/),
    "master-task"
  );

  // register fast solver tasks 
  register_solver_tasks();
    
  // register customized mapper
  register_custom_mapper();

  // start legion master task
  return HighLevelRuntime::start(argc, argv);
}
