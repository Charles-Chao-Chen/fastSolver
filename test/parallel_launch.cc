#include <iostream>
#include <string>
#include <sstream>
#include <math.h>

#include "legion.h"
#include "sub_solve_task.hpp"
#include "custom_mapper.h"

enum {
  TOP_LEVEL_TASK_ID = 0,
};


void top_level_task(const Task *task,
		    const std::vector<PhysicalRegion> &regions,
		    Context ctx, HighLevelRuntime *runtime) {  
 
  // assume the tree is balanced
  int treeLevel = 9;
  int launchAtLevel = 2;
  int localTreeLevel = treeLevel - launchAtLevel + 1;  
  int numTasks = pow(2, launchAtLevel);
  int numLocalLeaf = pow(2, localTreeLevel);
  LogicalRegion regionsGlobal[ numTasks ][ numLocalLeaf ];

  // target at 32 nodes as the first step
  int numMachineNodes = 32;
  int nodesPerTask = numMachineNodes / numTasks;
  
  // -------------------------------------------------
  // the first step is to launch two solvers seperately
  // -------------------------------------------------

  numTasks = 2; // two sub-tasks now
  nodesPerTask = 1; // run each sub-task on one node
  localTreeLevel = 6; // rank, threshold and other parameters
                      //  are set inside the sub-tasks
  
  // TODO: use IndexLaunch instead for efficiency
  int nodeIdx = 0;
  std::string str("sub");
  for (int i=0; i<numTasks; i++, nodeIdx+=nodesPerTask) {
    Range rg(nodeIdx, nodesPerTask);
    std::stringstream ss;
    ss << i;
    SubSolveTask::TaskArgs args(localTreeLevel, rg, str+ss.str());

    // TODO: specify as "inner tasks"
    SubSolveTask launcher(TaskArgument(&args, sizeof(args))); 
    
    // no region requirement needed
    Future f = runtime -> execute_task(ctx, launcher);
    //f.get_void_result();
    
    // TODO: return region array
  }
  
  // TODO: solve the global problem
  /*
  HTree treeGlobal;
  build_global_tree( treeGlobal, regionsGlobal );
  FastSolver solver;
  solver.solve( treeGlobal );
  */
}


int main(int argc, char *argv[]) {

  // register top level task
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  HighLevelRuntime::register_legion_task<top_level_task>(
    TOP_LEVEL_TASK_ID,   /* task id */
    Processor::LOC_PROC, /* proc kind */
    true,  /* single */
    false, /* index  */
    AUTO_GENERATE_ID,
    TaskConfigOptions(false /*leaf task*/),
    "master-task"
  );

  SubSolveTask::register_tasks();
  
  // register fast solver tasks 
  register_solver_tasks();
    
  // register customized mapper
  register_custom_mapper();

  // start legion master task
  return HighLevelRuntime::start(argc, argv);
}
