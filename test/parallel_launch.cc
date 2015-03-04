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

std::string AddSuffix(const std::string str, int x) {
  std::stringstream ss;
  ss << x;
  return str + ss.str();
}

void top_level_task(const Task *task,
		    const std::vector<PhysicalRegion> &regions,
		    Context ctx, HighLevelRuntime *runtime) {  
 
  // assume the tree is balanced
  int gloTreeLevel = 4;
  int launchAtLevel = 2; // two sub-tasks
  int locTreeLevel = gloTreeLevel - launchAtLevel + 1;
  int numTasks = pow(2, launchAtLevel-1);
  //int numLocalLeaf = pow(2, locTreeLevel);

  // TODO: class for array of regions
  // LogicalRegion regionsGlobal[ numTasks ][ numLocalLeaf ];
    
  // ---------------------------------------------------------
  // the first step is to launch two solvers seperately: check
  // the second step is to have two local partial solves
  // ---------------------------------------------------------

  // target at 32 nodes
  int numMachineNodes = 2;
  int nodesPerTask = numMachineNodes / numTasks;

  // random seed
  long seed = 1245667;
  
  // launch sub-tasks using loop
  // TODO: use IndexLaunch instead for efficiency
  for (int i=0, nodeIdx=0; i<numTasks; i++, nodeIdx+=nodesPerTask) {
    Range rg(nodeIdx, nodesPerTask);
    std::string taskName = AddSuffix( "sub", i );
    SubSolveTask::TaskArgs args(locTreeLevel, gloTreeLevel,
				seed, taskName, rg);
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
