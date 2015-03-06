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
  int launchAtLevel = 1; // two sub-tasks
  int locTreeLevel = gloTreeLevel - launchAtLevel;
  int numTasks = pow(2, launchAtLevel);
  //int numLocalLeaf = pow(2, locTreeLevel);

  // TODO: class for array of regions
  // LogicalRegion regionsGlobal[ numTasks ][ numLocalLeaf ];
    
  // ---------------------------------------------------------
  // the first step is to launch two solvers seperately: check
  // the second step is to have two local partial solves: check
  // the last step is to solve the global problem
  // ---------------------------------------------------------

  // target at 32 nodes
  int numMachineNodes = 2;
  int nodesPerTask = numMachineNodes / numTasks;

  // random seed
  long seed = 1245667;
  LMatrixArray matArr;
  
  // launch sub-tasks using loop
  // TODO: use IndexLaunch instead for efficiency
  for (int i=0, nodeIdx=0; i<numTasks; i++, nodeIdx+=nodesPerTask) {
    Range rg(nodeIdx, nodesPerTask);
    std::string taskName = AddSuffix( "sub", i );
    SubSolveTask::TaskArgs args(locTreeLevel, gloTreeLevel,
				seed, taskName, rg);
    SubSolveTask launcher(TaskArgument(&args, sizeof(args)));
    Future f = runtime -> execute_task(ctx, launcher);
    matArr  += f.get_result<LMatrixArray>();
  }
  
  // TODO: solve the global problem
  // TODO: use a different container for LMatrixArray
  //  probably a queue
  
  int gloLevel = gloTreeLevel;
  int subLevel = gloLevel;

  // TODO: pass them into sub-tasks.
  //   make sure these parameters are consitent throughout the solve
  int nRHS = 2;             // # of rhs
  int rank = 50;
  int threshold = 150;
  int leafSize = 1;         // legion leaf size
  double diagonal = 1.0e4;
  int nRow = threshold*(1<<subLevel);
  const char* name = "global";
  Range procs(numMachineNodes);
  
  HodlrMatrix hMat(nRHS, nRow, gloLevel, subLevel, rank,
                   threshold, leafSize, name);
  hMat.create_tree( ctx, runtime, &matArr );
  hMat.init_circulant_matrix(diagonal, procs, ctx, runtime,
			     true); //skip UMat
  
  FastSolver solver;
  solver.solve_top( hMat, procs, ctx, runtime );

  if (true) {
    assert( nRow%threshold == 0 );
    int nregion = hMat.get_num_leaf();
    compute_L2_error(hMat, seed, nRow, nregion, nRHS,
		     rank, diagonal, ctx, runtime);
  }
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
