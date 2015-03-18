#include <iostream>
#include <string>
#include <sstream>
#include <math.h>

#include "range.h"
#include "fast_solver.h"
#include "direct_solve.h"
#include "legion.h"
#include "custom_mapper.h"

enum {
  TOP_LEVEL_TASK_ID = 0,
};

void top_level_task(const Task *task,
		    const std::vector<PhysicalRegion> &regions,
		    Context ctx, HighLevelRuntime *runtime) {  
 
  // assume the tree is balanced
  int gloTreeLevel = 3;
  int numMachineNodes = 2;

  // random seed
  long seed = 1245667;

  // ---------------------------------------------------------
  // Problem configuration
  int nRHS = 2;             // # of rhs
  int rank = 30; //300;
  int threshold = 60; //600;
  int leafSize = 1;         // legion leaf size
  double diagonal = 1.0e4;
  // ---------------------------------------------------------  
    
  int gloLevel = gloTreeLevel;
  int subLevel = gloLevel;  
  int nRow = threshold*(1<<subLevel);
  const char* name = "global";
  Range procs(numMachineNodes);
  
  HodlrMatrix hMatrix(nRHS, nRow, gloLevel, subLevel, rank,
			threshold, leafSize, name);
  hMatrix.create_tree( ctx, runtime );

  //-----------------------------------------------
  // deferred initialization
  // (1) umat : add row_beg in leaf_solve()
  // (2) kmat : the same as above
  // (3) vmat : the same as above
  //-----------------------------------------------

  
  // random right hand side
  hMatrix.init_rhs(seed, procs, ctx, runtime);
  hMatrix.init_circulant_matrix(diagonal, procs, ctx, runtime);
  
  FastSolver fs;
  fs.bfs_solve(hMatrix, procs, ctx, runtime);
  
  std::cout << "\n================================" << std::endl;
  std::cout << "problem information:" << std::endl;
  int nleaf = hMatrix.get_num_leaf();
  std::cout << "  Legion leaf : "       << nleaf << std::endl
	    << "  Legion leaf / node: " << nleaf / procs.size()
	    << std::endl;
  hMatrix.display_launch_time();
  fs.display_launch_time();

  assert( nRow%threshold == 0 );
  int nregion = nleaf;
  if (true) {
    compute_L2_error(hMatrix, seed, nRow, nregion, nRHS,
         		   rank, diagonal, ctx, runtime);
  }
  std::cout << "================================\n" << std::endl;
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

  // register fast solver tasks 
  register_solver_tasks();
  // register customized mapper
  register_custom_mapper();
  // start legion master task
  return HighLevelRuntime::start(argc, argv);
}
