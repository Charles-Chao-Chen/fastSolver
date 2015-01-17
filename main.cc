#include <iostream>

#include "fast_solver.h"
#include "direct_solve.h"
#include "custom_mapper.h"

enum {
  MASTER_TASK_ID = 0,
};

void run_test      (int rank, int N, int threshold,
		    int nleaf_per_legion_node, double diag,
		    bool compute_accuracy, Context ctx,
		    HighLevelRuntime *runtime);

void top_level_task(const Task *task,
		    const std::vector<PhysicalRegion> &regions,
		    Context ctx, HighLevelRuntime *runtime);

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


void top_level_task(const Task *task,
		    const std::vector<PhysicalRegion> &regions,
		    Context ctx, HighLevelRuntime *runtime) {

#if 0
  //test_accuracy
  run_test(6,   /* rank */
	   15*(8),  /* N */
	   15,   /* threshold*/
	   1,    /* nleaf_per_legion_node */
	   1.e1, /* diagonal */
	   true, /* compute accuracy */
	   ctx,
	   runtime);
#else
  //test_performance
  run_test(400,   /* rank */
	   1<<15, /* N */
	   1<<9,  /* threshold*/
	   2,     /* nleaf_per_legion_node */
	   1.e5,  /* diagonal */
	   false, /* compute accuracy */
	   ctx,
	   runtime);
#endif

  return;
}


// leaf_size: how many real leaves every legion
// node has
void run_test(int rank, int N, int threshold,
	      int leaf_size, double diag,
	      bool compute_accuracy,
	      Context ctx, HighLevelRuntime *runtime) {

  int rand_seed = 1123;

  // TODO: bug: it seems all leaf_solve tasks run on node0 when
  //  num_node=4 with two machines
  int num_proc = 2;
  int rhs_cols = 2;
  int rhs_rows = N;
  
  HodlrMatrix hMatrix;

  // create H-tree with legion leaf
  hMatrix.create_tree(N, threshold, rhs_cols, rank,
		      leaf_size, ctx, runtime);
  int nleaf = hMatrix.get_num_leaf();
  std::cout << "Number of legion leaves: "
	    << nleaf
	    << std::endl;

  // random right hand size
  hMatrix.init_rhs(rand_seed, rhs_cols, num_proc, ctx, runtime);
  
  // A = U U^T + diag and U is a circulant matrix
  hMatrix.init_circulant_matrix(diag, num_proc, ctx, runtime);
    
 
  FastSolver fs;
  fs.solve_dfs(hMatrix, num_proc, ctx, runtime);
  //fs.solve_bfs(hMatrix, num_proc, ctx, runtime);
  
  std::cout << "Tasks launching time: " << fs.get_elapsed_time()
	    << std::endl;

  if (compute_accuracy) {
    assert( N%threshold == 0 );
    int nregion = nleaf;
    compute_L2_error(hMatrix, rand_seed, rhs_rows, nregion,
		     rhs_cols, rank, diag, ctx, runtime);
  }
}
