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

#if 1
  //test_accuracy
  run_test(6,   /* rank */
	   240,  /* N */
	   15,   /* threshold*/
	   1,    /* nleaf_per_legion_node */
	   1.e1, /* diagonal */
	   true, /* compute accuracy */
	   ctx,
	   runtime);
#else
  //test_performance
  run_test(150,   /* rank */
	   1<<14, /* N */
	   1<<8,  /* threshold*/
	   4,     /* nleaf_per_legion_node */
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
  int num_node = 4;
  int rhs_cols = 2;
  int rhs_rows = N;
  
  HodlrMatrix lr_mat;

  // create H-tree with legion leaf
  lr_mat.create_tree(N, threshold, rhs_cols, rank,
				 leaf_size, ctx, runtime);
  int nleaf = lr_mat.get_num_leaf();
  std::cout << "Number of legion leaves: "
	    << nleaf
	    << std::endl;

  // random right hand size
  lr_mat.init_right_hand_side(rand_seed, rhs_cols, num_node,
			      ctx, runtime);
  
  // A = U U^T + diag and U is a circulant matrix
  lr_mat.init_circulant_matrix(diag, num_node, ctx, runtime);
    
 
  FastSolver fs;
  fs.solve_dfs(lr_mat, num_node, ctx, runtime);
  //fs.solve_bfs(lr_mat, num_node, ctx, runtime);
  
  std::cout << "Tasks launching time: " << fs.get_elapsed_time()
	    << std::endl;

  if (compute_accuracy) {
    assert( N%threshold == 0 );
    int nregion = nleaf;
    compute_L2_error(lr_mat, rand_seed, rhs_rows, nregion,
		     rhs_cols, rank, diag, ctx, runtime);
  }

}
