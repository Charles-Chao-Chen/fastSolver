#include <iostream>

#include "fastSolver.h"
#include "direct_solve.h"
#include "custom_mapper.h"
#include "timer.h"

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
  run_test(10,   /* rank */
	   800,  /* N */
	   50,   /* threshold*/
	   1,    /* nleaf_per_legion_node */
	   1.e5, /* diagonal */
	   true, /* compute accuracy */
	   ctx,
	   runtime);
#else
  //test_performance
  run_test(150,   /* rank */
	   1<<14, /* N */
	   1<<8,  /* threshold*/
	   1,     /* nleaf_per_legion_node */
	   1.e5,  /* diagonal */
	   false, /* compute accuracy */
	   ctx,
	   runtime);
#endif

  return;
}


void run_test(int rank, int N, int threshold,
	      int nleaf_per_legion_node, double diag,
	      bool compute_accuracy,
	      Context ctx, HighLevelRuntime *runtime) {

  int rand_seed = 1123;
  int num_node = 4;
  int rhs_cols = 1;
  int rhs_rows = N;
  
  LR_Matrix lr_mat(N, threshold, rhs_cols, rank, ctx, runtime);
  lr_mat.create_legion_leaf(nleaf_per_legion_node);  
  lr_mat.init_RHS(rand_seed, num_node);
  
  // A = U U^T + diag and U is a circulant matrix
  lr_mat.init_circulant_matrix(diag, num_node);
    
 
  FastSolver fs;
  fs.solve_dfs(lr_mat, num_node, ctx, runtime);
  //fs.solve_bfs(lr_mat, num_node, ctx, runtime);

  
  std::cout << "Tasks launching time: "
	    << fs.get_elapsed_time()
	    << std::endl;

  /*
  if (compute_accuracy) {
    // write the solution from fast solver
    const char *soln_file = "soln.txt";
    if (remove(soln_file) == 0)
      std::cout << "Remove old solution file." << std::endl;
  
    lr_mat.save_solution(soln_file);
  
    assert( N%threshold == 0);
    dirct_circulant_solve(soln_file, rand_seed, rhs_rows, N/threshold, rhs_cols, rank, diag);
  }
  */
}
