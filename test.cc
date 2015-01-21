#include "test.h"



void test_accuracy(Context ctx, HighLevelRuntime *runtime) {
  run_test(6,   /* rank */
	   15*(8),  /* N */
	   15,   /* threshold*/
	   1,    /* nleaf_per_legion_node */
	   1.e1, /* diagonal */
	   1,    /* # of processors */
	   true, /* compute accuracy */
	   ctx,
	   runtime);
}


void test_performance(Context ctx, HighLevelRuntime *runtime) {
  run_test(300,   /* rank */
	   8*400*(1<<(2+5)), /* N */
	   600,   /* threshold*/
	   1,     /* nleaf_per_legion_node */
	   1.e5,  /* diagonal */
	   1<<0,  /* # of processors */
	   false, /* compute accuracy */
	   ctx,
	   runtime);
}


// leaf_size: how many real leaves every legion
// node has
void run_test(int rank, int N, int threshold,
	      int leaf_size, double diag, int num_proc,
	      bool compute_accuracy,
	      Context ctx, HighLevelRuntime *runtime) {

  int rhs_cols = 2;
  int rhs_rows = N;
  int rand_seed = 1123;
  
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
  fs.solve_bfs(hMatrix, num_proc, ctx, runtime);
  //fs.solve_dfs(hMatrix, num_proc, ctx, runtime);
  
  std::cout << "Tasks launching time: " << fs.get_elapsed_time()
	    << std::endl;

  if (compute_accuracy) {
    assert( N%threshold == 0 );
    int nregion = nleaf;
    compute_L2_error(hMatrix, rand_seed, rhs_rows, nregion,
		     rhs_cols, rank, diag, ctx, runtime);
  }
}
