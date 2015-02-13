#include "test.h"
#include "timer.h"

// this test aims at scalability for 32 nodes
/* Note: the leaf solve takes about 70% of the execution time;
    aim at iso-efficience for 70%; small rank
 */
void test3
(int nproc, int nleaf, Context ctx, HighLevelRuntime *runtime) {
  run_test(100,   /* rank */
	   400*(1<<(1+12)), /* N */
	   600,   /* threshold*/
	   nleaf, /* nleaf_per_legion_node */
	   1.e5,  /* diagonal */
	   nproc, /* # of processors */
	   false, /* compute accuracy */
	   ctx,
	   runtime);
}


/* test 3.3 : small rank, timing for 1-16 nodes:
     23.56, 12.92, 6.56, 3.44, 2
 */
/*
void test3(int nproc, Context ctx, HighLevelRuntime *runtime) {
run_test(200,   
	   400*(1<<(1+8)), 
	   600,   
	   nleaf,     
	   1.e5,  
	   nproc, 
	   false, 
	   ctx,
	   runtime);
}
*/


/* test 3.2, timing for 1-16 nodes:
     53.4, 28.2, 14.36, 7.50, 4.34
 */
/*
void test3(int nproc, Context ctx, HighLevelRuntime *runtime) {
run_test(300,   
	   400*(1<<(1+8)), 
	   600,   
	   nleaf,     
	   1.e5,  
	   nproc, 
	   false, 
	   ctx,
	   runtime);
}
*/


/* test 3.1, timing for 1-16 nodes:
     23.5, 11.6, 6.1, 3.3, 2.1
 */
/*
void test3(int nproc, Context ctx, HighLevelRuntime *runtime) {
run_test(300,   
	   400*(1<<(1+7)), 
	   600,   
	   1,     
	   1.e5,  
	   nproc, 
	   false, 
	   ctx,
	   runtime);
}
*/

void test_accuracy(int nproc, Context ctx, HighLevelRuntime *runtime) {
  run_test(6,   /* rank */
	   15*(8),  /* N */
	   15,   /* threshold*/
	   1,    /* nleaf_per_legion_node */
	   1.e1, /* diagonal */
	   nproc,    /* # of processors */
	   true, /* compute accuracy */
	   ctx,
	   runtime);
}


void test_performance(Context ctx, HighLevelRuntime *runtime) {
  run_test(300,   /* rank */
	   400*(1<<(2+5)), /* N */
	   600,   /* threshold*/
	   1,     /* nleaf_per_legion_node */
	   1.e5,  /* diagonal */
	   1<<0,  /* # of processors */
	   false, /* compute accuracy */
	   ctx,
	   runtime);
}


// This test case is for performance debugging after Mike
//  fixed the long message handler issue.
// Run: two nodes
// Issue: node 1 has unnecessary idle time between leaf_solve()
//  tasks that have no dependency
//  on each other. This is very similar to the message handler
//  issue except the message handler is gone. The consequence
//  is node 1 runs slower than node 0.
// Date: Jan. 21, 2015
void test1(int nproc, Context ctx, HighLevelRuntime *runtime) {
  run_test(300,   /* rank */
	   400*(1<<(1+6)), /* N */
	   600,   /* threshold*/
	   1,     /* nleaf_per_legion_node */
	   1.e5,  /* diagonal */
	   nproc,  /* # of processors */
	   false, /* compute accuracy */
	   ctx,
	   runtime);
}


// This test case is for performance debugging after Mike
//  fixed the long message handler issue.
// Run: two nodes
// Issue: node 0 has wired big post execution tasks on the
//  utility processors and the parallelism is not as good
//  as expected.
// Date: Jan. 21, 2015
void test2(int nproc, Context ctx, HighLevelRuntime *runtime) {
  run_test(300,   /* rank */
	   400*(1<<(1+4)), /* N */
	   600,   /* threshold*/
	   1,     /* nleaf_per_legion_node */
	   1.e5,  /* diagonal */
	   nproc,  /* # of processors */
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

#ifdef DEBUG
  std::cout << "************* debugging mode ************"
	    << std::endl;
#endif

  printf(" %d processes\n %d leaves\n\n", num_proc, leaf_size);
     
  int rhs_cols = 2;
  int rhs_rows = N;
  int rand_seed = 1123;
  
  HodlrMatrix hMatrix;

  // create H-tree with legion leaf
  hMatrix.create_tree(N, threshold, rhs_cols, rank,
		      leaf_size, num_proc, ctx, runtime);
  int nleaf = hMatrix.get_num_leaf();
  std::cout << "Legion leaf : "
	    << nleaf
	    << std::endl
	    << "Legion leaf / node: "
	    << nleaf / num_proc
	    << std::endl;

  std::cout << "Launch node : "
	    << hMatrix.get_num_launch_node()
	    << std::endl;
  
  double t0 = timer();
  
  // random right hand size
  hMatrix.init_rhs(rand_seed, rhs_cols, ctx, runtime);
  
  // A = U U^T + diag and U is a circulant matrix
  hMatrix.init_circulant_matrix(diag, ctx, runtime);

  double t1 = timer();
  std::cout << "Init launching time: " << t1 - t0
	    << std::endl;
  
 
  FastSolver fs;
  fs.bfs_solve(hMatrix, num_proc, ctx, runtime);
  //fs.solve_bfs(hMatrix, num_proc, ctx, runtime);
  //fs.solve_dfs(hMatrix, num_proc, ctx, runtime);
  
  std::cout << "Tasks launching time: " << fs.get_elapsed_time()
	    << std::endl;

  if (compute_accuracy) {
    assert( N%threshold == 0 );
    int nregion = nleaf;
    compute_L2_error(hMatrix, rand_seed, rhs_rows,
		     nregion, rhs_cols,
		     rank, diag, ctx, runtime);
  }
}
