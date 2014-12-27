#include <iostream>
#include <time.h>

#include "custom_mapper.h"
#include "fastSolver.h"
#include "utility.h"

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
  //test_accuracy(ctx, runtime);
  run_test(10,   /* rank */
	   800,  /* N */
	   50,   /* threshold*/
	   1,    /* nleaf_per_legion_node */
	   1.e5, /* diagonal */
	   true, /* compute accuracy */
	   ctx,
	   runtime);
#else
  //test_performance(ctx, runtime);
  run_test(150,   /* rank */
	   1<<12, /* N */
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

  clock_t t0 = clock();  

  int rand_seed = 1123;
  
  int num_node = 4;
  int rhs_cols = 1;
  int rhs_rows = N;
  
  LR_Matrix lr_mat(N, threshold, rhs_cols, rank, ctx, runtime);
  lr_mat.create_legion_leaf(nleaf_per_legion_node);  
  lr_mat.init_RHS(rand_seed, num_node);
  
  // A = U U^T + diag and U is a circulant matrix
  lr_mat.init_circulant_matrix(diag, num_node);
    
 
  FastSolver fs(ctx, runtime);
  fs.recLU_solve(lr_mat, num_node);

  // display timing
  clock_t t1 = clock();
  printf("Init Time: %f.\n", (double)(t1-t0)/CLOCKS_PER_SEC);

  if (compute_accuracy) {
    // write the solution from fast solver
    const char *soln_file = "solution.txt";
    if (remove(soln_file) == 0)
      std::cout << "Remove old solution file." << std::endl;
  
    lr_mat.save_solution(soln_file);
  
    assert( N%threshold == 0);
    dirct_circulant_solve(soln_file, rand_seed, rhs_rows, N/threshold, rhs_cols, rank, diag);
  }
}


/*
void test_accuracy(Context ctx, HighLevelRuntime *runtime) {


  // set random seed
  //srand( time(NULL) );
  clock_t t0 = clock();  


  // make sure the dense block is bigger than r
  int r = 10;
  int N = 800;
  int threshold = 50;
  int nleaf_per_legion_node = 1;
  double diag = 1e5; 

  int rand_seed = 1123;
  
  // get input args
  //const InputArgs &command_args = HighLevelRuntime::get_input_args();
  //diag = atof(command_args.argv[1]);

  int num_node = 4;
  int rhs_cols = 1;
  int rhs_rows = N;
  double *rhs = (double*) malloc(rhs_cols*rhs_rows*sizeof(double));
  for (int j=0; j<rhs_cols; j++)
    for (int i=0; i<rhs_rows; i++)
      rhs[i+j*rhs_rows] = frand(0, 1);

  LR_Matrix lr_mat(N, threshold, rhs_cols, r, ctx, runtime);
  lr_mat.create_legion_leaf(nleaf_per_legion_node);  
  //lr_mat.init_RHS( rand_seed, true);
  //lr_mat.init_RHS(rhs);
  lr_mat.init_RHS( rand_seed, num_node );
  
  // A = U U^T + diag and U is a circulant matrix
  //lr_mat.init_circulant_matrix(diag); 
  lr_mat.init_circulant_matrix(diag, num_node);
  
  //lr_mat.save_solution("rhs.txt");
  //save_region(lr_mat.uroot, "Umat.txt", ctx, runtime);
  //lr_mat.print_Vmat(lr_mat.vroot, "Vmat.txt");

  
  
  FastSolver fs(ctx, runtime);
  fs.recLU_solve(lr_mat, num_node);
  //fs.recLU_solve(lr_mat);


  // write the solution to file
  const char *soln_file = "solution.txt";
  int rm = remove(soln_file);
  if (rm == 0)
    std::cout << "Removed solution file." << std::endl;
  
  lr_mat.save_solution(soln_file);
  //save_region(lr_mat.uroot, "Umat_2.txt", ctx, runtime);

  
  clock_t t1 = clock();
  printf("Init Time: %f.\n", (double)(t1-t0)/CLOCKS_PER_SEC);

  //double *Soln = (double *) malloc(N*rhs_cols*sizeof(double));
  //lr_mat.get_soln_from_region(Soln);

  //dirct_circulant_solve(Soln, rhs, rhs_rows, rhs_cols, r, diag);
  assert( N%threshold == 0);
  //dirct_circulant_solve(Soln, rand_seed, rhs_rows, N/threshold, rhs_cols, r, diag);

  dirct_circulant_solve(soln_file, rand_seed, rhs_rows, N/threshold, rhs_cols, r, diag);
  
  //free(Soln); Soln = NULL;
  free(rhs); rhs = NULL;

}


void test_performance(Context ctx, HighLevelRuntime *runtime) {


  // set random seed
  //srand( time(NULL) );
  clock_t t0 = clock();


  // make sure the dense block is bigger than r
  int r = 150;
  int N = 1<<12;
  int threshold = 1<<8;
  int nleaf_per_legion_node = 1;
  double diag = 1e5; 
  
  // get input args
  //const InputArgs &command_args = HighLevelRuntime::get_input_args();
  //diag = atof(command_args.argv[1]);

  int num_node = 4;
  int rhs_cols = 1;
  int rhs_rows = N;
  
  LR_Matrix lr_mat(N, threshold, rhs_cols, r, ctx, runtime);
  lr_mat.create_legion_leaf(nleaf_per_legion_node);  
  //lr_mat.init_RHS(rhs);
  //lr_mat.init_RHS(1234);
  lr_mat.init_RHS(1234, num_node);
  
  // A = U U^T + diag and U is a circulant matrix
  //lr_mat.init_circulant_matrix(diag);
  lr_mat.init_circulant_matrix(diag, num_node); 

  FastSolver fs(ctx, runtime);
  fs.recLU_solve(lr_mat, num_node);
  
  clock_t t1 = clock();
  printf("Init Time: %f.\n", (double)(t1-t0)/CLOCKS_PER_SEC);

  //free(rhs); rhs = NULL;
}
*/
