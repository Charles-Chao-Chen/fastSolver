#include <iostream>
#include <time.h>

#include "custom_mapper.h"
#include "fastSolver.h"
#include "utility.h"

enum {
  TOP_LEVEL_TASK_ID = 0,
};


void test_accuracy(Context ctx, HighLevelRuntime *runtime);
void test_performance(Context ctx, HighLevelRuntime *runtime);

void top_level_task(const Task *task,
		    const std::vector<PhysicalRegion> &regions,
		    Context ctx, HighLevelRuntime *runtime) {
  
  //test_accuracy(ctx, runtime);
  test_performance(ctx, runtime);

  return;
}


int main(int argc, char *argv[]) {

  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
							 Processor::LOC_PROC, true, false);

  register_solver_task();
  register_gemm_task();
  register_save_task();
  register_zero_matrix_task();
  register_circulant_matrix_task();
  register_circulant_kmat_task();

  
  SaveRegionTask::register_tasks();
  InitRHSTask::register_tasks();
  LUSolveTask::register_tasks();

  
  HighLevelRuntime::set_registration_callback(mapper_registration);

 
  return HighLevelRuntime::start(argc, argv);
}


void test_accuracy(Context ctx, HighLevelRuntime *runtime) {


  // set random seed
  //srand( time(NULL) );
  clock_t t0 = clock();  


  // make sure the dense block is bigger than r
  int r = 10;
  int N = 200;
  int threshold = 50;
  int nleaf_per_legion_node = 1;
  double diag = 1e5; 

  int rand_seed = 1123;
  
  // get input args
  //const InputArgs &command_args = HighLevelRuntime::get_input_args();
  //diag = atof(command_args.argv[1]);


  int rhs_cols = 1;
  int rhs_rows = N;
  double *rhs = (double*) malloc(rhs_cols*rhs_rows*sizeof(double));
  for (int j=0; j<rhs_cols; j++)
    for (int i=0; i<rhs_rows; i++)
      rhs[i+j*rhs_rows] = frand(0, 1);
    
  LR_Matrix lr_mat(N, threshold, rhs_cols, r, ctx, runtime);
  lr_mat.create_legion_leaf(nleaf_per_legion_node);  
  lr_mat.init_RHS( rand_seed, true /*wait*/ );
  //lr_mat.init_RHS(rhs);
  
  
  // A = U U^T + diag and U is a circulant matrix
  lr_mat.init_circulant_matrix(diag); 

  //lr_mat.save_solution("rhs.txt");
  //save_region(lr_mat.uroot, "Umat.txt", ctx, runtime);
  //lr_mat.print_Vmat(lr_mat.vroot, "Vmat.txt");

  
  
  FastSolver fs(ctx, runtime);
  fs.recLU_solve(lr_mat);


  // write the solution to file
  const char *soln_file = "solution.txt";
  int rm = remove(soln_file);
  if (rm == 0)
    std::cout << "Removed solution file." << std::endl;
  
  lr_mat.save_solution(soln_file);
  //save_region(lr_mat.uroot, "Umat.txt", ctx, runtime);

  
  clock_t t1 = clock();
  printf("Init Time: %f.\n", (double)(t1-t0)/CLOCKS_PER_SEC);

  double *Soln = (double *) malloc(N*rhs_cols*sizeof(double));
  //lr_mat.get_soln_from_region(Soln);

  //dirct_circulant_solve(Soln, rhs, rhs_rows, rhs_cols, r, diag);
  assert( N%threshold == 0);
  //dirct_circulant_solve(Soln, rand_seed, rhs_rows, N/threshold, rhs_cols, r, diag);

  dirct_circulant_solve(soln_file, rand_seed, rhs_rows, N/threshold, rhs_cols, r, diag);
  
  free(Soln); Soln = NULL;
  free(rhs); rhs = NULL;

}


void test_performance(Context ctx, HighLevelRuntime *runtime) {


  // set random seed
  //srand( time(NULL) );
  clock_t t0 = clock();  


  // make sure the dense block is bigger than r
  int r = 150;
  int N = 12800;
  int threshold = 400;
  int nleaf_per_legion_node = 1;
  double diag = 1e5; 
  
  // get input args
  //const InputArgs &command_args = HighLevelRuntime::get_input_args();
  //diag = atof(command_args.argv[1]);



  int rhs_cols = 1;
  int rhs_rows = N;

  /*
  double *rhs = (double*) malloc(rhs_cols*rhs_rows*sizeof(double));
  for (int j=0; j<rhs_cols; j++)
    for (int i=0; i<rhs_rows; i++)
      rhs[i+j*rhs_rows] = frand(0, 1);
*/

    
  LR_Matrix lr_mat(N, threshold, rhs_cols, r, ctx, runtime);
  lr_mat.create_legion_leaf(nleaf_per_legion_node);  
  //lr_mat.init_RHS(rhs);
  lr_mat.init_RHS(1234);
  
  // A = U U^T + diag and U is a circulant matrix
  lr_mat.init_circulant_matrix(diag); 

  FastSolver fs(ctx, runtime);
  fs.recLU_solve(lr_mat);

  
  clock_t t1 = clock();
  printf("Init Time: %f.\n", (double)(t1-t0)/CLOCKS_PER_SEC);

  //free(rhs); rhs = NULL;

}
