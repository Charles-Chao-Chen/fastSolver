#include <iostream>
#include <time.h>

#include "custom_mapper.h"
#include "fastSolver.h"
#include "utility.h"

enum {
  TOP_LEVEL_TASK_ID = 0,
};


//void test_assemble_matrix(Context ctx, HighLevelRuntime *runtime);

void top_level_task(const Task *task,
		    const std::vector<PhysicalRegion> &regions,
		    Context ctx, HighLevelRuntime *runtime) {

  //test_assemble_matrix(ctx, runtime);

  // set random seed
  //srand( time(NULL) );
  
  clock_t t0 = clock();

  // make sure the dense block is bigger than r
  int r = 60; //500;
  int N = 2000; //18000;
  int threshold = 200; //1000;
  int nleaf_per_legion_node = 2;
  double diag = 1e5; 
  
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
  lr_mat.init_RHS(rhs);
  
  // A = U U^T + diag and U is a circulant matrix
  lr_mat.init_circulant_matrix(diag); 

    

  FastSolver fs(ctx, runtime);
  fs.recLU_solve(lr_mat);

  
  clock_t t1 = clock();
  printf("Init Time: %f.\n", (double)(t1-t0)/CLOCKS_PER_SEC);


  /*
  double *Soln = (double *) malloc(N*rhs_cols*sizeof(double));
  lr_mat.get_soln_from_region(Soln);


  dirct_circulant_solve(Soln, rhs, rhs_rows, rhs_cols, r, diag);
  free(Soln); Soln = NULL;
  */
  

  /*
  //lr_mat.save_solution("solution.txt");
  Eigen::MatrixXd soln(N, rhs_cols);
  for (int i=0; i<N; i++)
    soln(i, 0) = Soln[i];

  Eigen::MatrixXd U(N, r);

  for (int j=0; j<r; j++) {
    for (int i=0; i<N; i++) {
      U(i, j) = (i+j) % r;
    }
  }
  
  Eigen::MatrixXd Diag = diag * Eigen::MatrixXd::Identity(N, N);
  Eigen::MatrixXd A = U * U.transpose() + Diag;
  Eigen::MatrixXd soln_ref = A.colPivHouseholderQr().solve(RHS);

  Eigen::MatrixXd err = soln_ref - soln;
  std::cout << "Error: " << err.norm() / soln_ref.norm() << std::endl;
*/
  //saveVectorToText("solution_ref.txt", soln_ref);


  free(rhs); rhs = NULL;

  return;
}


int main(int argc, char *argv[]) {

  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
							 Processor::LOC_PROC, true, false);

  register_solver_task();
  register_gemm_task();
  register_save_task();
  
  HighLevelRuntime::set_registration_callback(mapper_registration);

 
  return HighLevelRuntime::start(argc, argv);
}


/*
void test_assemble_matrix(Context ctx, HighLevelRuntime *runtime) {


  clock_t t0 = clock();


  int minTol        = -5;
  int maxTol        = -10;
  int numPts        =  1000;
  int intervalMin   = -1;
  int intervalMax   =  1;
  int sizeThreshold =  30;
  double diagValue  =  0.0;
  

  Eigen::MatrixXd denseMatrix = makeMatrix1DUniformPts (intervalMin, intervalMax, intervalMin, intervalMax, numPts, numPts, diagValue, inverseMultiQuadraticKernel);
  //Eigen::VectorXd exact1 = Eigen::VectorXd::LinSpaced(Eigen::Sequential,numPts,-2,2);
  srand( time(NULL) );
  Eigen::VectorXd exactSoln = Eigen::VectorXd::Random(numPts);
  

  Eigen::VectorXd RHS = denseMatrix * exactSoln;
  HODLR_Matrix denseHODLR(denseMatrix, sizeThreshold);

  double tol = pow(10,minTol);
  denseHODLR.set_LRTolerance(tol);
  Eigen::VectorXd solverSoln = denseHODLR.recLU_Solve(RHS);
  saveVectorToText("solution_ref.txt", solverSoln);


  //--- create legion tree root

  LR_Matrix lr_mat(4, ctx, runtime);
  lr_mat.initialize(denseHODLR.indexTree.rootNode, RHS);

  clock_t t1 = clock();
  printf("Init Time: %f.\n", (double)(t1-t0)/CLOCKS_PER_SEC);


  
  FastSolver fs(ctx, runtime);  
  fs.recLU_solve(lr_mat.uroot, lr_mat.vroot);

  lr_mat.save_solution("solution.txt");
}
*/
