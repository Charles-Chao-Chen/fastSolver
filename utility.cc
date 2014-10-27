#include <assert.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include "utility.h"


void dirct_circulant_solve(double *soln, double *rhs, int rhs_rows, int rhs_cols, int r, double diag) {

  double *U = (double *) malloc(rhs_rows*r*sizeof(double));
  for (int j=0; j<r; j++)
    for (int i=0; i<rhs_rows; i++)
      U[i+j*rhs_rows] = (i+j)%r;

  double *A = (double *) calloc(rhs_rows*rhs_rows, sizeof(double));
  for (int i=0; i<rhs_rows; i++)
    A[i*(rhs_rows+1)] = diag;

  char transa = 'n';
  char transb = 't';
  int  m = rhs_rows;
  int  n = rhs_rows;
  int  k = r;
  int  lda = rhs_rows;
  int  ldb = rhs_rows;
  int  ldc = rhs_rows;
  double alpha = 1.0;
  double beta  = 1.0;
  
  blas::dgemm_(&transa, &transb, &m, &n, &k, &alpha, U, &lda, U, &ldb, &beta, A, &ldc);

  int INFO;
  int IPIV[m];
  lapack::dgesv_(&m, &rhs_cols, A, &lda, IPIV, rhs, &lda, &INFO);
  assert(INFO == 0);

  double diff  = 0;
  double denom = 0;
  for (int j=0; j<rhs_cols; j++)
    for (int i=0; i<rhs_rows; i++) {
      diff += (soln[i+j*rhs_rows] - rhs[i+j*rhs_rows]) *(soln[i+j*rhs_rows] - rhs[i+j*rhs_rows]);
      denom += rhs[i+j*rhs_rows] * rhs[i+j*rhs_rows];
    }

  std::cout << "Err: " << sqrt(diff/denom) << std::endl;
  
  free(U);
  free(A);
}


void dirct_circulant_solve(double *soln, int rand_seed, int rhs_rows,
			   int nregions, int rhs_cols, int r, double diag) {

  double *rhs = (double *) malloc(rhs_rows*sizeof(double));
  int block_size = rhs_rows/nregions;
  for (int i=0; i<nregions; i++) {
    srand( rand_seed );
    for (int j=0; j<block_size; j++)
      rhs[i*block_size+j] = frand(0, 1);
  }
  
  double *U = (double *) malloc(rhs_rows*r*sizeof(double));
  for (int j=0; j<r; j++)
    for (int i=0; i<rhs_rows; i++)
      U[i+j*rhs_rows] = (i+j)%r;

  double *A = (double *) calloc(rhs_rows*rhs_rows, sizeof(double));
  for (int i=0; i<rhs_rows; i++)
    A[i*(rhs_rows+1)] = diag;

  char transa = 'n';
  char transb = 't';
  int  m = rhs_rows;
  int  n = rhs_rows;
  int  k = r;
  int  lda = rhs_rows;
  int  ldb = rhs_rows;
  int  ldc = rhs_rows;
  double alpha = 1.0;
  double beta  = 1.0;
  
  blas::dgemm_(&transa, &transb, &m, &n, &k, &alpha, U, &lda, U, &ldb, &beta, A, &ldc);

  int INFO;
  int IPIV[m];
  lapack::dgesv_(&m, &rhs_cols, A, &lda, IPIV, rhs, &lda, &INFO);
  assert(INFO == 0);

  double diff  = 0;
  double denom = 0;
  for (int j=0; j<rhs_cols; j++)
    for (int i=0; i<rhs_rows; i++) {
      diff += (soln[i+j*rhs_rows] - rhs[i+j*rhs_rows]) *(soln[i+j*rhs_rows] - rhs[i+j*rhs_rows]);
      denom += rhs[i+j*rhs_rows] * rhs[i+j*rhs_rows];
    }

  std::cout << "Err: " << sqrt(diff/denom) << std::endl;

  free(rhs);
  free(U);
  free(A);
}


void dirct_circulant_solve(std::string soln_file, int rand_seed, int rhs_rows,
			   int nregions, int rhs_cols, int r, double diag) {

  double *rhs = (double *) malloc(rhs_rows*sizeof(double));
  int block_size = rhs_rows/nregions;
  for (int i=0; i<nregions; i++) {
    srand( rand_seed );
    for (int j=0; j<block_size; j++)
      rhs[i*block_size+j] = frand(0, 1);
  }
  
  double *U = (double *) malloc(rhs_rows*r*sizeof(double));
  for (int j=0; j<r; j++)
    for (int i=0; i<rhs_rows; i++)
      U[i+j*rhs_rows] = (i+j)%r;

  double *A = (double *) calloc(rhs_rows*rhs_rows, sizeof(double));
  for (int i=0; i<rhs_rows; i++)
    A[i*(rhs_rows+1)] = diag;

  char transa = 'n';
  char transb = 't';
  int  m = rhs_rows;
  int  n = rhs_rows;
  int  k = r;
  int  lda = rhs_rows;
  int  ldb = rhs_rows;
  int  ldc = rhs_rows;
  double alpha = 1.0;
  double beta  = 1.0;
  
  blas::dgemm_(&transa, &transb, &m, &n, &k, &alpha, U, &lda, U, &ldb, &beta, A, &ldc);

  int INFO;
  int IPIV[m];
  lapack::dgesv_(&m, &rhs_cols, A, &lda, IPIV, rhs, &lda, &INFO);
  assert(INFO == 0);


  // write the direct output to file
  std::ofstream ofs("solution_ref.txt");
  for (int i=0; i<rhs_rows; i++)
    ofs << rhs[i] << std::endl;
  ofs.close();


  
  // read solver output from file
  double *soln = (double *) malloc(rhs_rows*sizeof(double));
  std::ifstream ifs(soln_file.c_str());
  for (int i=0; i<rhs_rows; i++)
    ifs >> soln[i];
  ifs.close();
  
  double diff  = 0;
  double denom = 0;
  for (int j=0; j<rhs_cols; j++)
    for (int i=0; i<rhs_rows; i++) {
      diff += (soln[i+j*rhs_rows] - rhs[i+j*rhs_rows]) *(soln[i+j*rhs_rows] - rhs[i+j*rhs_rows]);
      denom += rhs[i+j*rhs_rows] * rhs[i+j*rhs_rows];
    }

  std::cout << "Err: " << sqrt(diff/denom) << std::endl;

  free(rhs);
  free(soln);
  free(U);
  free(A);
}
