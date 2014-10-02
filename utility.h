#include <stdlib.h>

/* Uniform random number generator */
#define frand(xmin,xmax) ((double)xmin+(double)(xmax-xmin)*rand()/ \
			  (double)RAND_MAX) 

void dirct_circulant_solve(double *soln, double *rhs, int rhs_rows, int rhs_cols, int r, double diag);


  
namespace blas {
  extern "C" {
    // Declaration for BLAS matrix-vector multiply
    // note op(A) is m x k and op(B) is k x n, so C is m x n
    void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha,
		double *A, int *lda, double *B, int *ldb, double *beta,
		double *C, int *ldc);
  }
}

  
namespace lapack {
  extern "C" {
    // Declaration for lapack LU solve routine
    void dgesv_(int *N, int *NRHS, double *A, int *LDA, int *IPIV, double *B, int *LDB, int *INFO);
  }
}


