#ifndef __DIRECT_SOLVE
#define __DIRECT_SOLVE

#include <string>
#include "hodlr_matrix.h"

/*
void dirct_circulant_solve(double *soln, double *rhs, int rhs_rows, int rhs_cols, int r, double diag);

void dirct_circulant_solve(double *soln, int rand_seed, int rhs_rows,
			   int nregions, int rhs_cols, int r, double
			   diag);

void
dirct_circulant_solve(std::string soln_file, int rand_seed,
		      int rhs_rows, int nregions, int rhs_cols,
		      int r, double diag);
*/

void compute_L2_error
(const HodlrMatrix &lr_mat, const long rand_seed, const int rhs_rows,
 const int nregions, const int rhs_cols, const int rank,
 const double diag,
 Context ctx, HighLevelRuntime *runtime);


  
#endif
