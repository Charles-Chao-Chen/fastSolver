#ifndef __DIRECT_SOLVE
#define __DIRECT_SOLVE

void dirct_circulant_solve(double *soln, double *rhs, int rhs_rows, int rhs_cols, int r, double diag);

void dirct_circulant_solve(double *soln, int rand_seed, int rhs_rows,
			   int nregions, int rhs_cols, int r, double
			   diag);

void dirct_circulant_solve(std::string soln_file, int rand_seed, int rhs_rows,
			   int nregions, int rhs_cols, int r, double
			   diag);


#endif
