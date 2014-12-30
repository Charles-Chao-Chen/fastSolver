#ifndef __GEMM_H__
#define __GEMM_H__


#include "legion.h"
#include "Htree.h"

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;


enum {
  GEMM_TASK_ID = 2,
  GEMM2_TASK_ID = 4,
};


enum {
  REDUCE_ID = 1,
};

struct gemmArg {
  double alpha;
  int col_beg;
  int ncol;
};

struct gemm2Arg {
  double alpha;
  double beta;
  int u_col_beg;
  int u_ncol;
  int d_col_beg;
  int d_ncol;
};


void register_gemm_task();


// Reduction Op
class EntrySum {
	
 public:

  typedef double LHS;
  typedef double RHS;
  static const double identity;

  template <bool EXCLUSIVE> static void apply(LHS &lhs, RHS rhs);

  template <bool EXCLUSIVE> static void fold(RHS &rhs1, RHS rhs2);
};


/*
struct range {
  int lb, ub; // Lower and upper bounds of region
range(int lb_, int ub_) : lb(lb_), ub(ub_) {};
};
*/

// These are the two types of GEMM that are needed.

// This GEMM requires a reduction.
//void gemm(double alpha, FSTreeNode * v, range rv, FSTreeNode * u, range ru, double beta,
//	  LogicalRegion & res, Context ctx, HighLevelRuntime *runtime);
// Returns res = alpha * v(rv)^T * u(ru) + beta * R;
// rv is always ALL in practice.
// Number of columns in res must match ru.
// Note the transpose on v.

void gemm_recursive(double, FSTreeNode *, FSTreeNode *, int, int, LogicalRegion &, Context, HighLevelRuntime *);

void gemm_recursive(double alpha, FSTreeNode * v, FSTreeNode * u, int
		    col_beg, int ncol, LogicalRegion & res, Range tag,
		    Context ctx, HighLevelRuntime * runtime);

  
void gemm(double, FSTreeNode *, FSTreeNode *, range, double, LogicalRegion &, Context, HighLevelRuntime *);

void gemm(double alpha, FSTreeNode *v, FSTreeNode *u, range ru, double
	  beta, LogicalRegion & res, Range task_tag,
	  Context ctx, HighLevelRuntime *runtime);

  
void gemm2(double, FSTreeNode *, range, LogicalRegion &, double, FSTreeNode *, range, Context, HighLevelRuntime *);

void gemm2(double alpha, FSTreeNode * u, range ru, LogicalRegion &
	   eta, double beta, FSTreeNode * v, range rv, Range tag,
	   Context ctx, HighLevelRuntime *runtime);

  
void gemm_task(const Task *task, const std::vector<PhysicalRegion> &regions,
	       Context ctx, HighLevelRuntime *runtime);

void gemm2_task(const Task *task, const std::vector<PhysicalRegion> &regions,
	       Context ctx, HighLevelRuntime *runtime);




#endif // _GEMM_H
