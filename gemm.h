#ifndef _GEMM_H
#define _GEMM_H

#include "legion.h"
#include "Htree.h"


void register_gemm_tasks();


// Reduction Op
class EntrySum {
	
 public:

  typedef double LHS;
  typedef double RHS;
  static const double identity;

  template <bool EXCLUSIVE> static void apply(LHS &lhs, RHS rhs);

  template <bool EXCLUSIVE> static void fold(RHS &rhs1, RHS rhs2);
};


// These are the two types of GEMM that are needed.

// This GEMM requires a reduction.
//void gemm(double alpha, FSTreeNode * v, range rv, FSTreeNode * u, range ru, double beta,
//	  LogicalRegion & res, Context ctx, HighLevelRuntime *runtime);
// Returns res = alpha * v(rv)^T * u(ru) + beta * R;
// rv is always ALL in practice.
// Number of columns in res must match ru.
// Note the transpose on v.

void gemm_recursive(double alpha, FSTreeNode * v, FSTreeNode * u, int
		    col_beg, int ncol, LogicalRegion & res, Range tag,
		    Context ctx, HighLevelRuntime * runtime);

void gemm(double alpha, FSTreeNode *v, FSTreeNode *u, range ru, double
	  beta, LogicalRegion & res, Range task_tag,
	  Context ctx, HighLevelRuntime *runtime);

void gemm2(double alpha, FSTreeNode * u, range ru, LogicalRegion &
	   eta, double beta, FSTreeNode * v, range rv, Range tag,
	   Context ctx, HighLevelRuntime *runtime);

void zero_matrix(LogicalRegion &matrix, Range tag, Context ctx,
		 HighLevelRuntime *runtime);



class ZeroMatrixTask : public TaskLauncher {

 public:
  ZeroMatrixTask(TaskArgument arg,
		 Predicate pred = Predicate::TRUE_PRED,
		 MapperID id = 0,
		 MappingTagID tag = 0);
  
  static int TASKID;

  static void register_tasks(void);

public:
  static void cpu_task(const Task *task,
		       const std::vector<PhysicalRegion> &regions,
		       Context ctx, HighLevelRuntime *runtime);
};


class GEMM_Reduce_Task : public TaskLauncher {

 public:
  struct TaskArgs {
    double alpha;
    int col_beg;
    int ncol;
  };

  GEMM_Reduce_Task(TaskArgument arg,
		   Predicate pred = Predicate::TRUE_PRED,
		   MapperID id = 0,
		   MappingTagID tag = 0);
  
  static int TASKID;

  static void register_tasks(void);

public:
  static void cpu_task(const Task *task,
		       const std::vector<PhysicalRegion> &regions,
		       Context ctx, HighLevelRuntime *runtime);
};


class GEMM_Broadcast_Task : public TaskLauncher {

 public:
  struct TaskArgs {
    double alpha;
    double beta;
    int u_col_beg;
    int u_ncol;
    int d_col_beg;
    int d_ncol;
  };

  GEMM_Broadcast_Task(TaskArgument arg,
		      Predicate pred = Predicate::TRUE_PRED,
		      MapperID id = 0,
		      MappingTagID tag = 0);
  
  static int TASKID;

  static void register_tasks(void);

public:
  static void cpu_task(const Task *task,
		       const std::vector<PhysicalRegion> &regions,
		       Context ctx, HighLevelRuntime *runtime);
};



#endif // _GEMM_H
