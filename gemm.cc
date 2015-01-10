#include "gemm.h"
#include "zero_matrix_task.h"
#include "htree_helper.h"
#include "lapack_blas.h"
#include "macros.h"

enum {
  REDUCE_ID = 1,
};

using namespace LegionRuntime::Accessor;


// TODO: to be moved into LMatrix class
static void
zero_matrix(LogicalRegion &matrix, Range tag,
	    Context ctx, HighLevelRuntime *runtime) {
  
  assert(matrix != LogicalRegion::NO_REGION);
  ZeroMatrixTask launcher(TaskArgument(NULL, 0),
			  Predicate::TRUE_PRED,
			  0,
			  tag.begin);
  launcher.add_region_requirement(
	     RegionRequirement(matrix,
			       WRITE_DISCARD,
			       EXCLUSIVE,
			       matrix));
  launcher.region_requirements[0].add_field(FID_X);
  runtime->execute_task(ctx, launcher);
}


static void
scale_matrix(double beta, LogicalRegion &matrix,
	     Context ctx, HighLevelRuntime *runtime) {
  assert(false);
}



namespace {
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
}


namespace {

  // Reduction Op
  class EntrySum {
	
  public:
    typedef double LHS;
    typedef double RHS;
    static const double identity;

  public:
    template <bool EXCLUSIVE> static void apply(LHS &lhs, RHS rhs);
    template <bool EXCLUSIVE> static void fold(RHS &rhs1, RHS rhs2);
  };

  
  /* ---- reduction class implementation ---- */

  const double EntrySum::identity = 0.0;

  template<>
  void EntrySum::apply<true>(LHS &lhs, RHS rhs)
  {
    lhs += rhs;
  }

  template<>
  void EntrySum::apply<false>(LHS &lhs, RHS rhs)
  {
    int64_t *target = (int64_t *)&lhs;
    union { int64_t as_int; double as_T; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_T = oldval.as_T + rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
  }

  template<>
  void EntrySum::fold<true>(RHS &rhs1, RHS rhs2)
  {
    rhs1 += rhs2;
  }

  template<>
  void EntrySum::fold<false>(RHS &rhs1, RHS rhs2)
  {
    int64_t *target = (int64_t *)&rhs1;
    union { int64_t as_int; double as_T; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_T = oldval.as_T + rhs2;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
  }
}


void register_gemm_tasks() {

  HighLevelRuntime   ::register_reduction_op<EntrySum>(REDUCE_ID);
  GEMM_Reduce_Task   ::register_tasks();
  GEMM_Broadcast_Task::register_tasks();
}


static void gemm_recursive
(double alpha, FSTreeNode * v, FSTreeNode * u,
 int col_beg, int ncol, LMatrix *(&result),
 const Range task_tag, Context ctx,
 HighLevelRuntime * runtime) {
    
  // assume that u and v have the same tree structure
  // (down to Legion leaf level)
  if (     v->is_legion_leaf() ) {
    assert(u->is_legion_leaf() );
    assert(v->lowrank_matrix->data != LogicalRegion::NO_REGION);
    assert(u->lowrank_matrix->data != LogicalRegion::NO_REGION);
    assert(result->data            != LogicalRegion::NO_REGION);

    GEMM_Reduce_Task::TaskArgs args = {alpha, col_beg, ncol};    
    GEMM_Reduce_Task launcher(TaskArgument(
					   &args,
					   sizeof(args)),
			      Predicate::TRUE_PRED,
			      0,
			      task_tag.begin);
    
    launcher.add_region_requirement(
      RegionRequirement(v->lowrank_matrix->data,
			READ_ONLY,
			EXCLUSIVE,
			v->lowrank_matrix->data)); // v
    launcher.add_region_requirement(
      RegionRequirement(u->lowrank_matrix->data,
			READ_ONLY,
			EXCLUSIVE,
			u->lowrank_matrix->data)); // u
    launcher.add_region_requirement(
      RegionRequirement(result->data,
			REDUCE_ID,
			SIMULTANEOUS,
			result->data));            // result
    launcher.region_requirements[0].add_field(FID_X);
    launcher.region_requirements[1].add_field(FID_X);
    launcher.region_requirements[2].add_field(FID_X);
    runtime->execute_task(ctx, launcher);
      
  } else {

    const Range tag0 = task_tag.lchild();
    const Range tag1 = task_tag.rchild();
    gemm_recursive(alpha, v->lchild, u->lchild, col_beg, ncol,
		   result, tag0, ctx, runtime);
    gemm_recursive(alpha, v->rchild, u->rchild, col_beg, ncol,
		   result, tag1, ctx, runtime);
  }
}


void
gemm_reduce(double alpha, FSTreeNode *v, FSTreeNode *u, range ru,
	    double beta, LMatrix *(&result), Range taskTag,
	    Context ctx, HighLevelRuntime *runtime) {

  // create and initialize the result
  if (result == 0) {
    
    int nrow = v->ncol;
    int ncol = ru.ncol;
    assert(v->nrow == u->nrow);
    create_matrix(result, nrow, ncol, ctx, runtime);
    result->zero(taskTag, ctx, runtime);

    //zero_matrix(res, task_tag, ctx, runtime);
    
  } else scale_matrix(beta, result->data, ctx, runtime);

  gemm_recursive(alpha, v, u, ru.col_beg, ru.ncol, result,
		 taskTag, ctx, runtime);
}


// d = beta * d + alpha* u * eta 
void gemm_broadcast
(double alpha, FSTreeNode * u, range ru,
 LMatrix *(&eta),
 double beta, FSTreeNode * v, range rv,
 const Range tag,
 Context ctx, HighLevelRuntime *runtime) {

  if (      u->is_legion_leaf() ) { 
    assert( v->is_legion_leaf() );    
    assert( u->lowrank_matrix->data == v->lowrank_matrix->data);
    
    GEMM_Broadcast_Task::TaskArgs
      args = {alpha, beta,
	      ru.col_beg, ru.ncol,
	      rv.col_beg, rv.ncol};
    GEMM_Broadcast_Task launcher(TaskArgument(
				   &args,
				   sizeof(args)),
				 Predicate::TRUE_PRED,
				 0,
				 tag.begin);

    launcher.add_region_requirement(
               RegionRequirement(u->lowrank_matrix->data,
				 READ_WRITE,
				 EXCLUSIVE,
				 u->lowrank_matrix->data));
    launcher.add_region_requirement(
               RegionRequirement(eta->data,
				 READ_ONLY,
				 EXCLUSIVE,
				 eta->data)); // eta
    launcher.region_requirements[0].add_field(FID_X);
    launcher.region_requirements[1].add_field(FID_X);
    runtime->execute_task(ctx, launcher);
    
  } else {

    const Range tag0 = tag.lchild();
    const Range tag1 = tag.rchild();
    gemm_broadcast(alpha, u->lchild, ru, eta, beta, v->lchild, rv,
		   tag0, ctx, runtime);
    gemm_broadcast(alpha, u->rchild, ru, eta, beta, v->rchild, rv,
		   tag1, ctx, runtime);
  }  
}


/* ---- gemm_reduce implementation ---- */

/*static*/
int GEMM_Reduce_Task::TASKID;

GEMM_Reduce_Task::GEMM_Reduce_Task(
  TaskArgument arg,
  Predicate pred /*= Predicate::TRUE_PRED*/,
  MapperID id /*= 0*/,
  MappingTagID tag /*= 0*/)
  : TaskLauncher(TASKID, arg, pred, id, tag) {}

/*static*/
void GEMM_Reduce_Task::register_tasks(void)
{
  TASKID =
  HighLevelRuntime::register_legion_task<GEMM_Reduce_Task::cpu_task>(
    AUTO_GENERATE_ID,
    Processor::LOC_PROC, 
    true,
    true,
    AUTO_GENERATE_ID,
    TaskConfigOptions(true/*leaf*/),
    "GEMM_Reduce");
  
  printf("Register task %d : GEMM_Reduce\n", TASKID);
}

void
GEMM_Reduce_Task::cpu_task(const Task *task,
			   const std::vector<PhysicalRegion> &regions,
			   Context ctx, HighLevelRuntime *runtime) {
  
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  assert(task->arglen == sizeof(TaskArgs));
  TaskArgs arg = *((TaskArgs*)task->args);
  int      u_ncol    = arg.ncol;
  int      u_col_beg = arg.col_beg;
  double   alpha     = arg.alpha;
  
  IndexSpace is_v = task->regions[0].region.get_index_space();
  IndexSpace is_u = task->regions[1].region.get_index_space();
  IndexSpace is_w = task->regions[2].region.get_index_space();

  Domain dom_v = runtime->get_index_space_domain(ctx, is_v);
  Domain dom_u = runtime->get_index_space_domain(ctx, is_u);
  Domain dom_w = runtime->get_index_space_domain(ctx, is_w);

  Rect<2> rect_v = dom_v.get_rect<2>();
  Rect<2> rect_u = dom_u.get_rect<2>();
  Rect<2> rect_w = dom_w.get_rect<2>();

  RegionAccessor<AccessorType::Generic, double> acc_v =
    regions[0].get_field_accessor(FID_X).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> acc_u =
    regions[1].get_field_accessor(FID_X).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> acc_w =
    regions[2].get_accessor().typeify<double>();
  
  Rect<2> subrect;
  ByteOffset offsets[2];

  double *v_ptr = regions[0].get_field_accessor(FID_X).
    typeify<double>().raw_rect_ptr<2>(rect_v, subrect, offsets);
  assert(rect_v == subrect);
  
  double *u_ptr = regions[1].get_field_accessor(FID_X).
    typeify<double>().raw_rect_ptr<2>(rect_u, subrect, offsets);
  assert(rect_u == subrect);
    
  double *w_ptr = regions[2].get_accessor().typeify<double>().raw_rect_ptr<2>(rect_w, subrect, offsets);
  assert(rect_w == subrect);
  
  char transa = 't';
  char transb = 'n';
  int  m = rect_v.dim_size(1);
  int  n = u_ncol;
  int  k = rect_v.dim_size(0);
  assert(k == rect_u.dim_size(0));
  assert(m == rect_w.dim_size(0));
  assert(n == rect_w.dim_size(1));

  
  double beta = 1.0;
  int u_nrow = rect_u.dim_size(0);
  double * u  = u_ptr + u_col_beg * u_nrow;
  blas::dgemm_(&transa, &transb, &m, &n, &k, &alpha, v_ptr, &k, u, &k, &beta, w_ptr, &m);
}


/* ---- gemm_broadcast implementation ---- */

/*static*/
int GEMM_Broadcast_Task::TASKID;

GEMM_Broadcast_Task::GEMM_Broadcast_Task(
  TaskArgument arg,
  Predicate pred /*= Predicate::TRUE_PRED*/,
  MapperID id /*= 0*/,
  MappingTagID tag /*= 0*/)
  : TaskLauncher(TASKID, arg, pred, id, tag) {}

/*static*/
void GEMM_Broadcast_Task::register_tasks(void)
{
  TASKID =
    HighLevelRuntime::register_legion_task<GEMM_Broadcast_Task::cpu_task>(
    AUTO_GENERATE_ID,
    Processor::LOC_PROC, 
    true,
    true,
    AUTO_GENERATE_ID,
    TaskConfigOptions(true/*leaf*/),
    "GEMM_Broadcast");
  
  printf("Register task %d : GEMM_Broadcast\n", TASKID);
}

void
GEMM_Broadcast_Task::cpu_task(const Task *task,
			   const std::vector<PhysicalRegion> &regions,
			   Context ctx, HighLevelRuntime *runtime) {

  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->arglen == sizeof(TaskArgs));
  TaskArgs arg = *((TaskArgs*)task->args);
  int      u_ncol    = arg.u_ncol;
  int      d_ncol    = arg.d_ncol;
  int      u_col_beg = arg.u_col_beg;
  int      d_col_beg = arg.d_col_beg;
  double   alpha     = arg.alpha;
  double   beta      = arg.beta;
  
  IndexSpace is_u = task->regions[0].region.get_index_space();
  IndexSpace is_v = task->regions[1].region.get_index_space();

  Domain dom_u = runtime->get_index_space_domain(ctx, is_v);
  Domain dom_v = runtime->get_index_space_domain(ctx, is_u);

  Rect<2> rect_u = dom_v.get_rect<2>();
  Rect<2> rect_v = dom_u.get_rect<2>();

  RegionAccessor<AccessorType::Generic, double> acc_u =
    regions[0].get_field_accessor(FID_X).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> acc_v =
    regions[1].get_field_accessor(FID_X).typeify<double>();
  
  Rect<2> subrect;
  ByteOffset offsets[2];

  double *u_ptr = acc_u.raw_rect_ptr<2>(rect_u, subrect, offsets);
  assert(rect_u == subrect);
  
  double *v_ptr = acc_v.raw_rect_ptr<2>(rect_v, subrect, offsets);
  assert(rect_v == subrect);


  int u_rows = rect_u.dim_size(0);
  int u_cols = u_ncol;
  int v_rows = rect_v.dim_size(0);
  int v_cols = rect_v.dim_size(1);
  int d_rows = rect_u.dim_size(0);
  int d_cols = d_ncol;
  
  char transa = 'n';
  char transb = 'n';
  int  m = u_rows;
  int  n = v_cols;
  int  k = u_cols;
  assert(k == v_rows);
  
  double * u = u_ptr + u_col_beg * u_rows;
  double * d = u_ptr + d_col_beg * d_rows;
  blas::dgemm_(&transa, &transb, &m, &n, &k, &alpha, u, &m, v_ptr, &k, &beta, d, &m);  
}






