#include "gemm.h"
#include "lapack_blas.h"
#include "macros.h"

void register_gemm_tasks() {

  HighLevelRuntime::register_reduction_op<EntrySum>(REDUCE_ID);
  GEMM_Reduce_Task::register_tasks();

    
  HighLevelRuntime::register_legion_task<gemm2_task>(GEMM2_TASK_ID,
						     Processor::LOC_PROC,
						     true,
						     true,
						     AUTO_GENERATE_ID,
						     TaskConfigOptions(true/*leaf*/),
						     "gemm_broadcast");


  HighLevelRuntime::register_legion_task<zero_matrix_task>(ZERO_MATRIX_TASK_ID,
							   Processor::LOC_PROC,
							   true, true,
							   AUTO_GENERATE_ID,
							   TaskConfigOptions(true/*leaf*/),
							   "init_zero_matrix");

}


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




// compute w = v^T u
void gemm_task(const Task *task, const std::vector<PhysicalRegion> &regions,
	       Context ctx, HighLevelRuntime *runtime) {

  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  assert(task->arglen == sizeof(gemmArg));
  gemmArg arg = *((gemmArg*)task->args);
  int     u_ncol    = arg.ncol;
  int     u_col_beg = arg.col_beg;
  double  alpha     = arg.alpha;
  
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

  /*
  double *v_ptr = acc_v.raw_rect_ptr<2>(rect_v, subrect, offsets);
  assert(rect_v == subrect);
  double *u_ptr = acc_u.raw_rect_ptr<2>(rect_u, subrect, offsets);
  assert(rect_u == subrect);  
  double *w_ptr = acc_w.raw_rect_ptr<2>(rect_w, subrect, offsets);
  assert(rect_w == subrect);
  */

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


/*
void gemm_recursive(double alpha, FSTreeNode * v, FSTreeNode * u, int col_beg, int ncol, LogicalRegion & res, Context ctx, HighLevelRuntime * runtime) {
    
  // assume that u and v have the same tree structure
  // (down to Legion leaf level)
  if (v->isLegionLeaf == true) {

    assert(u->isLegionLeaf == true);

    gemmArg arg = {alpha, col_beg, ncol};
    TaskLauncher gemm_task1(GEMM_TASK_ID, TaskArgument(&arg, sizeof(gemmArg)));

    assert(v->matrix->data != LogicalRegion::NO_REGION);
    assert(u->matrix->data != LogicalRegion::NO_REGION);
    assert(res             != LogicalRegion::NO_REGION);
    gemm_task1.add_region_requirement(RegionRequirement(v->matrix->data, READ_ONLY, EXCLUSIVE,    v->matrix->data)); // v
    gemm_task1.add_region_requirement(RegionRequirement(u->matrix->data, READ_ONLY, EXCLUSIVE,    u->matrix->data)); // u
    gemm_task1.add_region_requirement(RegionRequirement(res,             REDUCE_ID, SIMULTANEOUS, res));             // res
    gemm_task1.region_requirements[0].add_field(FID_X);
    gemm_task1.region_requirements[1].add_field(FID_X);
    gemm_task1.region_requirements[2].add_field(FID_X);

    runtime->execute_task(ctx, gemm_task1);

  } else {
    gemm_recursive(alpha, v->lchild, u->lchild, col_beg, ncol, res, ctx, runtime);
    gemm_recursive(alpha, v->rchild, u->rchild, col_beg, ncol, res, ctx, runtime);
  }
}
*/

void gemm_recursive(double alpha,
		    FSTreeNode * v, FSTreeNode * u,
		    int col_beg, int ncol,
		    LogicalRegion & res,
		    Range task_tag,
		    Context ctx,
		    HighLevelRuntime * runtime) {
    
  // assume that u and v have the same tree structure
  // (down to Legion leaf level)
  if (v->isLegionLeaf == true) {

    assert(u->isLegionLeaf == true);
    assert(v->matrix->data != LogicalRegion::NO_REGION);
    assert(u->matrix->data != LogicalRegion::NO_REGION);
    assert(res             != LogicalRegion::NO_REGION);

    gemmArg arg = {alpha, col_beg, ncol};
    GEMM_Reduce_Task launcher(TaskArgument(
				&arg,
				sizeof(gemmArg)),
			      Predicate::TRUE_PRED,
			      0,
			      task_tag.begin);
    
    launcher.add_region_requirement(
      RegionRequirement(v->matrix->data,
			READ_ONLY,
			EXCLUSIVE,
			v->matrix->data)); // v
    launcher.add_region_requirement(
      RegionRequirement(u->matrix->data,
			READ_ONLY,
			EXCLUSIVE,
			u->matrix->data)); // u
    launcher.add_region_requirement(
      RegionRequirement(res,
			REDUCE_ID,
			SIMULTANEOUS,
			res));             // res
    launcher.region_requirements[0].add_field(FID_X);
    launcher.region_requirements[1].add_field(FID_X);
    launcher.region_requirements[2].add_field(FID_X);
    runtime->execute_task(ctx, launcher);


    /*
    TaskLauncher gemm_task1(GEMM_TASK_ID, TaskArgument(&arg,
						       sizeof(gemmArg)),
			    Predicate::TRUE_PRED, 0, tag.begin);

    gemm_task1.add_region_requirement(RegionRequirement(v->matrix->data, READ_ONLY, EXCLUSIVE,    v->matrix->data)); // v
    gemm_task1.add_region_requirement(RegionRequirement(u->matrix->data, READ_ONLY, EXCLUSIVE,    u->matrix->data)); // u
    gemm_task1.add_region_requirement(RegionRequirement(res,             REDUCE_ID, SIMULTANEOUS, res));             // res
    gemm_task1.region_requirements[0].add_field(FID_X);
    gemm_task1.region_requirements[1].add_field(FID_X);
    gemm_task1.region_requirements[2].add_field(FID_X);
    runtime->execute_task(ctx, gemm_task1);
*/
      
  } else {

    int   half = task_tag.size/2;
    Range tag0 = {task_tag.begin,      half};
    Range tag1 = {task_tag.begin+half, half};
    gemm_recursive(alpha, v->lchild, u->lchild, col_beg, ncol, res,
		   tag0, ctx, runtime);
    gemm_recursive(alpha, v->rchild, u->rchild, col_beg, ncol, res,
		   tag1, ctx, runtime);
  }
}

/*
void gemm(double alpha, FSTreeNode *v, FSTreeNode *u, range ru, double beta, LogicalRegion & res,
	  Context ctx, HighLevelRuntime *runtime) {

  // create and initialize the result region
  if (res == LogicalRegion::NO_REGION) {

    int nrow = v->ncol;
    int ncol = ru.ncol;
    assert(v->nrow == u->nrow);
    create_matrix(res, nrow, ncol, ctx, runtime);
    zero_matrix(res, ctx, runtime);
    
  } else scale_matrix(beta, res, ctx, runtime);

  gemm_recursive(alpha, v, u, ru.col_beg, ru.ncol, res, ctx, runtime);
}
*/


void gemm(double alpha, FSTreeNode *v, FSTreeNode *u, range ru, double
	  beta, LogicalRegion & res, Range task_tag,
	  Context ctx, HighLevelRuntime *runtime) {

  // create and initialize the result region
  if (res == LogicalRegion::NO_REGION) {

    int nrow = v->ncol;
    int ncol = ru.ncol;
    assert(v->nrow == u->nrow);
    create_matrix(res, nrow, ncol, ctx, runtime);
    zero_matrix(res, task_tag, ctx, runtime);
    
  } else scale_matrix(beta, res, ctx, runtime);

  gemm_recursive(alpha, v, u, ru.col_beg, ru.ncol, res, task_tag, ctx, runtime);
}


// compute d -= alpha * u * v
void gemm2_task(const Task *task, const std::vector<PhysicalRegion> &regions,
		Context ctx, HighLevelRuntime *runtime) {

  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(task->arglen == sizeof(gemm2Arg));
  gemm2Arg arg = *((gemm2Arg*)task->args);
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


// d = beta * d + alpha* u * eta 
void gemm2(double alpha, FSTreeNode * u, range ru, LogicalRegion & eta, double beta, FSTreeNode * v, range rv, Context ctx, HighLevelRuntime *runtime) {

  if (u->isLegionLeaf == true) {
  
    assert(v->isLegionLeaf == true);

    //save_region(v->matrix->data,                "V.txt", ctx, runtime);
    //save_region(u->matrix->data,                "BigU.txt", ctx, runtime);
    //save_region(u->matrix->data, col_beg, ncol, "U.txt", ctx, runtime);
    
    
    gemm2Arg arg = {alpha, beta, ru.col_beg, ru.ncol, rv.col_beg, rv.ncol};
    TaskLauncher gemm2_task(GEMM2_TASK_ID, TaskArgument(&arg, sizeof(gemm2Arg)));

    assert(u->matrix->data == v->matrix->data);
    gemm2_task.add_region_requirement(RegionRequirement(u->matrix->data, READ_WRITE, EXCLUSIVE, u->matrix->data));
    gemm2_task.add_region_requirement(RegionRequirement(eta,             READ_ONLY,  EXCLUSIVE, eta)); // eta
    gemm2_task.region_requirements[0].add_field(FID_X);
    gemm2_task.region_requirements[1].add_field(FID_X);

    runtime->execute_task(ctx, gemm2_task);

  } else {
    gemm2(alpha, u->lchild, ru, eta, beta, v->lchild, rv, ctx, runtime);
    gemm2(alpha, u->rchild, ru, eta, beta, v->rchild, rv, ctx, runtime);
  }  
}



// d = beta * d + alpha* u * eta 
void gemm2(double alpha, FSTreeNode * u, range ru, LogicalRegion &
	   eta, double beta, FSTreeNode * v, range rv, Range tag,
	   Context ctx, HighLevelRuntime *runtime) {

  if (u->isLegionLeaf == true) {
  
    assert(v->isLegionLeaf == true);    
    gemm2Arg arg = {alpha, beta, ru.col_beg, ru.ncol, rv.col_beg, rv.ncol};
    TaskLauncher gemm2_task(GEMM2_TASK_ID, TaskArgument(&arg,
							sizeof(gemm2Arg)),
			    Predicate::TRUE_PRED, 0, tag.begin);

    assert(u->matrix->data == v->matrix->data);
    gemm2_task.add_region_requirement(RegionRequirement(u->matrix->data, READ_WRITE, EXCLUSIVE, u->matrix->data));
    gemm2_task.add_region_requirement(RegionRequirement(eta,             READ_ONLY,  EXCLUSIVE, eta)); // eta
    gemm2_task.region_requirements[0].add_field(FID_X);
    gemm2_task.region_requirements[1].add_field(FID_X);

    runtime->execute_task(ctx, gemm2_task);

  } else {

    int   half = tag.size/2;
    Range tag0 = {tag.begin,      half};
    Range tag1 = {tag.begin+half, half};
    gemm2(alpha, u->lchild, ru, eta, beta, v->lchild, rv, tag0, ctx, runtime);
    gemm2(alpha, u->rchild, ru, eta, beta, v->rchild, rv, tag1, ctx, runtime);
  }  
}



void zero_matrix(LogicalRegion &matrix, Range tag, Context ctx, HighLevelRuntime *runtime) {
  assert(matrix != LogicalRegion::NO_REGION);
  TaskLauncher zero_matrix_task(ZERO_MATRIX_TASK_ID,
				TaskArgument(NULL, 0),
				Predicate::TRUE_PRED, 0, tag.begin);
  zero_matrix_task.add_region_requirement(RegionRequirement(matrix, WRITE_DISCARD, EXCLUSIVE, matrix));
  zero_matrix_task.region_requirements[0].add_field(FID_X);
  runtime->execute_task(ctx, zero_matrix_task);
}


void zero_matrix_task(const Task *task, const std::vector<PhysicalRegion> &regions,
		      Context ctx, HighLevelRuntime *runtime) {

  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->arglen == 0);

  IndexSpace is = task->regions[0].region.get_index_space();
  Domain dom = runtime->get_index_space_domain(ctx, is);
  Rect<2> rect = dom.get_rect<2>();

  Rect<2> subrect;
  ByteOffset offsets[2];

  double *ptr = regions[0].get_field_accessor(FID_X).typeify<double>().raw_rect_ptr<2>(rect, subrect, offsets);
  assert(rect == subrect);
  assert(ptr  != NULL);
  
  int nrow = rect.dim_size(0);
  int ncol = rect.dim_size(1);
  int size = nrow * ncol;

  memset(ptr, 0, size*sizeof(double));
}


void scale_matrix(double beta, LogicalRegion &matrix, Context ctx, HighLevelRuntime *runtime) {

  assert(false);

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
  assert(task->arglen == sizeof(gemmArg));
  gemmArg arg = *((gemmArg*)task->args);
  int     u_ncol    = arg.ncol;
  int     u_col_beg = arg.col_beg;
  double  alpha     = arg.alpha;
  
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
