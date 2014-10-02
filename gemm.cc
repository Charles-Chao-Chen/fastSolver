#include "gemm.h"


void register_gemm_task() {

  HighLevelRuntime::register_legion_task<gemm_task>(GEMM_TASK_ID, Processor::LOC_PROC, true, true, AUTO_GENERATE_ID, TaskConfigOptions(true/*leaf*/), "gemm1");
  HighLevelRuntime::register_legion_task<gemm2_task>(GEMM2_TASK_ID, Processor::LOC_PROC, true, true, AUTO_GENERATE_ID, TaskConfigOptions(true/*leaf*/), "gemm2");

  HighLevelRuntime::register_reduction_op<EntrySum>(REDUCE_ID);
}


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
  
  double *v_ptr = acc_v.raw_rect_ptr<2>(rect_v, subrect, offsets);
  assert(rect_v == subrect);

  double *u_ptr = acc_u.raw_rect_ptr<2>(rect_u, subrect, offsets);
  assert(rect_u == subrect);
  
  double *w_ptr = acc_w.raw_rect_ptr<2>(rect_w, subrect, offsets);
  assert(rect_w == subrect);

  char transa = 't';
  char transb = 'n';
  int  m = rect_v.dim_size(0);
  int  n = u_ncol;
  int  k = rect_v.dim_size(1);
  assert(k == rect_u.dim_size(1));
  assert(m == rect_w.dim_size(1));
  assert(n == rect_w.dim_size(0));

  
  double beta = 1.0;
  double * u  = u_ptr + u_col_beg * rect_u.dim_size(1);
  blas::dgemm_(&transa, &transb, &m, &n, &k, &alpha, v_ptr, &k, u, &k, &beta, w_ptr, &m);
}


void gemm_recursive(double alpha, FSTreeNode * v, FSTreeNode * u, int col_beg, int ncol, LogicalRegion & res, Context ctx, HighLevelRuntime * runtime) {
    
  // assume that u and v have the same tree structure
  // (down to Legion leaf level)
  if (v->isLegionLeaf == true) {

    assert(u->isLegionLeaf == true);


    gemmArg arg = {alpha, col_beg, ncol};
    TaskLauncher gemm_task(GEMM_TASK_ID, TaskArgument(&arg, sizeof(gemmArg)));

    assert(v->matrix->data != LogicalRegion::NO_REGION);
    assert(u->matrix->data != LogicalRegion::NO_REGION);
    assert(res             != LogicalRegion::NO_REGION);
    gemm_task.add_region_requirement(RegionRequirement(v->matrix->data, READ_ONLY, EXCLUSIVE,    v->matrix->data)); // v
    gemm_task.add_region_requirement(RegionRequirement(u->matrix->data, READ_ONLY, EXCLUSIVE,    u->matrix->data)); // u
    gemm_task.add_region_requirement(RegionRequirement(res,             REDUCE_ID, SIMULTANEOUS, res));             // res
    gemm_task.region_requirements[0].add_field(FID_X);
    gemm_task.region_requirements[1].add_field(FID_X);
    gemm_task.region_requirements[2].add_field(FID_X);

    runtime->execute_task(ctx, gemm_task);

  } else {
    gemm_recursive(alpha, v->lchild, u->lchild, col_beg, ncol, res, ctx, runtime);
    gemm_recursive(alpha, v->rchild, u->rchild, col_beg, ncol, res, ctx, runtime);
  }
}


void gemm(double alpha, FSTreeNode *v, FSTreeNode *u, range ru, double beta, LogicalRegion & res,
	  Context ctx, HighLevelRuntime *runtime) {

  // create and initialize the result region
  if (res == LogicalRegion::NO_REGION) {

    int nrow = v->ncol;
    int ncol = ru.ncol;
    assert(v->nrow == u->nrow);
    create_matrix(res, nrow, ncol, ctx, runtime);
    set_element(0.0, res, ctx, runtime);
    
  } else scale_matrix(beta, res, ctx, runtime);

  gemm_recursive(alpha, v, u, ru.col_beg, ru.ncol, res, ctx, runtime);
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


  int u_rows = rect_u.dim_size(1);
  int u_cols = u_ncol;
  int v_rows = rect_v.dim_size(1);
  int v_cols = rect_v.dim_size(0);
  int d_rows = rect_u.dim_size(1);
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

/*
void test_gemm(Context ctx, HighLevelRuntime *runtime) {

  LeafData V1(13, 6, ctx, runtime), V2(11, 6, ctx, runtime);
  V1.set_matrix(1.0, ctx, runtime);
  V2.set_matrix(1.0, ctx, runtime);

  FSTreeNode V, V_lchild, V_rchild; // 24 x 6
  V.lchild = & V_lchild;
  V.rchild = & V_rchild;  
  V_lchild.data = & V1;
  V_rchild.data = & V2;
  V_lchild.lchild = NULL;
  V_lchild.rchild = NULL;
  V_rchild.lchild = NULL;
  V_rchild.rchild = NULL; 

  
  LeafData U1(13, 5, ctx, runtime), U2(11, 5, ctx, runtime);
  U1.set_matrix(1.0, ctx, runtime);
  U2.set_matrix(1.0, ctx, runtime);

  FSTreeNode U, U_lchild, U_rchild; // 24 x 5
  U.lchild = & U_lchild;
  U.rchild = & U_rchild;  
  U_lchild.data = & U1;
  U_rchild.data = & U2;
  U_lchild.lchild = NULL;
  U_lchild.rchild = NULL;
  U_rchild.lchild = NULL; 
  U_rchild.rchild = NULL;

  LogicalRegion res;
  gemm(1.0, &V, ALL, &U, ALL, 0.0, res, ctx, runtime);


  print_matrix(res, ctx, runtime);
}
*/


/*    
// assume that v and u have the same tree structure
void add_leaf_matrix(std::vector<RegionRequirement> & req, FSTreeNode * v, FSTreeNode * u) {

  if (v->lchild == NULL && v->rchild == NULL) { // real leaf
    LogicalRegion & V = v->data->matrix;
    LogicalRegion & U = u->data->matrix;
    req.push_back(RegionRequirement(V, READ_ONLY, EXCLUSIVE, V));
    req.push_back(RegionRequirement(U, READ_ONLY, EXCLUSIVE, U));
  } else {
    add_leaf_matrix(req, v->lchild, u->lchild);
    add_leaf_matrix(req, v->rchild, u->rchild);
  }
}
*/


