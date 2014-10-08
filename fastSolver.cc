#include "fastSolver.h"


#include <algorithm>
#include <assert.h>
#include <iomanip>

void register_solver_task() {
  
  HighLevelRuntime::register_legion_task<leaf_task>(LEAF_TASK_ID,
						    Processor::LOC_PROC,
						    true,
						    true,
						    AUTO_GENERATE_ID,
						    TaskConfigOptions(true/*leaf*/),
						    "leaf_task");
  
  HighLevelRuntime::register_legion_task<lu_solve_task>(LU_SOLVE_TASK_ID, Processor::LOC_PROC, true, true, AUTO_GENERATE_ID, TaskConfigOptions(true/*leaf*/), "lu_task");
}


FastSolver::FastSolver(Context ctx, HighLevelRuntime *runtime) {
  this -> ctx     = ctx;
  this -> runtime = runtime;
}


/*
void FastSolver::initialize() {

}


void FastSolver::recLU_solve() {
  recLU_solve(uroot, vroot);


  // output the result, i.e. the first column
  range ru = {0, 1};
  save_region(uroot, ru, "solution_out.txt", ctx, runtime);

  //save_region(uroot, "solution_out.txt", ctx, runtime);
}
*/


void FastSolver::recLU_solve(LR_Matrix &lr_mat) {
  recLU_solve(lr_mat.uroot, lr_mat.vroot);
}

void FastSolver::recLU_solve(FSTreeNode * unode, FSTreeNode * vnode) {

  if (unode->isLegionLeaf) {

    assert(vnode->isLegionLeaf);
    
    solve_legion_leaf(unode, vnode);

    return;
  }

  FSTreeNode * b0 = unode->lchild;
  FSTreeNode * b1 = unode->rchild;
  recLU_solve(b0, vnode->lchild);
  recLU_solve(b1, vnode->rchild);

  
  FSTreeNode * V0 = vnode->lchild;
  FSTreeNode * V1 = vnode->rchild;


  assert(unode->isLegionLeaf == false);
  assert(V0->Hmat != NULL);
  assert(V1->Hmat != NULL);

  // This involves a reduction for V0Tu0, V0Td0, V1Tu1, V1Td1
  // from leaves to root in the H tree.
  LogicalRegion V0Tu0, V0Td0, V1Tu1, V1Td1;
  range ru0 = {b0->col_beg, b0->ncol};
  range ru1 = {b1->col_beg, b1->ncol};
  range rd0 = {0,           b0->col_beg};
  range rd1 = {0,           b1->col_beg};
  gemm(1., V0->Hmat, b0, ru0, 0., V0Tu0, ctx, runtime);
  gemm(1., V1->Hmat, b1, ru1, 0., V1Tu1, ctx, runtime);
  gemm(1., V0->Hmat, b0, rd0, 0., V0Td0, ctx, runtime);
  gemm(1., V1->Hmat, b1, rd1, 0., V1Td1, ctx, runtime);


  //save_region(V0Tu0, "V0Tu0.txt", ctx, runtime);
  //save_region(V1Td1, "V1Td1.txt", ctx, runtime);
  
  // V0Td0 and V1Td1 contain the solution on output.
  // eta0 = V1Td1
  // eta1 = V0Td0
  solve_node_matrix(V0Tu0, V1Tu1, V0Td0, V1Td1, ctx, runtime);

  //save_region(V0Td0, "eta1.txt", ctx, runtime);
  //save_region(V1Td1, "eta0.txt", ctx, runtime);
  

  // This step requires a broadcast of V0Td0 and V1Td1 from root to leaves.
  // Assemble x from d0 and d1: merge two trees
  gemm2(-1., b0, ru0, V1Td1, 1., b0, rd0, ctx, runtime);
  gemm2(-1., b1, ru1, V0Td0, 1., b1, rd1, ctx, runtime);

  //save_region(b0, rd0, "d0.txt", ctx, runtime);
  //save_region(b1, rd1, "d1.txt", ctx, runtime);


  //save_region(unode, "solution_in.txt", ctx, runtime);
  //save_region(unode, "solution1_in.txt", ctx, runtime);
  //save_region(unode, "solution2_in.txt", ctx, runtime);
  
  
  //range ru = {0, 1};
  //save_region(unode, ru, "solution_in.txt", ctx, runtime);
  //printf("col_beg: %d, ncol: %d.\n", unode->col_beg, unode->ncol);

}


// this function launches leaf tasks
void FastSolver::solve_legion_leaf(FSTreeNode * uleaf, FSTreeNode * vleaf) {

  int nleaf = count_leaf(uleaf);
  //assert(nleaf == nleaf_per_node);
  //int max_tree_size = nleaf_per_node * 2;
  int max_tree_size = nleaf * 2;
  FSTreeNode arg[max_tree_size*2];

  arg[0] = *vleaf;
  int tree_size = tree_to_array(vleaf, arg, 0);
  //std::cout << "Tree size: " << tree_size << std::endl;
  assert(tree_size < max_tree_size);

  arg[max_tree_size] = *uleaf;
  tree_to_array(uleaf, arg, 0, max_tree_size);

  // encode the array size
  arg[0].col_beg = max_tree_size;
    
  TaskLauncher leaf_task(LEAF_TASK_ID, TaskArgument(&arg[0], sizeof(FSTreeNode)*(max_tree_size*2)));

  // u region
  leaf_task.add_region_requirement(RegionRequirement(uleaf->matrix->data, READ_WRITE, EXCLUSIVE, uleaf->matrix->data));
  leaf_task.region_requirements[0].add_field(FID_X);

  // v region
  leaf_task.add_region_requirement(RegionRequirement(vleaf->matrix->data, READ_ONLY,  EXCLUSIVE, vleaf->matrix->data));
  leaf_task.region_requirements[1].add_field(FID_X);

  // k region
  leaf_task.add_region_requirement(RegionRequirement(vleaf->kmat->data,   READ_ONLY,  EXCLUSIVE, vleaf->kmat->data));
  leaf_task.region_requirements[2].add_field(FID_X);

  runtime->execute_task(ctx, leaf_task);
}


void leaf_task(const Task *task, const std::vector<PhysicalRegion> &regions,
	       Context ctx, HighLevelRuntime *runtime) {

  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  FSTreeNode *arg = (FSTreeNode *)task->args;
  
  // decode tree size
  int tree_size = arg[0].col_beg;
  //std::cout << "Tree size: " << tree_size << std::endl;
  arg[0].col_beg = 0;
  assert(task->arglen == sizeof(FSTreeNode)*(tree_size*2));

  FSTreeNode *vroot = arg;
  array_to_tree(arg, 0);

  FSTreeNode *uroot = &arg[tree_size];
  array_to_tree(arg, 0, tree_size);
  
  //print_legion_tree(vroot);
  //print_legion_tree(uroot);
  
   RegionAccessor<AccessorType::Generic, double> acc_u = 
     regions[0].get_field_accessor(FID_X).typeify<double>();
   RegionAccessor<AccessorType::Generic, double> acc_v = 
     regions[1].get_field_accessor(FID_X).typeify<double>();
   RegionAccessor<AccessorType::Generic, double> acc_k = 
     regions[2].get_field_accessor(FID_X).typeify<double>();

   IndexSpace is_u = task->regions[0].region.get_index_space();
   IndexSpace is_v = task->regions[1].region.get_index_space();
   IndexSpace is_k = task->regions[2].region.get_index_space();

   Domain dom_u = runtime->get_index_space_domain(ctx, is_u);
   Domain dom_v = runtime->get_index_space_domain(ctx, is_v);
   Domain dom_k = runtime->get_index_space_domain(ctx, is_k);

   Rect<2> rect_u = dom_u.get_rect<2>();
   Rect<2> rect_v = dom_v.get_rect<2>();
   Rect<2> rect_k = dom_k.get_rect<2>();

   Rect<2> subrect;
   ByteOffset offsets[2];

   double *u_ptr = acc_u.raw_rect_ptr<2>(rect_u, subrect, offsets);
   assert(u_ptr != NULL);
   assert(rect_u == subrect);

   //printf("U size: %d x %d\n", rect_u.dim_size(0), rect_u.dim_size(1));
   //std::cout << "U Offset: " << offsets[0].offset
   //	     << ", "         << offsets[1].offset << std::endl;

   double *v_ptr = NULL;
   if (rect_v.volume() != 0) {
     // if legion leaf coinsides real leaf, no V is needed
     v_ptr = acc_v.raw_rect_ptr<2>(rect_v, subrect, offsets);
     assert(v_ptr != NULL);
     assert(rect_v == subrect);
     //std::cout << "V Offset: " << offsets[0].offset
     //	     << ", "         << offsets[1].offset << std::endl;
   }
   
   double *k_ptr = acc_k.raw_rect_ptr<2>(rect_k, subrect, offsets);
   assert(k_ptr != NULL);
   assert(rect_k == subrect);
   //std::cout << "K Offset: " << offsets[0].offset
   //	     << ", "         << offsets[1].offset << std::endl;

   int l_dim  = offsets[1].offset / sizeof(double);
   int u_nrow = rect_u.dim_size(0);
   assert( l_dim == u_nrow );   
   recLU_leaf_solve(uroot, vroot, u_ptr, v_ptr, k_ptr, l_dim);
}


int FastSolver::tree_to_array(FSTreeNode * leaf, FSTreeNode * arg, int idx) {

  if (leaf->lchild != NULL && leaf->rchild != NULL) {

    //assert(2*idx+2 < arg.size());
    arg[ 2*idx+1 ] = *(leaf -> lchild);
    arg[ 2*idx+2 ] = *(leaf -> rchild);
    int nl = tree_to_array(leaf->lchild, arg, 2*idx+1);
    int nr = tree_to_array(leaf->rchild, arg, 2*idx+2);
    return nl + nr + 1;
    
  } else return 1;
}


//void FastSolver::tree_to_array(FSTreeNode * leaf, std::vector<FSTreeNode> & arg, int idx, int shift) {
void FastSolver::tree_to_array(FSTreeNode * leaf, FSTreeNode * arg, int idx, int shift) {

  if (leaf->lchild != NULL && leaf->rchild != NULL) {

    //assert(2*idx+2+shift < arg.size());
    arg[ 2*idx+1+shift ] = *(leaf -> lchild);
    arg[ 2*idx+2+shift ] = *(leaf -> rchild);
    tree_to_array(leaf->lchild, arg, 2*idx+1, shift);
    tree_to_array(leaf->rchild, arg, 2*idx+2, shift); 
  }
}


void array_to_tree(FSTreeNode *arg, int idx) {

  if (arg[ idx ].lchild != NULL) {
    
    assert(arg[ idx ].rchild != NULL);
    arg[ idx ].lchild = &arg[ 2*idx+1 ];
    arg[ idx ].rchild = &arg[ 2*idx+2 ];
    
  } else {
    assert(arg[ idx ].rchild == NULL);
    return; 
  }
  
  array_to_tree(arg, 2*idx+1);
  array_to_tree(arg, 2*idx+2);
}


void array_to_tree(FSTreeNode *arg, int idx, int shift) {

  if (arg[ idx+shift ].lchild != NULL) {
    
    assert(arg[ idx+shift ].rchild != NULL);
    arg[ idx+shift ].lchild = &arg[ 2*idx+1+shift ];
    arg[ idx+shift ].rchild = &arg[ 2*idx+2+shift ];
    
  } else {
    assert(arg[ idx+shift ].rchild == NULL);
    return;
  }

  array_to_tree(arg, 2*idx+1);
  array_to_tree(arg, 2*idx+2);
}


void recLU_leaf_solve(FSTreeNode * unode, FSTreeNode * vnode, double * u_ptr, double * v_ptr, double * k_ptr, int LD) {

  if (unode->lchild == NULL && unode->rchild == NULL) {
    assert(vnode->lchild == NULL);
    assert(vnode->rchild == NULL);

    assert(unode->nrow == vnode->nrow);
    int N     = unode->nrow;
    int NRHS  = unode->col_beg + unode->ncol;
    int LDA   = LD;
    int LDB   = LD;
    double *A = k_ptr + vnode->row_beg;
    double *B = u_ptr + vnode->row_beg;
      
    int INFO;
    int IPIV[N];
    
    lapack::dgesv_(&N, &NRHS, A, &LDA, IPIV, B, &LDB, &INFO);
    assert(INFO == 0);

    return;
  }

  recLU_leaf_solve(unode->lchild, vnode->lchild, u_ptr, v_ptr, k_ptr, LD);
  recLU_leaf_solve(unode->rchild, vnode->rchild, u_ptr, v_ptr, k_ptr, LD);
  
  char   transa = 't';
  char   transb = 'n';
  double alpha  = 1.0;
  double beta   = 0.0;
  
  int V0_rows = vnode->lchild->nrow;
  int V0_cols = vnode->lchild->ncol;
  int V1_rows = vnode->rchild->nrow;
  int V1_cols = vnode->rchild->ncol;

  int u0_rows = unode->lchild->nrow;
  int u0_cols = unode->lchild->ncol;
  int u1_rows = unode->rchild->nrow;
  int u1_cols = unode->rchild->ncol;
  
  int d0_rows = unode->lchild->nrow;
  int d0_cols = unode->lchild->col_beg;
  int d1_rows = unode->rchild->nrow;
  int d1_cols = unode->rchild->col_beg;
  
  double *V0 = v_ptr + vnode->lchild->row_beg + vnode->lchild->col_beg*LD;
  double *V1 = v_ptr + vnode->rchild->row_beg + vnode->rchild->col_beg*LD;

  double *u0 = u_ptr + unode->lchild->row_beg + unode->lchild->col_beg*LD;
  double *u1 = u_ptr + unode->rchild->row_beg + unode->rchild->col_beg*LD;

  double *d0 = u_ptr + unode->lchild->row_beg;
  double *d1 = u_ptr + unode->rchild->row_beg;


  // Shur complement
  assert(V0_cols + V1_cols == u0_cols + u1_cols);
  int    S_size = V0_cols + V1_cols;
  double *S = (double *) calloc( S_size*S_size, sizeof(double) );

  assert(d0_cols == d1_cols);
  double *S_RHS = (double *) malloc( S_size*d0_cols * sizeof(double) );
  
  // initialize the off-diagonal blocks to identity
  for (int i=0; i<S_size; i++)
    S[ (V0_cols + i)%S_size + i*S_size ] = 1.0;

  assert(V0_rows == u0_rows);
  assert(V1_rows == u1_rows);
  assert(V0_rows == d0_rows);
  assert(V1_rows == d1_rows);

  double *V0Tu0 = S;
  double *V1Tu1 = S + (V0_cols + u0_cols*S_size);
  double *V0Td0 = S_RHS;
  double *V1Td1 = S_RHS + V0_cols;
  
  blas::dgemm_(&transa, &transb, &V0_cols, &u0_cols, &V0_rows, &alpha, V0, &LD, u0, &LD, &beta, V0Tu0, &S_size);
  blas::dgemm_(&transa, &transb, &V1_cols, &u1_cols, &V1_rows, &alpha, V1, &LD, u1, &LD, &beta, V1Tu1, &S_size);
  blas::dgemm_(&transa, &transb, &V0_cols, &d0_cols, &V0_rows, &alpha, V0, &LD, d0, &LD, &beta, V0Td0, &S_size);
  blas::dgemm_(&transa, &transb, &V1_cols, &d1_cols, &V1_rows, &alpha, V1, &LD, d1, &LD, &beta, V1Td1, &S_size);

  
  int INFO;
  int IPIV[S_size];
  assert(d0_cols == d1_cols);

  lapack::dgesv_(&S_size, &d0_cols, S, &S_size, IPIV, S_RHS, &S_size, &INFO);
  assert(INFO == 0);


  //save_matrix(S_RHS, S_size, d1_cols, "S_RHS.txt");  
  transa =  'n';
  alpha  = -1.0;
  beta   =  1.0;
  
  double * eta0 = S_RHS;          
  double * eta1 = S_RHS + V1_cols;

  int eta0_rows = V1_cols;
  int eta0_cols = d0_cols;
  int eta1_rows = V0_cols;
  int eta1_cols = d0_cols;
  
  assert(u0_cols == eta0_rows);
  assert(u1_cols == eta1_rows);
  blas::dgemm_(&transa, &transb, &u0_rows, &eta0_cols, &u0_cols, &alpha, u0, &LD, eta0, &S_size, &beta, d0, &LD);
  blas::dgemm_(&transa, &transb, &u1_rows, &eta1_cols, &u1_cols, &alpha, u1, &LD, eta1, &S_size, &beta, d1, &LD);

  
  //printf("d0 2x2: %f, %f, %f, %f.\n", d0[0], d0[1], d0[LD], d0[LD+1]);
  
  //save_matrix(V0Tu0, V0_cols, u0_cols, "V0Tu0.txt");
  //save_matrix(V1Tu1, V1_cols, u1_cols, "V1Tu1.txt");
  //save_matrix(V0Td0, V0_cols, d0_cols, "V0Td0.txt");
  //save_matrix(V1Td1, V1_cols, d1_cols, "V1Td1.txt");
  //save_matrix(S,     S_size, S_size,  "Shur.txt");
  //save_matrix(S_RHS, S_size, d1_cols, "S_RHS.txt");
  //save_matrix(d0, unode->nrow, unode->col_beg+unode->ncol, LD, "result.txt");

  free(S);
  free(S_RHS);
}


void save_matrix(double *A, int nRows, int nCols, int LD, std::string filename) {

  std::ofstream outputFile(filename.c_str(), std::ios_base::app);
  if (outputFile.is_open()){
    outputFile<<nRows<<std::endl;
    outputFile<<nCols<<std::endl;
    for (int i = 0; i < nRows ;i++) {
      for (int j = 0; j< nCols ;j++) {
	//outputFile<<A[i+j*nRows]<<'\t';
	outputFile<<A[i+j*LD]<<'\t';
      }
      outputFile<<std::endl;
    }
  }
  outputFile.close();
}


void solve_node_matrix(LogicalRegion & V0Tu0, LogicalRegion & V1Tu1, LogicalRegion & V0Td0, LogicalRegion & V1Td1,
		       Context ctx, HighLevelRuntime *runtime) {

  TaskLauncher lu_solve_task(LU_SOLVE_TASK_ID, TaskArgument(NULL, 0));

  assert(V0Tu0 != LogicalRegion::NO_REGION);
  assert(V1Tu1 != LogicalRegion::NO_REGION);
  assert(V0Td0 != LogicalRegion::NO_REGION);
  assert(V1Td1 != LogicalRegion::NO_REGION);

  lu_solve_task.add_region_requirement(RegionRequirement(V0Tu0, READ_ONLY,  EXCLUSIVE, V0Tu0));
  lu_solve_task.add_region_requirement(RegionRequirement(V1Tu1, READ_ONLY,  EXCLUSIVE, V1Tu1));
  lu_solve_task.add_region_requirement(RegionRequirement(V0Td0, READ_WRITE, EXCLUSIVE, V0Td0));
  lu_solve_task.add_region_requirement(RegionRequirement(V1Td1, READ_WRITE, EXCLUSIVE, V1Td1));
  
  lu_solve_task.region_requirements[0].add_field(FID_X);
  lu_solve_task.region_requirements[1].add_field(FID_X);
  lu_solve_task.region_requirements[2].add_field(FID_X);
  lu_solve_task.region_requirements[3].add_field(FID_X);

  runtime->execute_task(ctx, lu_solve_task);
}


// solve the system for Shur complement
void lu_solve_task(const Task *task, const std::vector<PhysicalRegion> &regions,
		   Context ctx, HighLevelRuntime *runtime) {

  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  assert(task->arglen == 0);
  
  IndexSpace is_V0Tu0 = task->regions[0].region.get_index_space();
  IndexSpace is_V1Tu1 = task->regions[1].region.get_index_space();
  IndexSpace is_V0Td0 = task->regions[2].region.get_index_space();
  IndexSpace is_V1Td1 = task->regions[3].region.get_index_space();

  Domain dom_V0Tu0 = runtime->get_index_space_domain(ctx, is_V0Tu0);
  Domain dom_V1Tu1 = runtime->get_index_space_domain(ctx, is_V1Tu1);
  Domain dom_V0Td0 = runtime->get_index_space_domain(ctx, is_V0Td0);
  Domain dom_V1Td1 = runtime->get_index_space_domain(ctx, is_V1Td1);

  Rect<2> rect_V0Tu0 = dom_V0Tu0.get_rect<2>();
  Rect<2> rect_V1Tu1 = dom_V1Tu1.get_rect<2>();
  Rect<2> rect_V0Td0 = dom_V0Td0.get_rect<2>();
  Rect<2> rect_V1Td1 = dom_V1Td1.get_rect<2>();

  RegionAccessor<AccessorType::Generic, double> acc_V0Tu0 =
    regions[0].get_field_accessor(FID_X).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> acc_V1Tu1 =
    regions[1].get_field_accessor(FID_X).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> acc_V0Td0 =
    regions[2].get_field_accessor(FID_X).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> acc_V1Td1 =
    regions[3].get_field_accessor(FID_X).typeify<double>();
    
  Rect<2> subrect;
  ByteOffset offsets[2];
  
  double *V0Tu0_ptr =  acc_V0Tu0.raw_rect_ptr<2>(rect_V0Tu0, subrect, offsets);
  assert(rect_V0Tu0 == subrect);

  double *V1Tu1_ptr =  acc_V1Tu1.raw_rect_ptr<2>(rect_V1Tu1, subrect, offsets);
  assert(rect_V1Tu1 == subrect);
  
  double *V0Td0_ptr =  acc_V0Td0.raw_rect_ptr<2>(rect_V0Td0, subrect, offsets);
  assert(rect_V0Td0 == subrect);

  double *V1Td1_ptr =  acc_V1Td1.raw_rect_ptr<2>(rect_V1Td1, subrect, offsets);
  assert(rect_V1Td1 == subrect);


  int V0Tu0_rows = rect_V0Tu0.dim_size(0);
  int V0Tu0_cols = rect_V0Tu0.dim_size(1);
  int V1Tu1_rows = rect_V1Tu1.dim_size(0);
  int V1Tu1_cols = rect_V1Tu1.dim_size(1);
  int V0Td0_rows = rect_V0Td0.dim_size(0);
  int V0Td0_cols = rect_V0Td0.dim_size(1);
  int V1Td1_rows = rect_V1Td1.dim_size(0);
  int V1Td1_cols = rect_V1Td1.dim_size(1);

  assert(V0Td0_cols == V1Td1_cols);
  assert(V0Tu0_rows + V1Tu1_rows == V0Tu0_cols + V1Tu1_cols);
  assert(V0Tu0_rows + V1Tu1_rows == V0Td0_rows + V1Td1_rows);
  
  int N     = V0Tu0_rows + V1Tu1_rows;
  int NRHS  = V0Td0_cols;
  int LDA   = N;
  int LDB   = N;
  int INFO;
  int IPIV[N];
  double *A = (double *)calloc(N*N,    sizeof(double));
  double *B = (double *)calloc(N*NRHS, sizeof(double));

  /* form the Shur complement:
     --            --
     |  I    V0Tu0  | 
     | V1Tu1  I     |
     --            --
     and the solutions eta0 and eta1 overwrite
     V1Td1 and V0Td0. (Note the reversed order)
   */

  /*
  for (int j=0; j<V0Tu0_cols; j++) {
    for (int i=0; i<V0Tu0_rows; i++) {
      A[i+j*N] = V0Tu0_ptr[i+j*V0Tu0_rows];
    }
  }

  for (int j=0; j<V1Tu1_cols; j++) {
    for (int i=0; i<V1Tu1_rows; i++) {
      A[(V0Tu0_rows+i)+(V0Tu0_cols+j)*N] = V1Tu1_ptr[i+j*V1Tu1_rows];
    }
  }

  // two identity matrices on the off-diagonal blocks
  for (int i=0; i<N; i++) {
    A[ (V0Tu0_rows+i)%N + i*N] = 1.0;
  }
  */

  // two identity matrices on the diagonal
  for (int i=0; i<N; i++) {
    A[ i + i*N ] = 1.0;
  }

  for (int j=0; j<V0Tu0_cols; j++) {
    for (int i=0; i<V0Tu0_rows; i++) {
      A[ i + (j+V1Tu1_cols)*N ] = V0Tu0_ptr[i+j*V0Tu0_rows];
    }
  }

  for (int j=0; j<V1Tu1_cols; j++) {
    for (int i=0; i<V1Tu1_rows; i++) {
      A[ (V0Tu0_rows+i) + j*N ] = V1Tu1_ptr[i+j*V1Tu1_rows];
    }
  }
  
    
  for (int j=0; j<V0Td0_cols; j++) {
    for (int i=0; i<V0Td0_rows; i++) {
      B[i+j*N] = V0Td0_ptr[i+j*V0Td0_rows];
    }
  }

  for (int j=0; j<V1Td1_cols; j++) {
    for (int i=0; i<V1Td1_rows; i++) {
      B[(V0Td0_rows+i)+j*N] = V1Td1_ptr[i+j*V1Td1_rows];
    }
  }

  lapack::dgesv_(&N, &NRHS, A, &LDA, IPIV, B, &LDB, &INFO);
  assert(INFO == 0);


  // eta1
  for (int j=0; j<V0Td0_cols; j++) {
    for (int i=0; i<V0Td0_rows; i++) {
      V0Td0_ptr[i+j*V0Td0_rows] = B[i+j*N];
    }
  }

  // eta0
  for (int j=0; j<V1Td1_cols; j++) {
    for (int i=0; i<V1Td1_rows; i++) {
      V1Td1_ptr[i+j*V1Td1_rows] = B[(V0Td0_rows+i)+j*N];
    }
  }

  free(A);
  free(B);
}


/*
void saveVectorToText(const std::string outputFileName, Eigen::VectorXd & inputVector){
  std::ofstream outputFile;
  outputFile.open(outputFileName.c_str());
  if (!outputFile.is_open()){
    std::cout<<"Error! Unable to open file for saving."<<std::endl;
    exit(EXIT_FAILURE);
  }
  outputFile << inputVector.size() << std::endl;
  for (unsigned int i = 0; i < inputVector.size(); i++)
    outputFile<<std::setprecision(20)<<inputVector[i]<<" "<<std::endl;
  outputFile.close();
}
*/

int count_leaf(FSTreeNode *node) {
  if (node->lchild == NULL && node->rchild == NULL)
    return 1;
  else {
    int n1 = count_leaf(node->lchild);
    int n2 = count_leaf(node->rchild);
    return n1+n2;
  }
}
