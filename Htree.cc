#include <iomanip>

#include "Htree.h"
#include "lapack_blas.h"
#include "macros.h"


void register_save_task() {
  
  HighLevelRuntime::register_legion_task<save_task>(SAVE_REGION_TASK_ID,
						    Processor::LOC_PROC,
						    true, true,
						    AUTO_GENERATE_ID,
						    TaskConfigOptions(true/*leaf*/),
						    "save_region");
}


void register_zero_matrix_task() {
  
  HighLevelRuntime::register_legion_task<zero_matrix_task>(ZERO_MATRIX_TASK_ID,
							   Processor::LOC_PROC,
							   true, true,
							   AUTO_GENERATE_ID,
							   TaskConfigOptions(true/*leaf*/),
							   "init_zero_matrix");

}


void register_circulant_matrix_task() {
  
  HighLevelRuntime::register_legion_task<circulant_matrix_task>(CIRCULANT_MATRIX_TASK_ID,
								Processor::LOC_PROC,
								true, true,
								AUTO_GENERATE_ID,
								TaskConfigOptions(true/*leaf*/),
								"init_circulant_matrix");

}


void register_circulant_kmat_task() {
  
  HighLevelRuntime::register_legion_task<circulant_kmat_task>(CIRCULANT_KMAT_TASK_ID,
							      Processor::LOC_PROC,
							      true, true,
							      AUTO_GENERATE_ID,
							      TaskConfigOptions(true/*leaf*/),
							      "init_circulant_kmat");

}


FSTreeNode::FSTreeNode() {

  nrow = 0;
  ncol = 0;

  row_beg = 0;
  col_beg = 0;

  lchild = NULL;
  rchild = NULL;
  Hmat   = NULL;
  
  matrix = NULL;
  kmat   = NULL;
  
  isLegionLeaf = false;
}


LR_Matrix::LR_Matrix(int nleaf_per_legion_node, Context ctx_, HighLevelRuntime *runtime_) {

  nleaf_per_node  = nleaf_per_legion_node;
  this -> ctx     = ctx_;
  this -> runtime = runtime_;
}


/* Input:
 *   N - problem size
 *   r - every off-diagonal block has the same rank
 *   rhs_cols  - column size of the right hand side
 *   threshold - size of dense blocks at the leaf level
 */
LR_Matrix::LR_Matrix(int N, int threshold, int rhs_cols_, int r_,
		     Context ctx_, HighLevelRuntime *runtime_): rhs_rows(N),
								rhs_cols(rhs_cols_),
								r(r_),
								ctx(ctx_),
								runtime(runtime_) {
  

  uroot  = new FSTreeNode;
  uroot -> nrow = N;
  uroot -> ncol = rhs_cols;
  create_default_tree(uroot, r, threshold);

  //print_legion_tree(uroot);

  // postpone creating V tree after setting the legion leaf
								}


void LR_Matrix::create_legion_leaf(int nleaf_per_legion_node) {

  nleaf_per_node  = nleaf_per_legion_node;
  
  int nLegionLeaf = 0;
  create_legion_leaf(uroot, nleaf_per_node, nLegionLeaf);
  std::cout << "Number of legion leaves: " << nLegionLeaf << std::endl;

  vroot  = new FSTreeNode;
  vroot -> nrow = uroot->nrow;
  vroot -> ncol = 0;
  create_vnode_from_unode(uroot, vroot);

  //print_legion_tree(vroot);
}


/* Implicit input: a rank R matrix U U^T plus diagonal (to make it non-singular)
 *   if U has a specific pattern, it does not require be stored as a whole matrix. E.g. U(:,1) = (1:N)%m, U(:,2) = (2:N+1)%m
 *
 * Args:
 *   diag - the diagonal entry for the dense block
 *   RHS  - right hand side of the problem
 */
void LR_Matrix::init_circulant_matrix(double diag) {

  init_Umat(uroot);       // row_beg = 0
  init_Vmat(vroot, diag); // row_beg = 0
}


void LR_Matrix::init_circulant_matrix(double diag, int num_node) {

  Range tag = {0, num_node};
  init_Umat(uroot, tag);       // row_beg = 0
  init_Vmat(vroot, diag, tag); // row_beg = 0

  //print_legion_tree(uroot);
  //print_legion_tree(vroot);
}


void create_default_tree(FSTreeNode *node, int r, int threshold) {

  int N = node->nrow;
  if (N > threshold) { // sub-divide the matrix, otherwise it is a dense block

    node->lchild = new FSTreeNode;
    node->rchild = new FSTreeNode;

    node->lchild->nrow = N/2;
    node->rchild->nrow = N - N/2;
    
    node->lchild->ncol = r;
    node->rchild->ncol = r;

    node->lchild->col_beg = node->col_beg + node->ncol;
    node->rchild->col_beg = node->col_beg + node->ncol;

    // recursive call
    create_default_tree(node->lchild, r, threshold);
    create_default_tree(node->rchild, r, threshold);
    
  } else {
    assert(N > r); // assume the dense block is still low rank (without diagnal)
  }

}


void LR_Matrix::init_RHS(double *RHS) {
  init_RHS(uroot, RHS); // row_beg = 0
}


void LR_Matrix::init_RHS(int rand_seed, bool wait /*=false*/) {

  assert(rhs_cols == 1);
  init_RHS(uroot, rand_seed, wait /*, row_beg = 0*/);
}


void LR_Matrix::init_RHS(int rand_seed, int node_num, bool wait /*=false*/) {

  assert(rhs_cols == 1);
  Range tag = {0, node_num};
  init_RHS(uroot, rand_seed, tag, wait /*, row_beg = 0*/); 
}


void LR_Matrix::init_RHS(FSTreeNode *node, double *RHS, int row_beg) {

  if (node->isLegionLeaf == true) {
    assert(node->matrix != NULL);
    node->matrix->set_matrix_data(RHS, rhs_rows, rhs_cols, ctx, runtime, row_beg);  
    return;
    
  } else { // recursively split RHS    
    init_RHS(node->lchild, RHS, row_beg);
    init_RHS(node->rchild, RHS, row_beg+node->lchild->nrow);
  }  
}


void LR_Matrix::init_RHS(FSTreeNode *node, int rand_seed, bool wait, int row_beg) {

  if (node->isLegionLeaf == true) {
    assert(node->matrix != NULL);
    //node->matrix->set_matrix_data(RHS, rhs_rows, rhs_cols, ctx,
    //runtime, row_beg);

    InitRHSTask::TaskArgs args;
    args.rand_seed = rand_seed;

    InitRHSTask launcher(TaskArgument(&args, sizeof(args)));
    
    launcher.add_region_requirement(RegionRequirement(node->matrix->data, WRITE_DISCARD, EXCLUSIVE, node->matrix->data).
				    add_field(FID_X));

    Future fm = runtime->execute_task(ctx, launcher);

    /*
      if(wait) {
      fm.get_void_result();
      }
    */
    return;
    
  } else { // recursively split RHS    
    init_RHS(node->lchild, rand_seed, wait, row_beg);
    init_RHS(node->rchild, rand_seed, wait, row_beg+node->lchild->nrow);
  }  
}


void LR_Matrix::init_RHS(FSTreeNode *node, int rand_seed, Range tag, bool wait, int row_beg) {

  if (node->isLegionLeaf == true) {
    assert(node->matrix != NULL);

    //typename
    InitRHSTask::TaskArgs args;
    args.rand_seed = rand_seed;

    InitRHSTask launcher(TaskArgument(&args, sizeof(args)),
			 Predicate::TRUE_PRED,
			 0,
			 tag.begin);
    
    launcher.add_region_requirement(RegionRequirement(node->matrix->data,
						      WRITE_DISCARD,
						      EXCLUSIVE,
						      node->matrix->data).add_field(FID_X));
    runtime->execute_task(ctx, launcher);
    
  } else { // recursively split RHS
    
    int   half = tag.size/2;
    Range ltag = {tag.begin,      half};
    Range rtag = {tag.begin+half, half};
    init_RHS(node->lchild, rand_seed, ltag, wait, row_beg);
    init_RHS(node->rchild, rand_seed, rtag, wait, row_beg+node->lchild->nrow);
  }  
}


void LR_Matrix::init_Umat(FSTreeNode *node, int row_beg) {

  if (node->isLegionLeaf == true) {

    assert(node->matrix != NULL); // initialize region here
    node->matrix->set_circulant_matrix_data(rhs_cols, row_beg, r, ctx, runtime);
    return;
    
  } else {
    init_Umat(node->lchild, row_beg);
    init_Umat(node->rchild, row_beg + node->lchild->nrow);
  }  
}


void LR_Matrix::init_Umat(FSTreeNode *node, Range tag, int row_beg) {

  if (node->isLegionLeaf == true) {

    assert(node->matrix != NULL); // initialize region here
    node->matrix->set_circulant_matrix_data(rhs_cols,
					    row_beg,
					    r,
					    tag,
					    ctx,
					    runtime);
  } else {
    int   half = tag.size/2;
    Range ltag = {tag.begin,      half};
    Range rtag = {tag.begin+half, half};
    init_Umat(node->lchild, ltag, row_beg);
    init_Umat(node->rchild, rtag, row_beg + node->lchild->nrow);
  }  
}


void LR_Matrix::init_Vmat(FSTreeNode *node, double diag, int row_beg) {

  if (node->Hmat != NULL) // skip vroot
    set_circulant_Hmatrix_data(node->Hmat, row_beg);

  if (node->isLegionLeaf == true) {

    // init V
    if (node->matrix->cols > 0)
      // when the legion leaf is the real leaf, there is
      // no data here.
      node->matrix->set_circulant_matrix_data(0, row_beg, r, ctx, runtime);

    // init K
    int nrow = node->kmat->rows;
    int ncol = node->kmat->cols;
   
    // only support real leaf currently
    assert(node->lchild == NULL && node->rchild == NULL);
    CirKmatArg arg = {row_beg, r, diag};
    node->kmat->set_circulant_kmat(arg, ctx, runtime);

  } else {
    init_Vmat(node->lchild, diag, row_beg);
    init_Vmat(node->rchild, diag, row_beg+node->lchild->nrow);
  }
}


void LR_Matrix::init_Vmat(FSTreeNode *node, double diag, Range tag, int row_beg) {

  if (node->Hmat != NULL) // skip vroot
    set_circulant_Hmatrix_data(node->Hmat, tag, row_beg);

  if (node->isLegionLeaf == true) {

    // init V
    // when the legion leaf is the real leaf, there is no data here.
    if (node->matrix->cols > 0)
      node->matrix->set_circulant_matrix_data(0, row_beg, r, tag, ctx, runtime);

    // init K
    int nrow = node->kmat->rows;
    int ncol = node->kmat->cols;

    // only support real leaf currently
    /*
      assert(node->lchild == NULL && node->rchild == NULL);
      CirKmatArg arg = {row_beg, r, diag};
      node->kmat->set_circulant_kmat(arg, tag, ctx, runtime);
    */
    init_circulant_Kmat(node, row_beg, r, diag, tag, ctx, runtime);
    
  } else {
    int   half = tag.size/2;
    Range ltag = {tag.begin,      half};
    Range rtag = {tag.begin+half, half};
    init_Vmat(node->lchild, diag, ltag, row_beg);
    init_Vmat(node->rchild, diag, rtag, row_beg+node->lchild->nrow);
  }
}


void init_circulant_Kmat(FSTreeNode *V_legion_leaf, int row_beg_glo, int rank,
			 double diag, Range mapping_tag, Context ctx,
			 HighLevelRuntime *runtime)
{
  int nleaf = count_leaf(V_legion_leaf);
  int max_tree_size = nleaf * 2;
  assert(max_tree_size < MAX_TREE_SIZE);
  
  //typename
  InitCirculantKmatTask::TaskArgs<MAX_TREE_SIZE> args;
  //FSTreeNode arg[max_tree_size];

  args.treeArray[0] = *V_legion_leaf;
  int size = tree_to_array(V_legion_leaf, args.treeArray, 0);
  assert(size < max_tree_size);

  // encode the array size
  //args.treeArray[0].col_beg = max_tree_size;
  //args.treeSize = max_tree_size;
  args.row_beg_global = row_beg_glo;
  args.rank = rank;
  args.diag = diag;
  InitCirculantKmatTask launcher(TaskArgument(&args, sizeof(args)),
				 Predicate::TRUE_PRED,
				 0,
				 mapping_tag.begin);
  
  // k region
  launcher.add_region_requirement(RegionRequirement(V_legion_leaf->kmat->data, WRITE_DISCARD, EXCLUSIVE, V_legion_leaf->kmat->data).add_field(FID_X));

  runtime->execute_task(ctx, launcher);
}


void LR_Matrix::save_solution(std::string filename) {

  ColRange ru = {0, rhs_cols};
  save_region(uroot, ru, filename, ctx, runtime, true/*wait*/);
}


// this function picks legion leaf nodes as those having the number of threshold real matrix leaves.
// when threshold = 1, the legion leaf and real matrix leaf coinside.
// nLegionLeaf records the number of legion leaves as an indicator of the number of leaf tasks.
int LR_Matrix::create_legion_leaf(FSTreeNode *node, int threshold, int &nLegionLeaf) {

  int nRealLeaf;
  
  if (node->lchild == NULL & node->rchild == NULL) // real matrix leaf
    nRealLeaf = 1;
  else {
    int nl = create_legion_leaf(node->lchild, threshold, nLegionLeaf);
    int nr = create_legion_leaf(node->rchild, threshold, nLegionLeaf);
    nRealLeaf = nl + nr;
  }

  // mark "Legion Leaf" on all leaves below the legion leaf level
  node->isLegionLeaf = (nRealLeaf > threshold) ? false : true;
  
  // count the number of legion leaves
  if (node->isLegionLeaf == false) {
    if (node->lchild->isLegionLeaf == true) { // legion leaf
      nLegionLeaf++;
      create_legion_matrix(node->lchild);
    }
    if (node->rchild->isLegionLeaf == true) { // legion leaf
      nLegionLeaf++;
      create_legion_matrix(node->rchild);
    }
  }
  
  return nRealLeaf;
}


void LR_Matrix::create_legion_matrix(FSTreeNode *node) {

  assert(node->isLegionLeaf == true);
    
  set_row_begin_index(node, 0);

  int row_size = node->nrow;
  int col_size = count_column_size(node, node->col_beg);
  //printf("row_size: %d, col_size: %d.\n", row_size, col_size);

  node->matrix = new LeafData;
  node->matrix->rows = row_size;
  node->matrix->cols = col_size;
  create_matrix(node->matrix->data, row_size, col_size, ctx, runtime);    
}


void LR_Matrix::create_matrix_region(FSTreeNode *node) {

  if (node->isLegionLeaf == true) {
    
    set_row_begin_index(node, 0);

    int row_size = node->nrow;
    int col_size = count_column_size(node, node->col_beg);
    //printf("row_size: %d, col_size: %d.\n", row_size, col_size);

    node->matrix = new LeafData;
    node->matrix->rows = row_size;
    node->matrix->cols = col_size;
    create_matrix(node->matrix->data, row_size, col_size, ctx, runtime);
    
  } else {
    create_matrix_region(node->lchild);
    create_matrix_region(node->rchild);
  }
}


// this function computes the begining row index in region of Legion leaf
void set_row_begin_index(FSTreeNode *node, int row_beg) {

  node->row_beg = row_beg;
  
  if (node->lchild == NULL && node->rchild == NULL) // real matrix leaf
    return;
  else {
    set_row_begin_index(node->lchild, row_beg);
    set_row_begin_index(node->rchild, row_beg + node->lchild->nrow);
  }
}


int count_column_size(FSTreeNode *node, int col_size) {

  if (node->lchild == NULL && node->rchild == NULL) // real matrix leaf
    return col_size + node->ncol;
  else {
    int n1 = count_column_size(node->lchild, col_size + node->ncol);
    int n2 = count_column_size(node->rchild, col_size + node->ncol);
    return std::max(n1, n2);
  }
}


int max_row_size(FSTreeNode * vnode) {

  if (vnode->lchild == NULL && vnode->rchild == NULL) {
    return vnode->nrow;
  }

  int m1 = max_row_size(vnode->lchild);
  int m2 = max_row_size(vnode->rchild);
  
  return std::max(m1, m2);
}


void LR_Matrix::create_vnode_from_unode(FSTreeNode *unode, FSTreeNode *vnode) {

  // create V tree
  if (unode -> lchild != NULL && unode -> rchild != NULL) { // two children both exist or not

    vnode -> lchild = new FSTreeNode; //unode -> rchild -> copy_node();
    vnode -> rchild = new FSTreeNode; //unode -> lchild -> copy_node();

    vnode -> lchild -> row_beg = unode -> lchild -> row_beg; // u and v have the same row structure
    vnode -> rchild -> row_beg = unode -> rchild -> row_beg;
    
    vnode -> lchild -> nrow = unode -> lchild -> nrow;
    vnode -> rchild -> nrow = unode -> rchild -> nrow;
    
    vnode -> lchild -> ncol = unode -> rchild -> ncol; // notice the order here
    vnode -> rchild -> ncol = unode -> lchild -> ncol; // it is reversed in v

    // set column begin index for Legion leaf,
    // to be used in the big V matrix at Legion leaf
    if (unode -> isLegionLeaf == true) {

      vnode -> isLegionLeaf = true;

      if (unode->matrix == NULL) { // skip Legion leaf
	vnode -> lchild -> col_beg = vnode -> col_beg + vnode -> ncol;
	vnode -> rchild -> col_beg = vnode -> col_beg + vnode -> ncol;
      }
    }
      
    create_vnode_from_unode(unode->lchild, vnode->lchild);
    create_vnode_from_unode(unode->rchild, vnode->rchild);
    
  } else {
    vnode -> lchild = NULL;
    vnode -> rchild = NULL;


    if (unode -> isLegionLeaf == true) {
      vnode -> isLegionLeaf = true;
    }

    
  }


  // create H-tiled matrices for two children including Legion leaf
  if (unode -> isLegionLeaf == false) {
    
    vnode -> lchild -> Hmat =  new FSTreeNode;
    vnode -> rchild -> Hmat =  new FSTreeNode;

    vnode -> lchild -> Hmat -> nrow = vnode -> lchild -> nrow;
    vnode -> rchild -> Hmat -> nrow = vnode -> rchild -> nrow;

    vnode -> lchild -> Hmat -> ncol = vnode -> lchild -> ncol;
    vnode -> rchild -> Hmat -> ncol = vnode -> rchild -> ncol;
    
    create_Hmatrix(vnode->lchild, vnode->lchild->Hmat, vnode->lchild->ncol);
    create_Hmatrix(vnode->rchild, vnode->rchild->Hmat, vnode->rchild->ncol);
    
    //vnode -> matrix -> rows = vnode -> nrow;
    //vnode -> matrix -> cols = vnode -> ncol;
    //create_matrix(vnode->matrix->data, vnode -> matrix -> rows, vnode -> matrix -> cols, ctx, runtime);
  }

    
  // create a big rectangle at Legion leaf for lower levels not including Legion leaf
  // please refer to Eric's slides of ver 2
  if (unode->matrix != NULL) {

    assert(unode->nrow == vnode->nrow);
    int urow = unode->matrix->rows;
    int ucol = unode->matrix->cols;
    int vrow = urow;
    int vcol = ucol - (unode->col_beg + unode->ncol); // u and v have the same size under Legion leaf

    //if (vcol > 0) {
    // when the legion leaf is the real leaf, there is
    // no data here.
    vnode->matrix = new LeafData;
    vnode->matrix->rows = vrow;
    vnode->matrix->cols = vcol;
    create_matrix(vnode->matrix->data, vrow, vcol, ctx, runtime);
    //}

    // create K matrix
    vnode->kmat = new LeafData;
    vnode->kmat->rows = vnode->nrow;
    vnode->kmat->cols = max_row_size(vnode);
    create_matrix(vnode->kmat->data, vnode->kmat->rows, vnode->kmat->cols, ctx, runtime);
  }
}


void fill_circulant_Kmat(FSTreeNode * vnode, int row_beg_glo, int r, double diag, double *Kmat, int LD) {

  if (vnode->lchild == NULL && vnode->rchild == NULL) {

    int ksize = vnode->nrow;
    
    // init U as a circulant matrix
    double *U = (double *) malloc(ksize*r * sizeof(double));
    for (int j=0; j<r; j++) {
      for (int i=0; i<ksize; i++) {
	U[i+j*ksize] = (vnode->row_beg+row_beg_glo+i+j) % r;
      }
    }

    // init the diagonal entries
    for (int i=0; i<ksize; i++)
      Kmat[vnode->row_beg + i + LD*i] = diag;
    
    char transa = 'n';
    char transb = 't';
    int  m = ksize;
    int  n = ksize;
    int  k = r;
    int  lda = ksize;
    int  ldb = ksize;
    int  ldc = LD;
    double alpha = 1.0;
    double beta  = 1.0;
    double *A = U;
    double *B = U;
    double *C = Kmat + vnode->row_beg;
    blas::dgemm_(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);

    /*
    std::cout << "Kmat: " << std::endl;
    for (int j=0; j<ksize; j++)
      for (int i=0; i<ksize; i++)
	std::cout << Kmat[vnode->row_beg + i + LD*j] << "\t";
    std::cout << std::endl;
    */

    //printf("After init Kmat.\n");
    
    free(U);
    return;
  }

  fill_circulant_Kmat(vnode->lchild, row_beg_glo, r, diag, Kmat, LD);
  fill_circulant_Kmat(vnode->rchild, row_beg_glo, r, diag, Kmat, LD);
}


void LR_Matrix::create_Hmatrix(FSTreeNode *node, FSTreeNode * Hmat, int ncol) {

  if (node->isLegionLeaf == true) {

    Hmat->nrow = node->nrow;
    Hmat->ncol = node->ncol;
    
    Hmat->matrix = new LeafData;
    Hmat->matrix->rows = node->nrow;
    Hmat->matrix->cols = ncol;
    Hmat->isLegionLeaf = true;
    create_matrix(Hmat->matrix->data, Hmat -> matrix -> rows, Hmat -> matrix -> cols, ctx, runtime);
 
  } else {
    
    Hmat->lchild = new FSTreeNode;
    Hmat->rchild = new FSTreeNode;

    // to be used in initialization
    Hmat->lchild->row_beg = Hmat->row_beg;
    Hmat->rchild->row_beg = Hmat->row_beg + node->lchild->nrow;
    
    create_Hmatrix(node->lchild, Hmat->lchild, ncol);
    create_Hmatrix(node->rchild, Hmat->rchild, ncol);
  }
}


void LR_Matrix::set_circulant_Hmatrix_data(FSTreeNode * Hmat, int row_beg) {

  if (Hmat->lchild == NULL && Hmat->rchild == NULL) {

    int glo = row_beg;
    int loc = Hmat->row_beg;
    assert(Hmat->ncol == r);
    Hmat->matrix->set_circulant_matrix_data(0, glo + loc, r, ctx, runtime);
    
  } else {
    set_circulant_Hmatrix_data(Hmat->lchild, row_beg);
    set_circulant_Hmatrix_data(Hmat->rchild, row_beg);
  }  
}


void LR_Matrix::set_circulant_Hmatrix_data(FSTreeNode * Hmat, Range
					   tag, int row_beg) {

  if (Hmat->lchild == NULL && Hmat->rchild == NULL) {

    int glo = row_beg;
    int loc = Hmat->row_beg;
    assert(Hmat->ncol == r);
    Hmat->matrix->set_circulant_matrix_data(0, glo + loc, r, tag, ctx, runtime);
    
  } else {
    int   half = tag.size/2;
    Range ltag = {tag.begin,      half};
    Range rtag = {tag.begin+half, half};
    set_circulant_Hmatrix_data(Hmat->lchild, ltag, row_beg);
    set_circulant_Hmatrix_data(Hmat->rchild, rtag, row_beg);
  }  
}


void LR_Matrix::print_Vmat(FSTreeNode *node, std::string filename) {

  //  if (node == vroot)
  //save_region(node, filename, ctx, runtime); // print big V matrix

  if (node->Hmat != NULL)
    save_region(node->Hmat, filename, ctx, runtime);
  else if (node != vroot)
    return;

  if (node->lchild != NULL && node->rchild != NULL) {
    print_Vmat(node->lchild, filename);
    print_Vmat(node->rchild, filename);
  }
}


void LeafData::set_circulant_matrix_data(int col_beg, int row_beg, int r, Context ctx, HighLevelRuntime *runtime) {    

  CirArg cir_arg = {col_beg, row_beg, r};
  TaskLauncher circulant_matrix_task(CIRCULANT_MATRIX_TASK_ID, TaskArgument(&cir_arg, sizeof(CirArg)));

  //circulant_matrix_task.add_region_requirement(RegionRequirement(data,
  //WRITE_DISCARD, EXCLUSIVE, data));
  circulant_matrix_task.add_region_requirement(RegionRequirement(data, READ_WRITE, EXCLUSIVE, data));
  circulant_matrix_task.region_requirements[0].add_field(FID_X);

  runtime->execute_task(ctx, circulant_matrix_task);
  //Future fm = runtime->execute_task(ctx, circulant_matrix_task);
  //fm.get_void_result();
}


void LeafData::set_circulant_matrix_data(int col_beg, int row_beg, int
					 r, Range tag, Context ctx, HighLevelRuntime *runtime) {    

  CirArg cir_arg = {col_beg, row_beg, r};
  TaskLauncher circulant_matrix_task(CIRCULANT_MATRIX_TASK_ID,
				     TaskArgument(&cir_arg,
						  sizeof(CirArg)),
				     Predicate::TRUE_PRED,
				     0,
				     tag.begin);
  circulant_matrix_task.add_region_requirement(RegionRequirement(data, READ_WRITE, EXCLUSIVE, data));
  circulant_matrix_task.region_requirements[0].add_field(FID_X);
  runtime->execute_task(ctx, circulant_matrix_task);
}


void circulant_matrix_task(const Task *task, const std::vector<PhysicalRegion> &regions,
			   Context ctx, HighLevelRuntime *runtime) {

  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->arglen == sizeof(CirArg));

  const CirArg cir_arg = *((const CirArg*)task->args);
  int col_beg = cir_arg.col_beg;
  int row_beg = cir_arg.row_beg;
  int r       = cir_arg.r;
  
  IndexSpace is = task->regions[0].region.get_index_space();
  Domain dom = runtime->get_index_space_domain(ctx, is);
  Rect<2> rect = dom.get_rect<2>();

  Rect<2> subrect;
  ByteOffset offsets[2];

  double *ptr = regions[0].get_field_accessor(FID_X).
    typeify<double>().raw_rect_ptr<2>(rect, subrect, offsets);
  assert(rect == subrect);
  assert(ptr  != NULL);
  
  int nrow = rect.dim_size(0);
  int ncol = rect.dim_size(1);
  int vol  = rect.volume();
  assert( (ncol - col_beg) % r == 0 );
    
  for (int j=0; j<ncol - col_beg; j++) {
    for (int i=0; i<nrow; i++) {
      int value = (j+i+row_beg)%r;

      int irow = i;
      int icol = j+col_beg;

      assert(irow + icol*nrow < vol);
      ptr[irow + icol*nrow] = value;
    }
  }
}


void LeafData::set_matrix_data(double *mat, int rhs_rows, int rhs_cols, Context ctx, HighLevelRuntime *runtime, int row_beg) {

  InlineLauncher launcher(RegionRequirement(data, WRITE_DISCARD, EXCLUSIVE, data));
  
  launcher.requirement.add_field(FID_X);
  
  PhysicalRegion region = runtime->map_region(ctx, launcher);
  
  RegionAccessor<AccessorType::Generic, double> acc = 
    region.get_field_accessor(FID_X).typeify<double>();
 
  Domain dom = runtime->get_index_space_domain(ctx, data.get_index_space());
  Rect<2> rect = dom.get_rect<2>();

  int nrow = rect.dim_size(0);
  assert(rhs_cols <= rect.dim_size(1));

  for (int j=0; j<rhs_cols; j++) {
    for (int i=0; i<nrow; i++) {
      int pt[2] = {i, j};
      acc.write(DomainPoint::from_point<2>( Point<2> (pt) ), mat[row_beg+i+j*rhs_rows]);
    }
  }    
  runtime->unmap_region(ctx, region);
}


void create_matrix(LogicalRegion & matrix, int nrow, int ncol, Context ctx, HighLevelRuntime *runtime) {  
  int lower[2] = {0,      0};
  int upper[2] = {nrow-1, ncol-1}; // inclusive bound
  Rect<2> rect((Point<2>(lower)), (Point<2>(upper)));
  IndexSpace is = runtime->create_index_space(ctx, Domain::from_rect<2>(rect));
  FieldSpace fs = runtime->create_field_space(ctx);
  FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
  allocator.allocate_field(sizeof(double), FID_X);
  matrix = runtime->create_logical_region(ctx, is, fs);
  assert(matrix != LogicalRegion::NO_REGION);
}


void zero_matrix(LogicalRegion &matrix, Context ctx, HighLevelRuntime *runtime) {
  assert(matrix != LogicalRegion::NO_REGION);
  TaskLauncher zero_matrix_task(ZERO_MATRIX_TASK_ID, TaskArgument(NULL, 0));
  zero_matrix_task.add_region_requirement(RegionRequirement(matrix, WRITE_DISCARD, EXCLUSIVE, matrix));
  zero_matrix_task.region_requirements[0].add_field(FID_X);
  runtime->execute_task(ctx, zero_matrix_task);
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


void save_region(LogicalRegion & matrix, int col_beg, int ncol, std::string filename, Context ctx, HighLevelRuntime *runtime) {

  RegionRequirement req(matrix, READ_ONLY, EXCLUSIVE, matrix);
  req.add_field(FID_X);

  InlineLauncher init(req);
  PhysicalRegion init_region = runtime->map_region(ctx, init);
  init_region.wait_until_valid();

  RegionAccessor<AccessorType::Generic, double> acc =
    init_region.get_field_accessor(FID_X).typeify<double>();

  Domain dom = runtime->get_index_space_domain(ctx, matrix.get_index_space());
  Rect<2> rect = dom.get_rect<2>();

  Rect<2> subrect;
  ByteOffset offsets[2];

  double *ptr = acc.raw_rect_ptr<2>(rect, subrect, offsets);
  assert(rect == subrect);

  int nrow = rect.dim_size(0);
  double *x = ptr + col_beg * nrow; // to be double checked

  //GenericPointInRectIterator<2> pir(rect);

  std::ofstream outputFile(filename.c_str(), std::ios_base::app);
  outputFile<<nrow<<std::endl;
  outputFile<<ncol<<std::endl;

  for (int j=0; j<ncol; j++) {
    for (int i=0; i<nrow; i++) {
      outputFile << std::setprecision(20) << x[i+j*nrow] << '\t';
    }
    outputFile << std::endl;
  }

  outputFile.close();  
  runtime->unmap_region(ctx, init_region);
}


void save_region(LogicalRegion & matrix, std::string filename, Context ctx, HighLevelRuntime *runtime) {

  RegionRequirement req(matrix, READ_ONLY, EXCLUSIVE, matrix);
  req.add_field(FID_X);

  InlineLauncher init(req);
  PhysicalRegion init_region = runtime->map_region(ctx, init);
  init_region.wait_until_valid();

  RegionAccessor<AccessorType::Generic, double> acc =
    init_region.get_field_accessor(FID_X).typeify<double>();

  Domain dom = runtime->get_index_space_domain(ctx, matrix.get_index_space());
  Rect<2> rect = dom.get_rect<2>();

  int nrow = rect.dim_size(0);
  int ncol = rect.dim_size(1);

  std::ofstream outputFile(filename.c_str(), std::ios_base::app);
  outputFile<<nrow<<std::endl;
  outputFile<<ncol<<std::endl;

  GenericPointInRectIterator<2> pir(rect);
  for (int i=0; i<nrow; i++) {
    for (int j=0; j<ncol; j++) {
      assert(pir.p[1] < nrow);
      assert(pir.p[0] < ncol);

      double x = acc.read(DomainPoint::from_point<2>(pir.p));
      outputFile << x << '\t';
      pir++;
    }
    outputFile << std::endl;
  }

  outputFile.close();  
  runtime->unmap_region(ctx, init_region);
}


void save_region(FSTreeNode * node, ColRange rg, std::string filename,
		 Context ctx, HighLevelRuntime *runtime, bool wait) {

  if (node->isLegionLeaf == true) {

    //save_region(node->matrix->data, rg.col_beg, rg.ncol, filename, ctx, runtime);

    //typename
    SaveRegionTask::TaskArgs args;
    int len = filename.size();
    filename.copy(args.filename, len, 0);
    args.filename[len] = '\0';
    args.col_range = rg;
    
    SaveRegionTask launcher(TaskArgument(&args, sizeof(args)));
    
    launcher.add_region_requirement(RegionRequirement(node->matrix->data, READ_ONLY, EXCLUSIVE, node->matrix->data).
				    add_field(FID_X));

    Future fm = runtime->execute_task(ctx, launcher);

    if(wait)
      fm.get_void_result();
    
  } else {
    save_region(node->lchild, rg, filename, ctx, runtime, wait);
    save_region(node->rchild, rg, filename, ctx, runtime, wait);
  }  
}


void save_region(FSTreeNode * node, std::string filename, Context ctx, HighLevelRuntime *runtime) {

  if (node->isLegionLeaf == true) {
    
    TaskLauncher save_task(SAVE_REGION_TASK_ID, TaskArgument(&filename[0], filename.size()+1));

    save_task.add_region_requirement(RegionRequirement(node->matrix->data, READ_ONLY,  EXCLUSIVE, node->matrix->data));
    save_task.region_requirements[0].add_field(FID_X);

    runtime->execute_task(ctx, save_task);
    
  } else {
    save_region(node->lchild, filename, ctx, runtime);
    save_region(node->rchild, filename, ctx, runtime);
  }  
}


void save_task(const Task *task, const std::vector<PhysicalRegion> &regions,
	       Context ctx, HighLevelRuntime *runtime) {

  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  char* filename = (char *)task->args;

  IndexSpace is_u = task->regions[0].region.get_index_space();
  Domain dom_u = runtime->get_index_space_domain(ctx, is_u);
  Rect<2> rect_u = dom_u.get_rect<2>();

  RegionAccessor<AccessorType::Generic, double> acc_u =
    regions[0].get_field_accessor(FID_X).typeify<double>();
  
  Rect<2> subrect;
  ByteOffset offsets[2];  

  double *u_ptr = acc_u.raw_rect_ptr<2>(rect_u, subrect, offsets);
  assert(rect_u == subrect);


  int nrow = rect_u.dim_size(0);
  int ncol = rect_u.dim_size(1);

  std::ofstream outputFile(filename, std::ios_base::app);
  if (!outputFile.is_open())
    std::cout << "Error opening file." << std::endl;
  outputFile<<nrow<<std::endl;
  outputFile<<ncol<<std::endl;

  for (int i=0; i<nrow; i++) {
    for (int j=0; j<ncol; j++) {

      int pnt[] = {i, j};
      double x = acc_u.read(DomainPoint::from_point<2>( Point<2>(pnt) ));
      outputFile << x << '\t';
    }
    outputFile << std::endl;
  }
  outputFile.close();
}


void print_legion_tree(FSTreeNode * node) {

  if (node == NULL) return;

  printf("col_beg: %d, row_beg: %d, nrow: %d, ncol: %d, %s\n", node->col_beg, node->row_beg, node->nrow, node->ncol, node->isLegionLeaf ? "legion leaf": "");

  //if (node->isLegionLeaf == true)
  //std::cout << "Legion leaf." << std::endl;
    
  if (node->matrix != NULL) {

    int nrow = node->matrix->rows;
    int ncol = node->matrix->cols;

    printf("Matrix size: %d x %d\n", nrow, ncol);
  }

  if (node->kmat != NULL) {
      
    int nrow = node->kmat->rows;
    int ncol = node->kmat->cols;
    printf("K Mat: %d x %d\n", nrow, ncol);
  }

    
  print_legion_tree(node->lchild);
  print_legion_tree(node->rchild);
}


void save_kmat(FSTreeNode * node, std::string filename, Context ctx, HighLevelRuntime *runtime) {

  if (node->isLegionLeaf == true) {
    
    TaskLauncher save_task(SAVE_REGION_TASK_ID, TaskArgument(&filename[0], filename.size()+1));

    save_task.add_region_requirement(RegionRequirement(node->kmat->data, READ_ONLY,  EXCLUSIVE, node->kmat->data));
    save_task.region_requirements[0].add_field(FID_X);

    runtime->execute_task(ctx, save_task);
    
  } else {
    save_kmat(node->lchild, filename, ctx, runtime);
    save_kmat(node->rchild, filename, ctx, runtime);
  }  
}


void LR_Matrix::get_soln_from_region(double *soln) {
  get_soln_from_region(soln, uroot); // row_beg=0
}


/* --- save solution to matrix --- */
void LR_Matrix::get_soln_from_region(double *soln, FSTreeNode *node, int row_beg) {

  if (node->isLegionLeaf == true) {

    RegionRequirement req(node->matrix->data, READ_ONLY, EXCLUSIVE, node->matrix->data);
    req.add_field(FID_X);
    
    InlineLauncher ilaunch(req);
    PhysicalRegion region = runtime->map_region(ctx, ilaunch);
    region.wait_until_valid();

    RegionAccessor<AccessorType::Generic, double> acc =
      region.get_field_accessor(FID_X).typeify<double>();

    Domain dom = runtime->get_index_space_domain(ctx, node->matrix->data.get_index_space());
    Rect<2> rect = dom.get_rect<2>();

    int nrow = rect.dim_size(0);
    //int ncol = rect.dim_size(1);

    Rect<2> subrect;
    ByteOffset offsets[2];

    double *ptr = acc.raw_rect_ptr<2>(rect, subrect, offsets);
    assert(rect == subrect);

    for (int j=0; j<rhs_cols; j++)
      for (int i=0; i<nrow; i++) {
	soln[i + row_beg + j*rhs_rows] = ptr[i+j*nrow];
      }

    runtime->unmap_region(ctx, region);
  
  } else {
    get_soln_from_region(soln, node->lchild, row_beg);
    get_soln_from_region(soln, node->rchild, row_beg+node->lchild->nrow);
  }
}


//int FastSolver::tree_to_array(FSTreeNode * leaf, FSTreeNode * arg,
//int idx) {
int tree_to_array(FSTreeNode * leaf, FSTreeNode * arg, int idx) {

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
//void FastSolver::tree_to_array(FSTreeNode * leaf, FSTreeNode * arg,
//int idx, int shift) {
void tree_to_array(FSTreeNode * leaf, FSTreeNode * arg, int idx, int shift) {

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


// assume at the real leaf
void LeafData::set_circulant_kmat(CirKmatArg arg, Context ctx, HighLevelRuntime *runtime) {
    
  TaskLauncher circulant_kmat_task(CIRCULANT_KMAT_TASK_ID, TaskArgument(&arg, sizeof(CirKmatArg)));

  // k region
  circulant_kmat_task.add_region_requirement(RegionRequirement(data,
							       WRITE_DISCARD,
							       EXCLUSIVE, data));
  circulant_kmat_task.region_requirements[0].add_field(FID_X);

  runtime->execute_task(ctx, circulant_kmat_task);
  //Future fm = runtime->execute_task(ctx, circulant_kmat_task);
  //fm.get_void_result();
}


// assume at the real leaf
void LeafData::set_circulant_kmat(CirKmatArg arg, Range tag, Context ctx, HighLevelRuntime *runtime) {
    
  TaskLauncher circulant_kmat_task(CIRCULANT_KMAT_TASK_ID,
				   TaskArgument(&arg,
						sizeof(CirKmatArg)),
				   Predicate::TRUE_PRED,
				   0,
				   tag.begin);

  // k region
  circulant_kmat_task.add_region_requirement(RegionRequirement(data,
							       WRITE_DISCARD,
							       EXCLUSIVE, data));
  circulant_kmat_task.region_requirements[0].add_field(FID_X);
  runtime->execute_task(ctx, circulant_kmat_task);
}


void circulant_kmat_task(const Task *task, const std::vector<PhysicalRegion> &regions,
			 Context ctx, HighLevelRuntime *runtime) {

  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  CirKmatArg *arg = (CirKmatArg *)task->args;

  int row_beg = arg->row_beg;
  int r       = arg->r;
  double diag = arg->diag;

  
  RegionAccessor<AccessorType::Generic, double> acc_k = 
    regions[0].get_field_accessor(FID_X).typeify<double>();

  IndexSpace is_k = task->regions[0].region.get_index_space();
  Domain dom_k = runtime->get_index_space_domain(ctx, is_k);
  Rect<2> rect_k = dom_k.get_rect<2>();

  Rect<2> subrect;
  ByteOffset offsets[2];
   
  double *k_ptr = acc_k.raw_rect_ptr<2>(rect_k, subrect, offsets);
  assert(k_ptr != NULL);
  assert(rect_k == subrect);

  int nrow = rect_k.dim_size(0);
  int ncol = rect_k.dim_size(1);
  assert(nrow == ncol);

  int ksize = nrow;
  int LD    = nrow;

  // init U as a circulant matrix
  double *U = (double *) malloc(ksize*r * sizeof(double));
  for (int j=0; j<r; j++) {
    for (int i=0; i<ksize; i++) {
      U[i+j*ksize] = (row_beg+i+j) % r;
    }
  }

  // init kmat to 0
  memset(k_ptr, 0, nrow*ncol*sizeof(double));
  
  // init the diagonal entries
  for (int i=0; i<ksize; i++)
    k_ptr[i + LD*i] = diag;
    
  char transa = 'n';
  char transb = 't';
  int  m = ksize;
  int  n = ksize;
  int  k = r;
  int  lda = ksize;
  int  ldb = ksize;
  int  ldc = LD;
  double alpha = 1.0;
  double beta  = 1.0;
  double *A = U;
  double *B = U;
  double *C = k_ptr;
  blas::dgemm_(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);


  // pre-compute the LU factorization
  // this is confusing (hard to notice) when writing
  // another function to initialize kmat
  /*
  int INFO;
  int IPIV[ksize];
  lapack::dgetrf_(&ksize, &ksize, C, &ldc, IPIV, &INFO);
  assert(INFO == 0);
  // assert no pivoting
  for (int i=0; i<ksize; i++)
    assert(IPIV[i] = i+1);
  */
}

/* ---- SaveRegionTask implementation ---- */

/*static*/
int SaveRegionTask::TASKID;

SaveRegionTask::SaveRegionTask(TaskArgument arg,
			       Predicate pred /*= Predicate::TRUE_PRED*/,
			       MapperID id /*= 0*/,
			       MappingTagID tag /*= 0*/)
  : TaskLauncher(TASKID, arg, pred, id, tag)
{
}

/*static*/
void SaveRegionTask::register_tasks(void)
{
  TASKID = HighLevelRuntime::register_legion_task<SaveRegionTask::cpu_task>(AUTO_GENERATE_ID,
									    Processor::LOC_PROC, 
									    true,
									    true,
									    AUTO_GENERATE_ID,
									    TaskConfigOptions(true/*leaf*/),
									    "save_region_to_file");
  printf("registered as task id %d\n", TASKID);
}

void SaveRegionTask::cpu_task(const Task *task,
			      const std::vector<PhysicalRegion> &regions,
			      Context ctx, HighLevelRuntime *runtime)
{

  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  TaskArgs* task_args = (TaskArgs *)task->args;
  int col_beg = task_args->col_range.col_beg;
  int ncol    = task_args->col_range.ncol;
  char* filename = task_args->filename;

  IndexSpace is   = task->regions[0].region.get_index_space();
  Domain     dom  = runtime->get_index_space_domain(ctx, is);
  Rect<2>    rect = dom.get_rect<2>();

  RegionAccessor<AccessorType::Generic, double> acc =
    regions[0].get_field_accessor(FID_X).typeify<double>();
  
  Rect<2> subrect;
  ByteOffset offsets[2];  

  double *ptr =  acc.raw_rect_ptr<2>(rect, subrect, offsets);
  assert(rect == subrect);
  assert(ptr  != NULL);

  int nrow = rect.dim_size(0);
  assert(col_beg+ncol <= rect.dim_size(1));
  //if (ncol == 0)
  //ncol = rect.dim_size(1) - col_beg;


  std::ofstream outputFile(filename, std::ios_base::app);
  //outputFile << nrow << std::endl;
  //outputFile << ncol << std::endl;

  for (int i=0; i<nrow; i++) {
    for (int j=0; j<ncol; j++) {
      int row_idx = i;
      int col_idx = j+col_beg;
      int pnt[] = {row_idx, col_idx};
      //double x = acc.read(DomainPoint::from_point<2>( Point<2>(pnt)));
      double x = ptr[ row_idx + col_idx*nrow ];
      outputFile << std::setprecision(20) << x << '\t';
    }
    outputFile << std::endl;
  }
  outputFile.close();
}


/* ---- InitRHSTask implementation ---- */

/*static*/
int InitRHSTask::TASKID;

InitRHSTask::InitRHSTask(TaskArgument arg,
			 Predicate pred /*= Predicate::TRUE_PRED*/,
			 MapperID id /*= 0*/,
			 MappingTagID tag /*= 0*/)
  : TaskLauncher(TASKID, arg, pred, id, tag)
{
}

/*static*/
void InitRHSTask::register_tasks(void)
{
  TASKID = HighLevelRuntime::register_legion_task<InitRHSTask::cpu_task>(AUTO_GENERATE_ID,
									 Processor::LOC_PROC, 
									 true,
									 true,
									 AUTO_GENERATE_ID,
									 TaskConfigOptions(true/*leaf*/),
									 "init_RHS");
  printf("registered as task id %d\n", TASKID);
}

void InitRHSTask::cpu_task(const Task *task,
			   const std::vector<PhysicalRegion> &regions,
			   Context ctx, HighLevelRuntime *runtime)
{


  assert(regions.size() == 1);
  assert(task->regions.size() == 1);

  TaskArgs* args = (TaskArgs *)task->args;
  int rand_seed = args->rand_seed;

  IndexSpace is   = task->regions[0].region.get_index_space();
  Domain     dom  = runtime->get_index_space_domain(ctx, is);
  Rect<2>    rect = dom.get_rect<2>();

  RegionAccessor<AccessorType::Generic, double> acc =
    regions[0].get_field_accessor(FID_X).typeify<double>();
  
  Rect<2> subrect;
  ByteOffset offsets[2];  

  double *ptr =  acc.raw_rect_ptr<2>(rect, subrect, offsets);
  assert(rect == subrect);
  assert(ptr  != NULL);

  int nrow = rect.dim_size(0);
  //printf("Start init_RHS task with %d rows.\n", nrow);
  
  srand( rand_seed );
  for (int i=0; i<nrow; i++) {
    int row_idx = i;
    int col_idx = 0;
    //int pnt[] = {row_idx, col_idx};
    ptr[ row_idx + col_idx*nrow ] = frand(0, 1);
  }
}


/* ---- InitCirculantKmatTask implementation ---- */

/*static*/
int InitCirculantKmatTask::TASKID;

InitCirculantKmatTask::InitCirculantKmatTask(TaskArgument arg,
					     Predicate pred /*= Predicate::TRUE_PRED*/,
					     MapperID id /*= 0*/,
					     MappingTagID tag /*= 0*/)
  : TaskLauncher(TASKID, arg, pred, id, tag)
{
}

/*static*/
void InitCirculantKmatTask::register_tasks(void)
{
  TASKID = HighLevelRuntime::register_legion_task<InitCirculantKmatTask::cpu_task>(AUTO_GENERATE_ID,
										   Processor::LOC_PROC, 
										   true,
										   true,
										   AUTO_GENERATE_ID,
										   TaskConfigOptions(true/*leaf*/),
										   "init_Kmat");
  printf("registered as task id %d\n", TASKID);
}

void InitCirculantKmatTask::cpu_task(const Task *task,
				     const std::vector<PhysicalRegion> &regions,
				     Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);

  TaskArgs<MAX_TREE_SIZE> *args = (TaskArgs<MAX_TREE_SIZE> *)task->args;
  int row_beg_global = args->row_beg_global;
  int rank = args->rank;
  double diag = args->diag;
  FSTreeNode *treeArray = args->treeArray;
  assert(task->arglen == sizeof(TaskArgs<MAX_TREE_SIZE>));

  FSTreeNode *vroot = treeArray;
  array_to_tree(treeArray, 0);
  
  RegionAccessor<AccessorType::Generic, double> acc_k = 
    regions[0].get_field_accessor(FID_X).typeify<double>();
  IndexSpace is_k = task->regions[0].region.get_index_space();
  Domain dom_k = runtime->get_index_space_domain(ctx, is_k);
  Rect<2> rect_k = dom_k.get_rect<2>();

  Rect<2> subrect;
  ByteOffset offsets[2];

  double *k_ptr = acc_k.raw_rect_ptr<2>(rect_k, subrect, offsets);
  assert(k_ptr != NULL);
  assert(rect_k == subrect);

  int leading_dimension  = offsets[1].offset / sizeof(double);
  int k_nrow = rect_k.dim_size(0);
  assert( leading_dimension == k_nrow );
  // initialize Kmat
  memset(k_ptr, 0, rect_k.dim_size(0)*rect_k.dim_size(1)*sizeof(double));
  fill_circulant_Kmat(vroot, row_beg_global, rank, diag, k_ptr, leading_dimension);
}


int count_leaf(FSTreeNode *node) {
  if (node->lchild == NULL && node->rchild == NULL)
    return 1;
  else {
    int n1 = count_leaf(node->lchild);
    int n2 = count_leaf(node->rchild);
    return n1+n2;
  }
}


Range Range::lchild ()
{
  int half_size = size/2;
  return (Range){begin, half_size};
}


Range Range::rchild ()
{
  int half_size = size/2;
  return (Range){begin+half_size, half_size};
}

