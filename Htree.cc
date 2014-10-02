#include "Htree.h"

#include <iomanip>

void register_save_task() {

  HighLevelRuntime::register_legion_task<save_task>(SAVE_REGION_TASK_ID, Processor::LOC_PROC, true, true);

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


/*
void LR_Matrix::initialize(HODLR_Tree::node *HODLRroot, Eigen::VectorXd &RHS) {

  assert(HODLRroot != NULL);

  uroot  = new FSTreeNode;
  uroot -> nrow = RHS.rows();
  uroot -> ncol = RHS.cols();

  create_legion_tree(HODLRroot, uroot);


  int nLegionLeaf = 0;
  create_legion_leaf(uroot, nleaf_per_node, nLegionLeaf);
  std::cout << "Number of legion leaves: " << nLegionLeaf << std::endl;

  create_matrix_region(uroot);

  //print_legion_tree(uroot);

  
  vroot =  new FSTreeNode;
  vroot -> nrow = RHS.rows();
  create_vnode_from_unode(uroot, vroot);


  Eigen::MatrixXd umat = RHS; //exactSoln; 
  init_Umat(HODLRroot, uroot, umat);

  Eigen::MatrixXd vmat(RHS.rows(), 0);
  init_Vmat(HODLRroot, vroot, vmat);  
}
*/

/* Input:
 *   N - problem size
 *   r - every off-diagonal block has the same rank
 *   rhs_cols  - column size of the right hand side
 *   threshold - size of dense blocks at the leaf level
 */
LR_Matrix::LR_Matrix(int N, int threshold, int rhs_cols_, int r_, Context ctx_, HighLevelRuntime *runtime_): rhs_rows(N), rhs_cols(rhs_cols_), r(r_), ctx(ctx_), runtime(runtime_) {

  uroot  = new FSTreeNode;
  uroot -> nrow = N;
  uroot -> ncol = rhs_cols;
  create_default_tree(uroot, r, threshold);

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


/*
void LR_Matrix::init_RHS(Eigen::MatrixXd &RHS) {
  assert(uroot->nrow == RHS.rows());
  init_RHS(uroot, RHS);
}


void LR_Matrix::init_RHS(FSTreeNode *node, Eigen::MatrixXd &RHS) {

  if (node->isLegionLeaf == true) {

    assert(node->matrix != NULL);
    // initialize region as RHS
    node->matrix->set_matrix_data(RHS, ctx, runtime);  
    return;
    
  } else { // recursively split RHS
    Eigen::MatrixXd RHS1 = RHS.topRows(node->lchild->nrow);
    Eigen::MatrixXd RHS2 = RHS.bottomRows(node->rchild->nrow);
    
    init_RHS(node->lchild, RHS1);
    init_RHS(node->rchild, RHS2);
  }  
}
*/


void LR_Matrix::init_RHS(double *RHS) {
  init_RHS(uroot, RHS); // row_beg = 0
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


void LR_Matrix::init_Umat(FSTreeNode *node, int row_beg) {

  if (node->isLegionLeaf == true) {

    assert(node->matrix != NULL); // initialize region here
    node->matrix->set_circulant_matrix_data(rhs_cols, row_beg, r, ctx, runtime); // initialize random pattern in region    
    return;
    
  } else {
    init_Umat(node->lchild, row_beg);
    init_Umat(node->rchild, row_beg + node->lchild->nrow);
  }  
}


void LR_Matrix::init_Vmat(FSTreeNode *node, double diag, int row_beg) {

  if (node->Hmat != NULL) // skip vroot
    set_circulant_Hmatrix_data(node->Hmat, row_beg);

  if (node->isLegionLeaf == true) {

    // init V
    node->matrix->set_circulant_matrix_data(0, row_beg, r, ctx, runtime);

    // init K
    int nrow = node->kmat->rows;
    int ncol = node->kmat->cols;
    double *K = (double *) calloc(nrow*ncol, sizeof(double));
    
    fill_circulant_kmat(node, row_beg, r, diag, K, nrow);
    node->kmat->set_matrix_data(K, nrow, ncol, ctx, runtime);
    free(K);
    
  } else {
    init_Vmat(node->lchild, diag, row_beg);
    init_Vmat(node->rchild, diag, row_beg+node->lchild->nrow);
  }
}

/*
  void LR_Matrix::fill_random_kmat(FSTreeNode *node, int r, int row_beg_glo, double diag) {

  if (node->lchild == NULL && node->rchild == NULL) {
  int row_beg_loc = node->row_beg;
  int row_beg = row_beg_loc + row_beg_glo;
  double U[row_beg_loc*r];
  form_LR_matrix(U, row_beg_loc, r);

  double res[row_beg_loc*row_beg_loc];
  for (int i=0; i<row_beg_loc; i++)
  for (int j=0; j<row_beg_loc; j++)
  res[i*row_beg_loc+j] = (i==j) ? diag: 0.0;

  char transa = 'n';
  char transb = 't';
  int m = row_beg_loc;
  int n = row_beg_loc;
  int k = r;
  double alpha = 1.0;
  double beta  = 1.0;
  int lda = row_beg_loc;
  int ldb = row_beg_loc; // ?
  int ldc = row_beg_loc;
    
  blas::dgemm_(&transa, &transb, &m, &n, &k, &alpha, U, &lda, U, &ldb, &beta, res, &ldc);

    
    
  } else {
  fill_random_kmat();
  }
  }
*/

void LR_Matrix::save_solution(std::string filename) {

  range ru = {0, rhs_cols};
  save_region(uroot, ru, filename.c_str(), ctx, runtime);

}


/*
// The left child corresponds to the top off-diagonal block.
// The left child lies above the right child in the Legion tree, as shown in the slides.
// This function constructs the Legion tree and computes the begin column index of every node.
void LR_Matrix::create_legion_tree(HODLR_Tree::node *HODLRnode, FSTreeNode *Lnode) {

  Lnode -> lchild = new FSTreeNode;
  Lnode -> rchild = new FSTreeNode;

  Lnode -> lchild -> col_beg = Lnode -> col_beg + Lnode -> ncol;
  Lnode -> rchild -> col_beg = Lnode -> col_beg + Lnode -> ncol;

  assert(HODLRnode -> splitIndex_i - HODLRnode -> min_i + 1 == HODLRnode->topOffDiagU.rows());
  assert(HODLRnode -> topOffDiagRank                    == HODLRnode->topOffDiagU.cols());
  Lnode -> lchild -> nrow = HODLRnode -> splitIndex_i - HODLRnode -> min_i + 1;
  Lnode -> lchild -> ncol = HODLRnode -> topOffDiagRank;

  assert(HODLRnode -> max_i - HODLRnode -> splitIndex_i == HODLRnode->bottOffDiagU.rows());
  assert(HODLRnode -> bottOffDiagRank               == HODLRnode->bottOffDiagU.cols());
  Lnode -> rchild -> nrow = HODLRnode -> max_i - HODLRnode -> splitIndex_i;
  Lnode -> rchild -> ncol = HODLRnode -> bottOffDiagRank;
  
  if (HODLRnode -> left -> isLeaf == false)
    create_legion_tree(HODLRnode->left, Lnode->lchild);
  else
    Lnode -> lchild -> lchild = Lnode -> lchild -> rchild = NULL;
  
  if (HODLRnode -> right -> isLeaf == false)
    create_legion_tree(HODLRnode->right, Lnode->rchild);
  else
    Lnode -> rchild -> lchild = Lnode -> rchild -> rchild = NULL;  
}
*/

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

    vnode->matrix = new LeafData;
    vnode->matrix->rows = vrow;
    vnode->matrix->cols = vcol;
    create_matrix(vnode->matrix->data, vrow, vcol, ctx, runtime);

    // create K matrix
    vnode->kmat = new LeafData;
    vnode->kmat->rows = vnode->nrow;
    vnode->kmat->cols = max_row_size(vnode);
    create_matrix(vnode->kmat->data, vnode->kmat->rows, vnode->kmat->cols, ctx, runtime);
  }
}


/*
template <MatrixType matType>
void append_matrix(HODLR_Tree::node *HODLRnode, FSTreeNode *node, Eigen::MatrixXd & pmat) {

  if (node->lchild == NULL && node->rchild == NULL) {
    assert(HODLRnode->isLeaf == true);
    return;
  }

  Eigen::MatrixXd lmat, rmat;
  if (matType == UMatrix) {
    lmat = HODLRnode->topOffDiagU;
    rmat = HODLRnode->bottOffDiagU;
  }
  else if (matType == VMatrix) {
    rmat = HODLRnode->topOffDiagV;
    lmat = HODLRnode->bottOffDiagV;
  } else assert(false);
  
  assert(node->lchild->nrow == lmat.rows());
  assert(node->lchild->ncol == lmat.cols());
  assert(node->rchild->nrow == rmat.rows());
  assert(node->rchild->ncol == rmat.cols());
  
  pmat.block(node->lchild->row_beg, node->lchild->col_beg, node->lchild->nrow, node->lchild->ncol) = lmat;
  pmat.block(node->rchild->row_beg, node->rchild->col_beg, node->rchild->nrow, node->rchild->ncol) = rmat;

  append_matrix<matType>(HODLRnode->left,  node->lchild, pmat);
  append_matrix<matType>(HODLRnode->right, node->rchild, pmat);
}


// the left most columns in umat are the right hand side
void LR_Matrix::init_Umat(HODLR_Tree::node *HODLRnode, FSTreeNode *unode, Eigen::MatrixXd &umat) {

  if (unode->isLegionLeaf == true) {

    //assert(vnode->isLegionLeaf == true);
    Eigen::MatrixXd u(unode->matrix->rows, unode->matrix->cols);
    //Eigen::MatrixXd v(vnode->matrix->rows, vnode->matrix->cols);

    assert(unode->col_beg + unode->ncol == umat.cols());
    //assert(vnode->col_beg + vnode->ncol == vmat.cols());
    assert(unode->nrow == umat.rows());
    //assert(vnode->nrow == vmat.rows());
    
    u.leftCols(unode->col_beg + unode->ncol) = umat;
    //v.leftCols(vnode->col_beg + vnode->ncol) = ;
    append_matrix<UMatrix>(HODLRnode, unode, u);
    //append_matrix(HODLRnode, HODLRnode->bottOffDiagV, HODLRnode->topOffDiagV,  vnode, v);

    //saveMatrixToText(u, "BigU2.txt");
    unode->matrix->set_matrix_data(u, ctx, runtime);
    //vnode->matrix->set_matrix_data(v, ctx, runtime);
    return;
  }
  
  Eigen::MatrixXd ul, ur;
  //Eigen::MatrixXd vl, vr;
  
  accumulate_matrix(umat, HODLRnode->topOffDiagU,  HODLRnode->bottOffDiagU, ul, unode->lchild, ur, unode->rchild);
  //accumulate_matrix(vmat, HODLRnode->bottOffDiagV, HODLRnode->topOffDiagV,  vl, vnode->lchild, vr, vnode->rchild);
  init_Umat(HODLRnode->left,  unode->lchild, ul);
  init_Umat(HODLRnode->right, unode->rchild, ur);
}


void LR_Matrix::init_Vmat(HODLR_Tree::node *HODLRnode, FSTreeNode *node, Eigen::MatrixXd & vmat) {

  if (node->Hmat != NULL) // skip vroot
    set_Hmatrix_data(node->Hmat, vmat);
  
  if (node->isLegionLeaf == true) {

    assert(node->matrix->rows == vmat.rows());
    //assert(node->ncol == vmat.cols());    
    //assert(node->nrow == vmat.rows());
    //v.leftCols(node->ncol) = vmat;
    
    Eigen::MatrixXd v(node->matrix->rows, node->matrix->cols);    
    append_matrix<VMatrix>(HODLRnode, node, v);
    node->matrix->set_matrix_data(v, ctx, runtime);

    Eigen::MatrixXd k(node->kmat->rows, node->kmat->cols);
    fill_kmat(HODLRnode, node, k);
    node->kmat->set_matrix_data(k, ctx, runtime);

  } else {
    init_Vmat(HODLRnode->left,  node->lchild, HODLRnode->bottOffDiagV);
    init_Vmat(HODLRnode->right, node->rchild, HODLRnode->topOffDiagV);
  }
}


void LR_Matrix::accumulate_matrix(Eigen::MatrixXd &pmat, Eigen::MatrixXd &LR_lmat, Eigen::MatrixXd &LR_rmat, Eigen::MatrixXd &lmat, FSTreeNode *lchild, Eigen::MatrixXd &rmat, FSTreeNode *rchild) {
  
  assert(lchild->col_beg == rchild->col_beg);
  assert(lchild->col_beg == pmat.cols());
  int col_beg = rchild->col_beg;

  lmat.resize(lchild->nrow, col_beg + lchild->ncol);
  rmat.resize(rchild->nrow, col_beg + rchild->ncol);

  assert(lchild->nrow + rchild->nrow == pmat.rows());
  lmat.leftCols(col_beg) = pmat.topRows(lchild->nrow);
  rmat.leftCols(col_beg) = pmat.bottomRows(rchild->nrow);

  assert(lchild->nrow == LR_lmat.rows());
  assert(rchild->nrow == LR_rmat.rows());
  assert(lchild->ncol == LR_lmat.cols());
  assert(rchild->ncol == LR_rmat.cols());
  lmat.rightCols(lchild->ncol) = LR_lmat;
  rmat.rightCols(rchild->ncol) = LR_rmat;
}


void LR_Matrix::fill_kmat(HODLR_Tree::node * HODLRnode, FSTreeNode * vnode, Eigen::MatrixXd & k) {

  if (HODLRnode->isLeaf == true) {

    assert(vnode->lchild == NULL && vnode->rchild == NULL);

    int ksize = vnode->nrow;
    assert(ksize == HODLRnode->leafMatrix.rows());

    k.block(vnode->row_beg, 0, ksize, ksize) = HODLRnode->leafMatrix;
    return;
  }

  fill_kmat(HODLRnode->left,  vnode->lchild, k);
  fill_kmat(HODLRnode->right, vnode->rchild, k);
}
*/

void LR_Matrix::fill_circulant_kmat(FSTreeNode * vnode, int row_beg_glo, int r, double diag, double *Kmat, int LD) {

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

    free(U);
    return;
  }

  fill_circulant_kmat(vnode->lchild, row_beg_glo, r, diag, Kmat, LD);
  fill_circulant_kmat(vnode->rchild, row_beg_glo, r, diag, Kmat, LD);
}


/*
  void LR_Matrix::init_kmat(HODLR_Tree::node * HODLRnode, FSTreeNode * vnode) {

  if (HODLRnode->isLeaf == true) {

  assert(vnode->lchild == NULL && vnode->rchild == NULL);
  assert(vnode->kmat->rows == HODLRnode->leafMatrix.rows());
  assert(vnode->kmat->cols == HODLRnode->leafMatrix.cols());

  vnode->kmat->set_matrix_data(HODLRnode->leafMatrix, ctx, runtime);
  return;
  }

  init_kmat(HODLRnode->left,  vnode->lchild);
  init_kmat(HODLRnode->right, vnode->rchild);
  }
*/


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


/*
void LR_Matrix::set_Hmatrix_data(FSTreeNode * Hmat, Eigen::MatrixXd & vmat) {

  if (Hmat->lchild == NULL && Hmat->rchild == NULL) {

    assert(Hmat->row_beg < vmat.rows());
    int nrow = Hmat->matrix->rows;
    int ncol = Hmat->matrix->cols;
    int row_beg = Hmat->row_beg;
    int col_beg = 0;
    Eigen::MatrixXd v_block = vmat.block(row_beg, col_beg, nrow, ncol);
    Hmat->matrix->set_matrix_data(v_block, ctx, runtime);
    
  } else {
    set_Hmatrix_data(Hmat->lchild, vmat);
    set_Hmatrix_data(Hmat->rchild, vmat);
  }  
}
*/

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


void LR_Matrix::print_Vmat(FSTreeNode *node, std::string filename) {

  if (node == vroot)
    save_region(node, filename, ctx, runtime); // print big V matrix

  if (node->Hmat != NULL)
    save_region(node->Hmat, filename, ctx, runtime);
  else if (node != vroot)
    return;
    
  print_Vmat(node->lchild, filename);
  print_Vmat(node->rchild, filename);
}



/*
  FSTreeNode * FSTreeNode::copy_node() {

  FSTreeNode *temp = new FSTreeNode;
  temp -> row_beg = row_beg;
  temp -> col_beg = col_beg;
  temp -> nrow = nrow;
  temp -> ncol = ncol;
  
  temp -> isLegionLeaf = isLegionLeaf;
  
  return temp;
  }
*/

void LeafData::set_circulant_matrix_data(int col_beg, int row_beg, int r, Context ctx, HighLevelRuntime *runtime) {

  
  InlineLauncher launcher(RegionRequirement(data, WRITE_DISCARD, EXCLUSIVE, data));
  
  launcher.requirement.add_field(FID_X);
  
  PhysicalRegion region = runtime->map_region(ctx, launcher);
  
  RegionAccessor<AccessorType::Generic, double> acc = 
    region.get_field_accessor(FID_X).typeify<double>();
 
  Domain dom = runtime->get_index_space_domain(ctx, data.get_index_space());
  Rect<2> rect = dom.get_rect<2>();

  assert( (rect.dim_size(0) - col_beg) % r == 0 );
  GenericPointInRectIterator<2> pir(rect);

  for (int j=0; j<rect.dim_size(0) - col_beg; j++) {
    for (int i=0; i<rect.dim_size(1); i++, pir++) {
      int value = (j+i+row_beg)%r;
      int pt[2] = {j+col_beg, i};
      acc.write(DomainPoint::from_point<2>( Point<2> (pt) ), value);
    }
  }
  
  runtime->unmap_region(ctx, region);
}


/*
// TODO: implement col_beg
void LeafData::set_matrix_data(Eigen::MatrixXd &mat, Context ctx, HighLevelRuntime *runtime, int col_beg) {

  InlineLauncher launcher(RegionRequirement(data, WRITE_DISCARD, EXCLUSIVE, data));
  
  launcher.requirement.add_field(FID_X);
  
  PhysicalRegion region = runtime->map_region(ctx, launcher);
  
  RegionAccessor<AccessorType::Generic, double> acc = 
    region.get_field_accessor(FID_X).typeify<double>();
 
  Domain dom = runtime->get_index_space_domain(ctx, data.get_index_space());
  Rect<2> rect = dom.get_rect<2>();

  assert(mat.rows() == rect.dim_size(1));
  assert(mat.cols() <= rect.dim_size(0));
  
  //for (; pir; pir++) {
  //GenericPointInRectIterator<2> pir(rect);
  //for (int i=0; i<mat.rows()*mat.cols(); i++, pir++) {
  for (int j=0; j<mat.cols(); j++) {
    for (int i=0; i<mat.rows(); i++) {

      //assert(pir.p[1] < mat.rows());
      //assert(pir.p[0] < mat.cols());
      int pt[2] = {j, i};
      //acc.write(DomainPoint::from_point<2>(pir.p), mat(pir.p[1], pir.p[0]));
      acc.write(DomainPoint::from_point<2>( Point<2> (pt) ), mat(i, j));
    }
  }
  
  runtime->unmap_region(ctx, region);
}
*/


void LeafData::set_matrix_data(double *mat, int rhs_rows, int rhs_cols, Context ctx, HighLevelRuntime *runtime, int row_beg) {

  InlineLauncher launcher(RegionRequirement(data, WRITE_DISCARD, EXCLUSIVE, data));
  
  launcher.requirement.add_field(FID_X);
  
  PhysicalRegion region = runtime->map_region(ctx, launcher);
  
  RegionAccessor<AccessorType::Generic, double> acc = 
    region.get_field_accessor(FID_X).typeify<double>();
 
  Domain dom = runtime->get_index_space_domain(ctx, data.get_index_space());
  Rect<2> rect = dom.get_rect<2>();

  //assert(nrow == rect.dim_size(1));
  int nrow = rect.dim_size(1);
  assert(rhs_cols <= rect.dim_size(0));
  
  for (int j=0; j<rhs_cols; j++) {
    for (int i=0; i<nrow; i++) {
      int pt[2] = {j, i};
      acc.write(DomainPoint::from_point<2>( Point<2> (pt) ), mat[row_beg+i+j*rhs_rows]);
    }
  }
  
  runtime->unmap_region(ctx, region);
}


void create_matrix(LogicalRegion & matrix, int nrow, int ncol, Context ctx, HighLevelRuntime *runtime) {
  
  int lower[2] = {0,      0};
  int upper[2] = {ncol-1, nrow-1}; // note the order and inclusive bound
  Rect<2> rect((Point<2>(lower)), (Point<2>(upper)));
  IndexSpace is = runtime->create_index_space(ctx, Domain::from_rect<2>(rect));
  FieldSpace fs = runtime->create_field_space(ctx);
  FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
  allocator.allocate_field(sizeof(double), FID_X);
  matrix = runtime->create_logical_region(ctx, is, fs);
  assert(matrix != LogicalRegion::NO_REGION);
}


void set_element(double x, LogicalRegion &matrix, Context ctx, HighLevelRuntime *runtime) {

  RegionRequirement req(matrix, WRITE_DISCARD, EXCLUSIVE, matrix);
  req.add_field(FID_X);

  InlineLauncher init(req);
  PhysicalRegion init_region = runtime->map_region(ctx, init);
  init_region.wait_until_valid();

  RegionAccessor<AccessorType::Generic, double> acc =
    init_region.get_field_accessor(FID_X).typeify<double>();

  Domain dom = runtime->get_index_space_domain(ctx, matrix.get_index_space());
  Rect<2> rect = dom.get_rect<2>();
  for (GenericPointInRectIterator<2> pir(rect); pir; pir++)
    acc.write(DomainPoint::from_point<2>(pir.p), x);

  runtime->unmap_region(ctx, init_region);
}


void scale_matrix(double beta, LogicalRegion &matrix, Context ctx, HighLevelRuntime *runtime) {

  RegionRequirement req(matrix, READ_WRITE, EXCLUSIVE, matrix);
  req.add_field(FID_X);

  InlineLauncher init(req);
  PhysicalRegion init_region = runtime->map_region(ctx, init);
  init_region.wait_until_valid();

  RegionAccessor<AccessorType::Generic, double> acc =
    init_region.get_field_accessor(FID_X).typeify<double>();

  Domain dom = runtime->get_index_space_domain(ctx, matrix.get_index_space());
  Rect<2> rect = dom.get_rect<2>();
  for (GenericPointInRectIterator<2> pir(rect); pir; pir++) {
    double x = acc.read(DomainPoint::from_point<2>(pir.p));
    acc.write(DomainPoint::from_point<2>(pir.p), beta * x);
  }

  runtime->unmap_region(ctx, init_region);
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

  int nrow = rect.dim_size(1);

  //int ncol = rect.dim_size(0);


  Rect<2> subrect;
  ByteOffset offsets[2];

  double *ptr = acc.raw_rect_ptr<2>(rect, subrect, offsets);
  assert(rect == subrect);

  double *x = ptr + col_beg * rect.dim_size(0);

  //GenericPointInRectIterator<2> pir(rect);

  std::ofstream outputFile(filename.c_str(), std::ios_base::app);
  outputFile<<nrow<<std::endl;
  outputFile<<ncol<<std::endl;

  for (int i=0; i<nrow; i++) {
    for (int j=0; j<ncol; j++) {
      //assert(pir.p[1] < nrow);
      //assert(pir.p[0] < ncol);

      //double x = acc.read(DomainPoint::from_point<2>(pir.p));
      //outputFile << x << '\t';
      //pir++;
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

  int nrow = rect.dim_size(1);
  int ncol = rect.dim_size(0);

  GenericPointInRectIterator<2> pir(rect);

  std::ofstream outputFile(filename.c_str(), std::ios_base::app);
  outputFile<<nrow<<std::endl;
  outputFile<<ncol<<std::endl;

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


void save_region(FSTreeNode * node, range rg, std::string filename, Context ctx, HighLevelRuntime *runtime) {

  if (node->isLegionLeaf == true) {
    save_region(node->matrix->data, rg.col_beg, rg.ncol, filename, ctx, runtime);
    
    
  } else {
    save_region(node->lchild, rg, filename, ctx, runtime);
    save_region(node->rchild, rg, filename, ctx, runtime);
  }  
}


void save_region(FSTreeNode * node, std::string filename, Context ctx, HighLevelRuntime *runtime) {

  if (node->isLegionLeaf == true) {
    //save_region(node->matrix->data, filename, ctx, runtime);
    
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
  //assert(task->arglen == 0);
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


  int nrow = rect_u.dim_size(1);
  int ncol = rect_u.dim_size(0);

  GenericPointInRectIterator<2> pir(rect_u);

  std::ofstream outputFile(filename, std::ios_base::app);
  outputFile<<nrow<<std::endl;
  outputFile<<ncol<<std::endl;

  for (int i=0; i<nrow; i++) {
    for (int j=0; j<ncol; j++) {
      assert(pir.p[1] < nrow);
      assert(pir.p[0] < ncol);

      double x = acc_u.read(DomainPoint::from_point<2>(pir.p));
      outputFile << x << '\t';
      pir++;
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


/* --- save solution to matrix --- */
/*
void get_soln_from_region(Eigen::MatrixXd &soln, FSTreeNode *node, Context ctx, HighLevelRuntime *runtime, int row_beg) {

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

    int nrow = rect.dim_size(1);
    int ncol = rect.dim_size(0);

    Rect<2> subrect;
    ByteOffset offsets[2];

    double *ptr = acc.raw_rect_ptr<2>(rect, subrect, offsets);
    assert(rect == subrect);

    int cols = soln.cols();
    for (int j=0; j<cols; j++)
      for (int i=0; i<nrow; i++) {
	soln(i + row_beg, j) = ptr[i+j*nrow];
      }

    runtime->unmap_region(ctx, region);
  
  } else {
    get_soln_from_region(soln, node->lchild, ctx, runtime, row_beg);
    get_soln_from_region(soln, node->rchild, ctx, runtime, row_beg+node->lchild->nrow);
  }
}
*/


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

    int nrow = rect.dim_size(1);
    int ncol = rect.dim_size(0);

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

