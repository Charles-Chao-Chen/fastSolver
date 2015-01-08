#include "Htree.h"
#include "htreeHelper.h"
#include "initMatrixTasks.h"
#include "lapack_blas.h"
#include "macros.h"


FSTreeNode::FSTreeNode(int nrow, int ncol,
		       int row_beg, int col_beg,
		       FSTreeNode *lchild,
		       FSTreeNode *rchild,
		       FSTreeNode *Hmat,
		       LMatrix *matrix,
		       LMatrix *kmat,
		       bool isLegionLeaf):
  nrow(nrow), ncol(ncol), row_beg(row_beg), col_beg(col_beg),
  lchild(lchild), rchild(rchild), Hmat(Hmat),
  matrix(matrix), kmat(kmat),
  isLegionLeaf(isLegionLeaf) {}

bool FSTreeNode::isRealLeaf() {
  return (lchild == NULL)
    &&   (rchild == NULL);
}

static void
create_balanced_tree(FSTreeNode *, int, int);
static int
mark_legion_leaf(FSTreeNode *node, int threshold);
static int
create_legion_node(FSTreeNode *node, Context ctx,
		   HighLevelRuntime *runtime);

/* Input:
 *   N - problem size
 *   r - every off-diagonal block has the same rank
 *   rhs_cols  - column size of the right hand side
 *   threshold - size of dense blocks at the leaf level
 */
void
LR_Matrix::create_tree(int N, int threshold, int rhs_cols,
		       int rank, int nleaf_per_legion_node,
		       Context ctx, HighLevelRuntime *runtime) {

  this->rank = rank;
  this->rhs_rows = N;
  this->rhs_cols = rhs_cols;

  uroot  = new FSTreeNode(N, rhs_cols);

  // create the H-tree for U matrices
  create_balanced_tree(uroot, rank, threshold);

  // set legion leaf for granularity control
  mark_legion_leaf(uroot, nleaf_per_legion_node);
  
  // create region at legion leaf
  set_num_leaf( create_legion_node(uroot, ctx, runtime) );
  
  //print_legion_tree(uroot);
  // postpone creating V tree after setting the legion leaf

  
  vroot  = new FSTreeNode;
  vroot -> nrow = uroot->nrow;
  vroot -> ncol = 0;
  create_vnode_from_unode(uroot, vroot, ctx, runtime);

  //print_legion_tree(vroot);
}


/* Implicit input: a rank R matrix U U^T plus diagonal (to make it non-singular)
 *   if U has a specific pattern, it does not require be stored as a whole matrix. E.g. U(:,1) = (1:N)%m, U(:,2) = (2:N+1)%m
 *
 * Args:
 *   diag - the diagonal entry for the dense block
 *   RHS  - right hand side of the problem
 */
void
LR_Matrix::init_circulant_matrix(double diag, int num_node,
				 Context ctx, HighLevelRuntime
				 *runtime)
{

  Range tag(num_node);
  init_Umat(uroot, tag, ctx, runtime);       // row_beg = 0
  init_Vmat(vroot, diag, tag, ctx, runtime); // row_beg = 0

  //print_legion_tree(uroot);
  //print_legion_tree(vroot);
}


/*static*/ void
create_balanced_tree(FSTreeNode *node, int rank, int threshold) {

  int N = node->nrow;
  if (N > threshold) { // sub-divide the matrix,
                       // otherwise it is a dense block

    node->lchild = new FSTreeNode;
    node->rchild = new FSTreeNode;

    node->lchild->nrow = N/2;
    node->rchild->nrow = N - N/2;
    
    node->lchild->ncol = rank;
    node->rchild->ncol = rank;

    node->lchild->col_beg = node->col_beg + node->ncol;
    node->rchild->col_beg = node->col_beg + node->ncol;

    // recursive call
    create_balanced_tree(node->lchild, rank, threshold);
    create_balanced_tree(node->rchild, rank, threshold);
    
  } else {
    assert(N > rank); // assume the size of dense blocks is larger
                      // than the rank
  }

}


void LR_Matrix::
init_right_hand_side(int rand_seed, int ncol, int node_num,
		     Context ctx, HighLevelRuntime *runtime)
{
  std::cout << "rhs_cols: " << rhs_cols << std::endl;
  //assert(rhs_cols == 1);
  Range tag(node_num);
  init_RHS(uroot, rand_seed, ncol, tag,
	   ctx, runtime /*, row_beg = 0*/); 
}


void LR_Matrix::
init_RHS(FSTreeNode *node, int rand_seed, int ncol, Range tag,
	 Context ctx, HighLevelRuntime *runtime, int row_beg) {

  if (node->isLegionLeaf == true) {
    assert(node->matrix != NULL);

    //typename
    InitRHSTask::TaskArgs args = {rand_seed, ncol};
    InitRHSTask launcher(TaskArgument(&args, sizeof(args)),
			 Predicate::TRUE_PRED,
			 0,
			 tag.begin);
    
    launcher.add_region_requirement(
	       RegionRequirement(node->matrix->data,
				 WRITE_DISCARD,
				 EXCLUSIVE,
				 node->matrix->data).
	       add_field(FID_X));
    runtime->execute_task(ctx, launcher);
    
  } else { // recursively split RHS
    
    int   half = tag.size/2;
    Range ltag = tag.lchild();
    Range rtag = tag.rchild();
    init_RHS(node->lchild, rand_seed, ncol, ltag,
	     ctx, runtime, row_beg);
    init_RHS(node->rchild, rand_seed, ncol, rtag,
	     ctx, runtime, row_beg+node->lchild->nrow);
  }  
}


void LR_Matrix::
init_Umat(FSTreeNode *node, Range tag, Context ctx,
	  HighLevelRuntime *runtime, int row_beg) {

  if (node->isLegionLeaf == true) {

    assert(node->matrix != NULL); // initialize region here
    node->matrix->init_circulant_matrix(rhs_cols, row_beg,
					rank, tag, ctx, runtime);
  } else {
    int   half = tag.size/2;
    Range ltag = tag.lchild();
    Range rtag = tag.rchild();
    init_Umat(node->lchild, ltag, ctx, runtime, row_beg);
    init_Umat(node->rchild, rtag, ctx, runtime, row_beg +
	      node->lchild->nrow);
  }  
}


static void
init_circulant_Kmat(FSTreeNode *V_legion_leaf, int row_beg_glo,
		    int rank, double diag, Range mapping_tag,
		    Context ctx, HighLevelRuntime *runtime);

void LR_Matrix::
init_Vmat(FSTreeNode *node, double diag, Range tag,
	  Context ctx, HighLevelRuntime *runtime,
	  int row_beg) {

  if (node->Hmat != NULL) // skip vroot
    set_circulant_Hmatrix_data(node->Hmat, tag,
			       ctx, runtime, row_beg);

  if (node->isLegionLeaf == true) {

    // init V
    // when the legion leaf is the real leaf, there is no data here.
    if (node->matrix->cols > 0)
      node->matrix->init_circulant_matrix(0, row_beg, rank,
					  tag, ctx, runtime);

    // init K
    int nrow = node->kmat->rows;
    int ncol = node->kmat->cols;
    init_circulant_Kmat(node, row_beg, rank, diag, tag, ctx, runtime);
    
  } else {
    int   half = tag.size/2;
    Range ltag = tag.lchild();
    Range rtag = tag.rchild();
    init_Vmat(node->lchild, diag, ltag,
	      ctx, runtime, row_beg);
    init_Vmat(node->rchild, diag, rtag,
	      ctx, runtime, row_beg+node->lchild->nrow);
  }
}


void
init_circulant_Kmat(FSTreeNode *V_legion_leaf, int row_beg_glo,
		    int rank, double diag, Range mapping_tag,
		    Context ctx, HighLevelRuntime *runtime)
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


// this function picks legion leaf nodes as those having the number of threshold real matrix leaves.
// when threshold = 1, the legion leaf and real matrix leaf coinside.
// nLegionLeaf records the number of legion leaves as an indicator of the number of leaf tasks.
/* static */ int
mark_legion_leaf(FSTreeNode *node, int threshold) {

  int nRealLeaf;  
  if (node->isRealLeaf()) { // real matrix leaf
    nRealLeaf = 1;
  } else {
    int nl = mark_legion_leaf(node->lchild, threshold);
    int nr = mark_legion_leaf(node->rchild, threshold);
    nRealLeaf = nl + nr;
  }

  // mark "Legion Leaf" on all leaves from the legion leaf level
  // (or lower levels)
  node->isLegionLeaf = (nRealLeaf > threshold) ? false : true;

  return nRealLeaf;
}


static void
create_region(FSTreeNode *node, Context ctx,
	      HighLevelRuntime *runtime);

/* static */ int
create_legion_node(FSTreeNode *node, Context ctx,
		   HighLevelRuntime *runtime) {
  // count the number of legion leaf
  int nleaf;
  if (node->isLegionLeaf == false) {
    int nl = create_legion_node(node->lchild, ctx, runtime);
    int nr = create_legion_node(node->rchild, ctx, runtime);
    nleaf = nl + nr;
  } else {
    create_region(node, ctx, runtime);
    build_subtree(node);
    nleaf = 1;
  }
  return nleaf;
}


/* static */ void
create_region(FSTreeNode *node, Context ctx,
	      HighLevelRuntime *runtime) {

  assert(node->isLegionLeaf == true);
  int row_size = node->nrow;
  
  // adding column number above and below legion node
  int col_size = node->col_beg + count_matrix_column(node);
  //printf("row_size: %d, col_size: %d.\n", row_size, col_size);

  create_matrix(node->matrix, row_size, col_size, ctx, runtime);
  assert(node->matrix != NULL);
}


void LR_Matrix::
create_vnode_from_unode(FSTreeNode *unode, FSTreeNode *vnode,
			Context ctx, HighLevelRuntime *runtime) {

  // create V tree
  if (!unode->isRealLeaf()) {

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
      
    create_vnode_from_unode(unode->lchild, vnode->lchild,
			    ctx, runtime);
    create_vnode_from_unode(unode->rchild, vnode->rchild,
			    ctx, runtime);
    
  } else {
    vnode -> lchild = NULL;
    vnode -> rchild = NULL;

    if (unode -> isLegionLeaf == true) {
      vnode -> isLegionLeaf = true;
    }
  }


  // create H-tiled matrices for two children
  // including Legion leaf
  if (unode -> isLegionLeaf == false) {
    
    vnode -> lchild -> Hmat =  new FSTreeNode;
    vnode -> rchild -> Hmat =  new FSTreeNode;

    vnode -> lchild -> Hmat -> nrow = vnode -> lchild -> nrow;
    vnode -> rchild -> Hmat -> nrow = vnode -> rchild -> nrow;

    vnode -> lchild -> Hmat -> ncol = vnode -> lchild -> ncol;
    vnode -> rchild -> Hmat -> ncol = vnode -> rchild -> ncol;
    
    create_Hmatrix(vnode->lchild,
		   vnode->lchild->Hmat,
		   vnode->lchild->ncol,
		   ctx, runtime);
    create_Hmatrix(vnode->rchild,
		   vnode->rchild->Hmat,
		   vnode->rchild->ncol,
		   ctx, runtime);
    
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

    // when the legion leaf is the real leaf, there is
    // no data here.
    create_matrix(vnode->matrix, vrow, vcol, ctx, runtime);
 
    // create K matrix
    create_matrix(vnode->kmat, vnode->nrow, max_row_size(vnode), ctx, runtime);
  }
}


void LR_Matrix::
create_Hmatrix(FSTreeNode *node, FSTreeNode * Hmat, int ncol,
	       Context ctx, HighLevelRuntime *runtime) {

  if (node->isLegionLeaf == true) {

    Hmat->nrow = node->nrow;
    Hmat->ncol = node->ncol;

    Hmat->isLegionLeaf = true;
    create_matrix(Hmat -> matrix,
		  node -> nrow,
		  ncol,
		  ctx, runtime);
 
  } else {
    
    Hmat->lchild = new FSTreeNode;
    Hmat->rchild = new FSTreeNode;

    // to be used in initialization
    Hmat->lchild->row_beg = Hmat->row_beg;
    Hmat->rchild->row_beg = Hmat->row_beg + node->lchild->nrow;
    
    create_Hmatrix(node->lchild, Hmat->lchild, ncol,
		   ctx, runtime);
    create_Hmatrix(node->rchild, Hmat->rchild, ncol,
		   ctx, runtime);
  }
}


void LR_Matrix::
set_circulant_Hmatrix_data(FSTreeNode * Hmat, Range tag,
			   Context ctx, HighLevelRuntime *runtime,
			   int row_beg) {

  if (Hmat->isRealLeaf()) {

    int glo = row_beg;
    int loc = Hmat->row_beg;
    assert(Hmat->ncol == rank);
    Hmat->matrix->init_circulant_matrix(0, glo + loc, rank,
					tag, ctx, runtime);
    
  } else {
    int   half = tag.size/2;
    Range ltag = tag.lchild();
    Range rtag = tag.rchild();
    set_circulant_Hmatrix_data(Hmat->lchild, ltag,
			       ctx, runtime, row_beg);
    set_circulant_Hmatrix_data(Hmat->rchild, rtag,
			       ctx, runtime, row_beg);
  }  
}


void LMatrix::
init_circulant_matrix(int col_beg, int row_beg, int r, Range tag,
		      Context ctx, HighLevelRuntime *runtime) {    

  InitCirculantMatrixTask::TaskArgs
    args = {col_beg, row_beg, r};
  InitCirculantMatrixTask launcher(TaskArgument(&args,
						sizeof(args)),
				   Predicate::TRUE_PRED,
				   0,
				   tag.begin);
  launcher.add_region_requirement(RegionRequirement(data,
						    READ_WRITE,
						    EXCLUSIVE,
						    data));
  launcher.region_requirements[0].add_field(FID_X);
  runtime->execute_task(ctx, launcher);
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


void fill_circulant_Kmat(FSTreeNode * vnode, int row_beg_glo, int r, double diag, double *Kmat, int LD) {

  if (vnode->isRealLeaf()) {

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

    //printf("After init Kmat.\n");
    
    free(U);
    return;
  }

  fill_circulant_Kmat(vnode->lchild, row_beg_glo, r, diag, Kmat, LD);
  fill_circulant_Kmat(vnode->rchild, row_beg_glo, r, diag, Kmat, LD);
}
