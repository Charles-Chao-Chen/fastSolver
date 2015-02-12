#include "htree_helper.h"
#include "macros.h"

// this function computes the begining row index in region of
// Legion leaf i.e. building the subtree with the legion node
// as the root. This subtree is used in leaf solve (serial) task
void build_subtree(FSTreeNode *node, int row_beg) {

  node->row_beg = row_beg;
  
  if (node->is_real_leaf()) { // real matrix leaf
    return;
  } else {
    build_subtree(node->lchild, row_beg);
    build_subtree(node->rchild, row_beg + node->lchild->nrow);
  }
}


int count_matrix_column(FSTreeNode *node, int col_size) {

  col_size += node->ncol;
  
  if (node->is_real_leaf())
    return col_size;
  else {
    int nl = count_matrix_column(node->lchild, col_size);
    int nr = count_matrix_column(node->rchild, col_size);
    return std::max(nl, nr);
  }
}


int max_row_size(FSTreeNode * vnode) {

  if (vnode->is_real_leaf()) {
    return vnode->nrow;
  }

  int m1 = max_row_size(vnode->lchild);
  int m2 = max_row_size(vnode->rchild);
  
  return std::max(m1, m2);
}



int
tree_to_array(FSTreeNode * leaf, FSTreeNode * arg, int idx) {

  if (leaf->lchild != NULL && leaf->rchild != NULL) {

    //assert(2*idx+2 < arg.size());
    arg[ 2*idx+1 ] = *(leaf -> lchild);
    arg[ 2*idx+2 ] = *(leaf -> rchild);
    int nl = tree_to_array(leaf->lchild, arg, 2*idx+1);
    int nr = tree_to_array(leaf->rchild, arg, 2*idx+2);
    return nl + nr + 1;
    
  } else return 1;
}


void
tree_to_array(FSTreeNode *tree, FSTreeNode *array, int idx,
	      int shift) {

  if (tree->lchild != NULL && tree->rchild != NULL) {

    //assert(2*idx+2+shift < arg.size());
    array[ 2*idx+1+shift ] = *(tree -> lchild);
    array[ 2*idx+2+shift ] = *(tree -> rchild);
    tree_to_array(tree->lchild, array, 2*idx+1, shift);
    tree_to_array(tree->rchild, array, 2*idx+2, shift); 
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

  array_to_tree(arg, 2*idx+1, shift);
  array_to_tree(arg, 2*idx+2, shift);
}


int count_leaf(FSTreeNode *node) {
  if (node->is_real_leaf())
    return 1;
  else {
    int n1 = count_leaf(node->lchild);
    int n2 = count_leaf(node->rchild);
    return n1+n2;
  }
}


void
create_matrix(LMatrix *(&matrix), int nrow, int ncol,
	      Context ctx, HighLevelRuntime *runtime) {

  // ncol can be 0 for the matrix below legion node
  // in v tree
  assert(nrow > 0);
  matrix = new LMatrix(nrow, ncol);

  int lower[2] = {0,      0};
  int upper[2] = {nrow-1, ncol-1}; // inclusive bound
  Rect<2> rect((Point<2>(lower)), (Point<2>(upper)));
  FieldSpace fs = runtime->create_field_space(ctx);
  IndexSpace is = runtime->
    create_index_space(ctx, Domain::from_rect<2>(rect));
  FieldAllocator allocator = runtime->
    create_field_allocator(ctx, fs);
  allocator.allocate_field(sizeof(double), FID_X);
  matrix->data = runtime->create_logical_region(ctx, is, fs);
  assert(matrix->data != LogicalRegion::NO_REGION);
}


void save_HodlrMatrix
(FSTreeNode * node, std::string filename,
 Context ctx, HighLevelRuntime *runtime, Range rg)
{
  if ( node->is_legion_leaf() ) {
    //save_LMatrix(node->lowrank_matrix, filename, ctx, runtime,rg);
    node->lowrank_matrix->save(filename, ctx, runtime, rg);
  } else {
    save_HodlrMatrix(node->lchild, filename, ctx, runtime, rg);
    save_HodlrMatrix(node->rchild, filename, ctx, runtime, rg);
  }
}


int count_launch_node(FSTreeNode *node) {
  if ( ! node->is_launch_node() ) {
    int nl = count_launch_node(node->lchild);
    int nr = count_launch_node(node->rchild);
    return nl + nr;
  } else {
    return 1;
  }
}


/*
  void HodlrMatrix::print_Vmat(FSTreeNode *node, std::string filename) {

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
*/
