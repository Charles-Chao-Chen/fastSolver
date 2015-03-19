#include "node.h"
#include "macros.h"

Node::Node(int nrow_, int ncol_,
		       int row_beg_, int col_beg_,
		       Node *lchild_,
		       Node *rchild_,
		       Node *Hmat_,
		       LMatrix *matrix_,
		       LMatrix *kmat_,
		       bool isLegionLeaf_):
  nrow(nrow_), ncol(ncol_),
  row_beg(row_beg_), col_beg(col_beg_),
  lchild(lchild_), rchild(rchild_), Hmat(Hmat_),
  lowrank_matrix(matrix_), dense_matrix(kmat_),
  isLegionLeaf(isLegionLeaf_) {}

bool Node::is_real_leaf() const {
  return (lchild == NULL)
    &&   (rchild == NULL);
}

bool Node::is_legion_leaf() const {
  return isLegionLeaf;
}

void Node::set_legion_leaf(bool is) {
  isLegionLeaf = is;
}

// this function computes the begining row index in region of
// Legion leaf i.e. building the subtree with the legion node
// as the root. This subtree is used in leaf solve (serial) task
void build_subtree(Node *node, int row_beg) {

  node->row_beg = row_beg;
  
  if (node->is_real_leaf()) { // real matrix leaf
    return;
  } else {
    build_subtree(node->lchild, row_beg);
    build_subtree(node->rchild, row_beg + node->lchild->nrow);
  }
}

int count_matrix_column(const Node *node, int col_size) {

  col_size += node->ncol;
  
  if (node->is_real_leaf())
    return col_size;
  else {
    int nl = count_matrix_column(node->lchild, col_size);
    int nr = count_matrix_column(node->rchild, col_size);
    return std::max(nl, nr);
  }
}

int max_row_size(const Node * vnode) {

  if (vnode->is_real_leaf()) {
    return vnode->nrow;
  }

  int m1 = max_row_size(vnode->lchild);
  int m2 = max_row_size(vnode->rchild);
  
  return std::max(m1, m2);
}

int tree_to_array(const Node * leaf, Node * arg, int idx) {

  if (leaf->lchild != NULL && leaf->rchild != NULL) {

    //assert(2*idx+2 < arg.size());
    arg[ 2*idx+1 ] = *(leaf -> lchild);
    arg[ 2*idx+2 ] = *(leaf -> rchild);
    int nl = tree_to_array(leaf->lchild, arg, 2*idx+1);
    int nr = tree_to_array(leaf->rchild, arg, 2*idx+2);
    return nl + nr + 1;
    
  } else return 1;
}

void tree_to_array(const Node *tree, Node *array, int idx,
	      int shift) {

  if (tree->lchild != NULL && tree->rchild != NULL) {

    //assert(2*idx+2+shift < arg.size());
    array[ 2*idx+1+shift ] = *(tree -> lchild);
    array[ 2*idx+2+shift ] = *(tree -> rchild);
    tree_to_array(tree->lchild, array, 2*idx+1, shift);
    tree_to_array(tree->rchild, array, 2*idx+2, shift); 
  }
}

void array_to_tree(Node *arg, int idx) {

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

void array_to_tree(Node *arg, int idx, int shift) {

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

int count_leaf(const Node *node) {
  if (node->is_real_leaf())
    return 1;
  else {
    int n1 = count_leaf(node->lchild);
    int n2 = count_leaf(node->rchild);
    return n1+n2;
  }
}

/*
  void HodlrMatrix::print_Vmat(Node *node, std::string filename) {

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
