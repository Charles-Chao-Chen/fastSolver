#include "htreeHelper.h"


// this function computes the begining row index in region of
// Legion leaf i.e. building the subtree with the legion node
// as the root. This subtree is used in leaf solve (serial) task
void build_subtree(FSTreeNode *node, int row_beg) {

  node->row_beg = row_beg;
  
  if (node->lchild == NULL &&
      node->rchild == NULL) { // real matrix leaf
    return;
  } else {
    build_subtree(node->lchild, row_beg);
    build_subtree(node->rchild, row_beg + node->lchild->nrow);
  }
}


int count_column_size(FSTreeNode *node, int col_size) {

  col_size += node->ncol;
  
  //if (node->lchild == NULL &&
  //  node->rchild == NULL) // real matrix leaf
  if (node->isRealLeaf())
    return col_size;
  else {
    int nl = count_column_size(node->lchild, col_size);
    int nr = count_column_size(node->rchild, col_size);
    return std::max(nl, nr);
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
  if (node->lchild == NULL && node->rchild == NULL)
    return 1;
  else {
    int n1 = count_leaf(node->lchild);
    int n2 = count_leaf(node->rchild);
    return n1+n2;
  }
}


/* ---- Range class methods ---- */
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
