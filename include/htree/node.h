#ifndef _HTREE_HELPER
#define _HTREE_HELPER

#include "range.h"

#include "legion.h"

using namespace LegionRuntime::HighLevel;

class LMatrix;

struct Node {
public:
  Node(int nrow=0,
       int ncol=0,
       int row_beg=0,
       int col_beg=0,
       Node *lchild=NULL,
       Node *rchild=NULL,
       Node *Hmat=NULL,
       LMatrix *matrix=NULL,
       LMatrix *kmat=NULL,
       bool isLegionLeaf=false);

  bool is_real_leaf()   const;
  bool is_legion_leaf() const;
  void set_legion_leaf(bool);

  int nrow;    
  int ncol;
  int row_beg; // begin index in the region
  int col_beg;

  Node *lchild;
  Node *rchild;
  Node *Hmat;
  
  LMatrix *lowrank_matrix; // low rank blocks
  LMatrix *dense_matrix;   // dense blocks

private:
  bool isLegionLeaf;
};

void build_subtree(Node *node, int row_beg = 0);

int count_matrix_column(const Node *node, int col_size=0);

int max_row_size(const Node *);

int count_leaf(const Node *node);

int  tree_to_array(const Node *, Node *, int);
void tree_to_array(const Node *, Node *, int, int);
void array_to_tree(Node *, int);
void array_to_tree(Node *, int, int);

#endif // _HTREE_HELPER
