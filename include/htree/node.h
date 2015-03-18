#ifndef _HTREE_HELPER
#define _HTREE_HELPER

#include "hodlr_matrix.h"

void build_subtree(Node *node, int row_beg = 0);

int count_matrix_column(const Node *node, int col_size=0);

int max_row_size(const Node *);

int count_leaf(const Node *node);

// TODO: move into LMatrix class
void create_matrix
  (LMatrix *(&matrix), int nrow, int ncol,
   Context ctx, HighLevelRuntime *runtime);

int  tree_to_array(const Node *, Node *, int);
void tree_to_array(const Node *, Node *, int, int);
void array_to_tree(Node *, int);
void array_to_tree(Node *, int, int);

void save_HodlrMatrix
(Node * node, std::string filename,
 Context ctx, HighLevelRuntime *runtime,
 Range rg, bool print_seed=false);

#endif // _HTREE_HELPER
