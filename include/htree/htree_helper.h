#ifndef _HTREE_HELPER
#define _HTREE_HELPER

#include "hodlr_matrix.h"


void build_subtree(FSTreeNode *node, int row_beg = 0);

int count_matrix_column(const FSTreeNode *node, int col_size=0);

int max_row_size(const FSTreeNode *);

int count_leaf(const FSTreeNode *node);

//int count_launch_node(FSTreeNode *node);

// TODO: move into LMatrix class
void create_matrix
  (LMatrix *(&matrix), int nrow, int ncol,
   Context ctx, HighLevelRuntime *runtime);


int  tree_to_array(const FSTreeNode *, FSTreeNode *, int);
void tree_to_array(const FSTreeNode *, FSTreeNode *, int, int);
void array_to_tree(FSTreeNode *, int);
void array_to_tree(FSTreeNode *, int, int);


void save_HodlrMatrix
(FSTreeNode * node, std::string filename,
 Context ctx, HighLevelRuntime *runtime,
 Range rg, bool print_seed=false);



//void print_Vmat
//(FSTreeNode *node, std::string filename);


#endif // _HTREE_HELPER
