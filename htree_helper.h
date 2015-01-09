#ifndef _HTREE_HELPER
#define _HTREE_HELPER

#include "Htree.h"


void build_subtree(FSTreeNode *node, int row_beg = 0);

int count_matrix_column(FSTreeNode *node, int col_size=0);

int max_row_size(FSTreeNode *);

int count_leaf(FSTreeNode *node);

void
create_matrix(LMatrix *(&matrix), int nrow, int ncol,
	      Context ctx, HighLevelRuntime *runtime);

int  tree_to_array(FSTreeNode *, FSTreeNode *, int);
void tree_to_array(FSTreeNode *, FSTreeNode *, int, int);
void array_to_tree(FSTreeNode *, int);
void array_to_tree(FSTreeNode *, int, int);



#endif // _HTREE_HELPER
