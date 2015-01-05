#ifndef _LEGION_TREE_
#define _LEGION_TREE_

#include <string>
#include <fstream>

#include "legion.h"

using namespace LegionRuntime::HighLevel;


#define MAX_TREE_SIZE 15 // used in InitCirculantKmatTask::TaskArgs


void register_Htree_tasks();


enum MatrixType {
  UMatrix,
  VMatrix,
};


struct range {
  int col_beg;
  int ncol;
};


class Range {
 public:
  Range lchild();
  Range rchild();
 public:
  int begin;
  int size;
};


struct ColRange {
  int col_beg;
  int ncol;
};


class LeafData {

public:
 LeafData(int rows=0, int cols=0):
  rows(rows), cols(cols), 
  data(LogicalRegion::NO_REGION) {}

  //~LeafData();

  void
    init_circulant_matrix(int col_beg, int row_beg, int r, Range tag,
			  Context ctx, HighLevelRuntime *runtime);

  int rows;  
  int cols;

  //IndexSpace iSpace;
  //FieldSpace fSpace;
  LogicalRegion data; // Region storing the actual data
    
  // Data at leaf nodes is stored in column major fashion.
  // This allows extracting a given column range.
};


// U and V have the same row structure
struct FSTreeNode {

  FSTreeNode(int nrow=0,
	     int ncol=0,
	     int row_beg=0,
	     int col_beg=0,
	     FSTreeNode *lchild=NULL,
	     FSTreeNode *rchild=NULL,
	     FSTreeNode *Hmat=NULL,
	     LeafData *matrix=NULL,
	     LeafData *kmat=NULL,
	     bool isLegionLeaf=false);
  
  int row_beg; // begin index in the region
  int col_beg;
  int nrow;    
  int ncol;

  FSTreeNode *lchild;
  FSTreeNode *rchild;
  FSTreeNode *Hmat;
  
  LeafData *matrix; // low rank blocks
  LeafData *kmat;   // dense blocks
    
  bool isLegionLeaf;
};


class LR_Matrix {

 public:
  // LR_Matrix() {}
  //~LR_Matrix();
  
  void create_tree(int, int, int, int, int,
		   Context, HighLevelRuntime *);
  void init_right_hand_side(int, int, Context, HighLevelRuntime *);
  void init_circulant_matrix(double, int, Context, HighLevelRuntime *);

  int get_num_rhs() {return rhs_cols;}
  
  /* --- tree root --- */
  int nleaf_per_node;
  FSTreeNode *uroot;
  FSTreeNode *vroot;

 private:

  /* --- create tree --- */

  void create_vnode_from_unode(FSTreeNode *, FSTreeNode *,
			       Context, HighLevelRuntime *);

  /* --- populate data --- */

  void init_RHS(FSTreeNode *node, int rand_seed, Range tag,
		Context, HighLevelRuntime *,
		int row_beg = 0);
  void init_Umat(FSTreeNode *node, Range tag,
		 Context, HighLevelRuntime *, int row_beg = 0);
  void init_Vmat(FSTreeNode *node, double diag, Range tag,
		 Context, HighLevelRuntime *, int row_beg = 0);

    
  /*--- helper functions ---*/

  void create_Hmatrix(FSTreeNode *, FSTreeNode *, int,
		      Context, HighLevelRuntime *);
  void set_circulant_Hmatrix_data(FSTreeNode * Hmat,
				  Range tag,
				  Context, HighLevelRuntime *,
				  int row_beg);
  
  
  /* --- private attributes --- */
  int rank; // only if every block has the same rank
  int rhs_rows;
  int rhs_cols;
};


void create_matrix(LogicalRegion &, int, int, Context,
		   HighLevelRuntime *);


/*--- for debugging purpose ---*/
void print_legion_tree(FSTreeNode *);


int count_leaf(FSTreeNode *node);

void fill_circulant_Kmat(FSTreeNode * vnode, int, int r, double diag,
			 double *Kmat, int LD);

#endif // _LEGION_TREE_
