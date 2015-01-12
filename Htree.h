#ifndef _LEGION_TREE_
#define _LEGION_TREE_

#include <string>
#include <fstream>

#include "legion_matrix.h"

#include "legion.h"


using namespace LegionRuntime::HighLevel;


// used in InitCirculantKmatTask::TaskArgs
// for the subtree stored in array
#define MAX_TREE_SIZE 30


enum MatrixType {
  UMatrix,
  VMatrix,
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
	     LMatrix *matrix=NULL,
	     LMatrix *kmat=NULL,
	     bool isLegionLeaf=false);

  bool is_real_leaf()   const;
  bool is_legion_leaf() const {return isLegionLeaf;}
  void set_legion_leaf(bool is) {isLegionLeaf = is;}
  
  int row_beg; // begin index in the region
  int col_beg;
  int nrow;    
  int ncol;

  FSTreeNode *lchild;
  FSTreeNode *rchild;
  FSTreeNode *Hmat;
  
  LMatrix *lowrank_matrix; // low rank blocks
  LMatrix *dense_matrix;   // dense blocks

private:
  bool isLegionLeaf;
};


class HodlrMatrix {

 public:
  // HodlrMatrix() {}
  //~HodlrMatrix();
  
  void create_tree(int, int, int, int, int,
		   Context, HighLevelRuntime *);
  void initialize_rhs(int, int, int, Context, HighLevelRuntime *);
  void init_circulant_matrix(double, int, Context, HighLevelRuntime *);

  int get_num_rhs() {return rhs_cols;}
  int get_num_leaf() {return nleaf;}
  int set_num_leaf(int nleaf) {this->nleaf = nleaf;}
  
  /* --- tree root --- */
  //int nleaf_per_node;
  FSTreeNode *uroot;
  FSTreeNode *vroot;

 private:

  /* --- create tree --- */

  void create_vnode_from_unode(FSTreeNode *, FSTreeNode *,
			       Context, HighLevelRuntime *);

  /* --- populate data --- */

  void init_Umat(FSTreeNode *node, Range tag,
		 Context, HighLevelRuntime *, int row_beg = 0);
  void init_Vmat(FSTreeNode *node, double diag, Range tag,
		 Context, HighLevelRuntime *, int row_beg = 0);

    
  /*--- helper functions ---*/

  void create_Hmatrix(FSTreeNode *, FSTreeNode *, int,
		      Context, HighLevelRuntime *);
  void set_circulant_Hmatrix_data
    (FSTreeNode * Hmat, Range tag,
     Context, HighLevelRuntime *,
     int row_beg);
  
  
  /* --- private attributes --- */
  int rank; // only if every block has the same rank
  int rhs_rows;
  int rhs_cols;
  int nleaf;
};



// TODO: move into LMatrix class
void fill_circulant_Kmat
  (FSTreeNode * vnode, int, int r, double diag,
   double *Kmat, int LD);




#endif // _LEGION_TREE_
