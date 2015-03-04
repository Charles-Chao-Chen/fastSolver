#ifndef _LEGION_TREE_
#define _LEGION_TREE_

#include <string>
#include <fstream>
#include "legion_matrix.h"
#include "timer.hpp"
#include "legion.h"

using namespace LegionRuntime::HighLevel;

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
  bool is_launch_node() const {return isLaunchNode;}
  void set_launch_node(bool is) {isLaunchNode = is;}

  int nrow;    
  int ncol;
  int row_beg; // begin index in the region
  int col_beg;

  FSTreeNode *lchild;
  FSTreeNode *rchild;
  FSTreeNode *Hmat;
  
  LMatrix *lowrank_matrix; // low rank blocks
  LMatrix *dense_matrix;   // dense blocks

private:
  bool isLegionLeaf;
  bool isLaunchNode;
};


class HodlrMatrix {

 public:
  HodlrMatrix() {}
  HodlrMatrix
    (int col, int row, int gl, int sl,
     int r, int t, int leaf, const std::string&);
  //~HodlrMatrix();
  
  void create_tree
    (Context, HighLevelRuntime *);
  void init_rhs
    (long, const Range&, Context, HighLevelRuntime *);
  void init_circulant_matrix
    (double, const Range&, Context, HighLevelRuntime *);

  void save_rhs
    (Context, HighLevelRuntime *);
  void save_solution
    (Context, HighLevelRuntime *);
  
  int  get_num_rhs() {return rhs_cols;}
  int  get_num_leaf() {return nleaf;}
  void set_num_leaf(int nleaf) {this->nleaf = nleaf;}
  int  get_num_launch_node() {return nLaunchNode;}
  void set_num_launch_node(int n) {nLaunchNode = n;}
  std::string get_file_soln() const {return file_soln;}

  void display_launch_time() const {
    std::cout << "Time cost for launching init-tasks :"
	      << timeInit << " s" << std::endl;
  }
  
  /* --- tree root --- */
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
  
  /* --- private attributes --- */
  int rhs_cols;
  int rhs_rows;
  int gloLevel;  // level of the global tree
  int subLevel;  // level of the sub problem
  int rank;      // same rank for all blocks
  int threshold; // threshold of dense blocks
  int nleaf;     // legion leaf size for controlling fine granularity
  
 private:
  int nLaunchNode;
  //Range procs;
  double timeInit;
  std::string file_rhs;
  std::string file_soln;
};


void create_Hmatrix(FSTreeNode *, FSTreeNode *, int,
		    Context, HighLevelRuntime *);
  
void set_circulant_Hmatrix_data
(FSTreeNode * Hmat, Range tag,
 Context, HighLevelRuntime *,
 int row_beg);
  


// TODO: move into LMatrix class
void fill_circulant_Kmat
  (FSTreeNode * vnode, int, int r, double diag,
   double *Kmat, int LD);




#endif // _LEGION_TREE_
