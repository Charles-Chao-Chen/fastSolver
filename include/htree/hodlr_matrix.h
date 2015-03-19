#ifndef _LEGION_TREE_
#define _LEGION_TREE_

#include <string>
#include <fstream>

#include "node.h"
#include "legion_matrix.h"
#include "matrix_array.hpp"
#include "timer.hpp"

#include "legion.h"

using namespace LegionRuntime::HighLevel;

class HodlrMatrix {
 public:
  HodlrMatrix() {}
  HodlrMatrix
    (int col, int row, int gl, int sl,
     int r, int t, int leaf, const std::string&);
  //~HodlrMatrix();
  
  void create_tree
    (Context, HighLevelRuntime *, const LMatrixArray* array=NULL);
  void init_rhs
    (const long, const Range&, Context, HighLevelRuntime *);
  void init_circulant_matrix
    (const double, const Range&, Context, HighLevelRuntime *,
     bool skipU=false);
  //  void init_from_regions(const LMatrixArray &);
  
  void save_rhs
    (Context, HighLevelRuntime *) const;
  void save_solution
    (Context, HighLevelRuntime *) const;

  int  launch_level() const {return gloLevel-subLevel;}
  int  get_num_rhs() {return rhs_cols;}
  int  get_num_leaf() {return nLegionLeaf;}
  std::string get_file_soln() const {return file_soln;}

  void display_launch_time() const {
    std::cout << "Time cost for launching init-tasks :"
	      << timeInit << " s" << std::endl;
  }
  
  /* --- tree root --- */
  Node *uroot;
  Node *vroot;

 private:

  /* --- populate data --- */
  void init_Umat(Node *node, Range tag,
		 Context, HighLevelRuntime *, int row_beg = 0);  
  void init_Vmat(Node *node, double diag, Range tag,
		 Context, HighLevelRuntime *, int row_beg = 0);
  
  /* --- private attributes --- */
  int rhs_cols;
  int rhs_rows;
  int gloLevel;  // level of the global tree
  int subLevel;  // level of the sub problem
  int rank;      // same rank for all blocks
  int threshold; // threshold of dense blocks
  int leafSize;  // legion leaf size for controlling fine granularity
  int nLegionLeaf;
  
 private:
  double timeInit;
  std::string file_rhs;
  std::string file_soln;
};

void create_Hmatrix(Node *, Node *, int,
		    Context, HighLevelRuntime *);
  
void set_circulant_Hmatrix_data
(Node * Hmat, Range tag,
 Context, HighLevelRuntime *,
 int row_beg);

// TODO: move into LMatrix class
void fill_circulant_Kmat
  (Node * vnode, int, int r, double diag,
   double *Kmat, int LD);

void save_HodlrMatrix
(Node * node, std::string filename,
 Context ctx, HighLevelRuntime *runtime,
 Range rg, bool print_seed=false);

#endif // _LEGION_TREE_
