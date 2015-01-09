#ifndef LEGION_MATRIX_H
#define LEGION_MATRIX_H

#include "legion.h"

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;


class Range {
 public:
 Range(                   ): begin(0),     size(0)    {}
 Range(int size           ): begin(0),     size(size) {}
 Range(int begin, int size): begin(begin), size(size) {}
  Range lchild();
  Range rchild();
 public:
  int begin;
  int size;
};


// Data at leaf nodes is stored in column major fashion.
// This allows extracting a given column range.
class LMatrix {

public:
 LMatrix(int rows=0, int cols=0):
  rows(rows), cols(cols), 
  data(LogicalRegion::NO_REGION) {}

  //~LMatrix();

  void init_circulant_matrix
    (int col_beg, int row_beg, int r, Range tag,
     Context ctx, HighLevelRuntime *runtime);

  
  void zero_matrix
    (Range, Context ctx, HighLevelRuntime *runtime);

  /* --- class members --- */
  
  int rows;  
  int cols;

  IndexSpace iSpace;
  FieldSpace fSpace;
  LogicalRegion data; // storing the data
};




#endif // LEGION_MATRIX_H