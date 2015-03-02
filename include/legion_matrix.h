#ifndef LEGION_MATRIX_H
#define LEGION_MATRIX_H

#include "legion.h"
#include "range.h"

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;


// Data at leaf nodes is stored in column major fashion.
// This allows extracting a given column range.
class LMatrix {

public:
 LMatrix(int rows=0, int cols=0):
  rows(rows), cols(cols), 
  data(LogicalRegion::NO_REGION) {}

  //TODO: deconstructor is not appropriate here, because
  // it has to take runtime and ctx as parameters.
  // So use destroy() instead.
  //~LMatrix();
  // destroy();

  // generate random matrix
  void rand
    (const int, const Range&, const Range&,
     Context, HighLevelRuntime*);

  // initialize zero matrix
  // TODO: add (probably default)
  // const Range& colRange parameter
  void zero
    (const Range&, Context, HighLevelRuntime*);

  // initialize the a skinny circulant matrix 
  // e.g. [ 0 1 2
  //        1 2 0
  //        2 0 1
  //        0 1 2
  //        1 2 0 ]
  void circulant
    (int col_beg, int row_beg, int r, Range tag,
     Context ctx, HighLevelRuntime *runtime);


  // initialize dense block as: U * U^T + D 
  void dense
    (int col_beg, int row_beg, int r, Range tag,
     Context ctx, HighLevelRuntime *runtime);


  void save
    (const std::string,
    Context, HighLevelRuntime *, const Range);


  /* --- class members --- */
  
  int rows;  
  int cols;

  IndexSpace iSpace;
  FieldSpace fSpace;
  LogicalRegion data; // storing the data
};




#endif // LEGION_MATRIX_H
