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
 LMatrix(const int rows=0, const int cols=0,
	 const LogicalRegion lr=LogicalRegion::NO_REGION);

  //TODO: deconstructor is not appropriate here, because
  // it has to take runtime and ctx as parameters.
  // So use destroy() instead.
  //~LMatrix();
  // destroy();

  // random matrix
  void rand(const long, const Range&, const int,
	     Context, HighLevelRuntime*);

  // zero matrix
  void zero(const int, Context, HighLevelRuntime*);

  // initialize the a skinny circulant matrix 
  // e.g. [ 0 1 2
  //        1 2 0
  //        2 0 1
  //        0 1 2
  //        1 2 0 ]
  void circulant
    (const int col, const int row, const int r, const int tag,
     Context ctx, HighLevelRuntime *runtime);

  // dense block as: U * U^T + D 
  void dense
    (const int col, const int row, const int r, const int tag,
     Context ctx, HighLevelRuntime *runtime);

  // output data to file
  void save
    (const std::string&, const Range&,
     Context, HighLevelRuntime *, bool print_seed=false);

 public:
  /* --- class members --- */
  int rows;  
  int cols;
  long seed;
  IndexSpace iSpace;
  FieldSpace fSpace;
  LogicalRegion data;
};

#endif // LEGION_MATRIX_H
