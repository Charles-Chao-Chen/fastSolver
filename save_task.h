#ifndef _SAVE_TASK_H
#define _SAVE_TASK_H

#include <string>

#include "hodlr_matrix.h"
#include "macros.h"


void register_output_tasks();


void
save_solution(HodlrMatrix &, std::string,
	      Context ctx, HighLevelRuntime *runtime);

void save_HodlrMatrix
  (FSTreeNode * node, std::string filename,
   Context ctx, HighLevelRuntime *runtime,
   Range rg = (Range)(0,-1));

// TODO: move into LMatrix class
void save_LMatrix
  (const LMatrix *matrix, const std::string filename,
   Context ctx, HighLevelRuntime *runtime,
   const Range rg = (Range)(0,-1));


void
save_data(double *ptr, int nRows, int colBegin, int nCols,
	    std::string filename);

void
print_Vmat(FSTreeNode *node, std::string filename);
  

#endif //_SAVE_TASK_H
