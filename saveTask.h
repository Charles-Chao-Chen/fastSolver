#ifndef _SAVE_TASK_H
#define _SAVE_TASK_H

#include <string>

#include "Htree.h"
#include "macros.h"


void register_output_tasks();


void
save_solution(LR_Matrix &, std::string,
	      Context ctx, HighLevelRuntime *runtime);

void
save_Htree(FSTreeNode * node, std::string filename,
	   Context ctx, HighLevelRuntime *runtime,
	   Range rg = (Range)(0,-1));

void
save_region(LogicalRegion data, std::string save_file,
	    Context ctx, HighLevelRuntime *runtime,
	    Range rg = (Range)(0,-1));

void
save_data(double *ptr, int nRows, int colBegin, int nCols,
	    std::string filename);

void
print_Vmat(FSTreeNode *node, std::string filename);
  

#endif //_SAVE_TASK_H
