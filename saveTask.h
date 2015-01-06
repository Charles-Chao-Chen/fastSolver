#ifndef _SAVE_TASK_H
#define _SAVE_TASK_H

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
print_Vmat(FSTreeNode *node, std::string filename);
  

#endif //_SAVE_TASK_H
