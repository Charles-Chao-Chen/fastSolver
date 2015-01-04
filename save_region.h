#ifndef _SAVE_REGION_H
#define _SAVE_REGION_H

#include "Htree.h"
#include "macros.h"


void register_output_tasks();


void
save_solution(LR_Matrix &, std::string,
	      Context ctx, HighLevelRuntime *runtime);

void
print_Vmat(FSTreeNode *node, std::string filename);
  

#endif //_SAVE_REGION_H
