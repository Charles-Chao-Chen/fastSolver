#ifndef _SOLVER_TASKS_H
#define _SOLVER_TASKS_H

#include "Htree.h"
#include "legion.h"

using namespace LegionRuntime::HighLevel;


void register_solver_operators();


void
solve_node_matrix(LogicalRegion & V0Tu0, LogicalRegion & V1Tu1,
		  LogicalRegion & V0Td0, LogicalRegion & V1Td1,
		  Range task_tag,
		  Context ctx, HighLevelRuntime *runtime);


void
solve_legion_leaf(FSTreeNode * uleaf, FSTreeNode * vleaf,
		  Range task_tag,
		  Context ctx, HighLevelRuntime *runtime);

  
#endif // _SOLVER_TASKS_H
