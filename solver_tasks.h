#ifndef _SOLVER_TASKS_H
#define _SOLVER_TASKS_H

#include "hodlr_matrix.h"
#include "legion.h"

using namespace LegionRuntime::HighLevel;


void register_solver_operators();


void solve_node_matrix
(LMatrix *(&V0Tu0), LMatrix *(&V1Tu1),
 LMatrix *(&V0Td0), LMatrix *(&V1Td1),
 Range task_tag,
 Context ctx, HighLevelRuntime *runtime);


void
solve_legion_leaf(FSTreeNode * uleaf, FSTreeNode * vleaf,
		  Range task_tag,
		  Context ctx, HighLevelRuntime *runtime);

  
#endif // _SOLVER_TASKS_H
