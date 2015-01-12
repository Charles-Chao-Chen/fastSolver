#include <iomanip>
#include "save_task.h"

using namespace LegionRuntime::Accessor;

namespace {
  class SaveRegionTask : public TaskLauncher {
    
  public:
    struct TaskArgs {
      Range col_range;
      char  filename[50];
    };

    SaveRegionTask(TaskArgument arg,
		   Predicate pred = Predicate::TRUE_PRED,
		   MapperID id = 0,
		   MappingTagID tag = 0);
  
    static int TASKID;
    static void register_tasks(void);

  public:
    static void cpu_task(const Task *task,
			 const std::vector<PhysicalRegion> &regions,
			 Context ctx, HighLevelRuntime *runtime);
  };
}


/* ---- SaveRegionTask implementation ---- */

/*static*/
int SaveRegionTask::TASKID;

SaveRegionTask::SaveRegionTask(TaskArgument arg,
			       Predicate pred /*= Predicate::TRUE_PRED*/,
			       MapperID id /*= 0*/,
			       MappingTagID tag /*= 0*/)
  : TaskLauncher(TASKID, arg, pred, id, tag) {}

/*static*/
void SaveRegionTask::register_tasks(void)
{
  TASKID = HighLevelRuntime::register_legion_task
    <SaveRegionTask::cpu_task>(
			       AUTO_GENERATE_ID,
			       Processor::LOC_PROC, 
			       true,
			       true,
			       AUTO_GENERATE_ID,
			       TaskConfigOptions(true/*leaf*/),
			       "save_solution");
  printf("Register task %d : save_solution\n", TASKID);
}


void SaveRegionTask::cpu_task(const Task *task,
			      const std::vector<PhysicalRegion> &regions,
			      Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  const TaskArgs* task_args = (TaskArgs *)task->args;
  const char* filename = task_args->filename;
  int col_beg          = task_args->col_range.begin;
  int ncol             = task_args->col_range.size;
  

  IndexSpace is   = task->regions[0].region.get_index_space();
  Domain     dom  = runtime->get_index_space_domain(ctx, is);
  Rect<2>    rect = dom.get_rect<2>();

  RegionAccessor<AccessorType::Generic, double> acc =
    regions[0].get_field_accessor(FID_X).typeify<double>();
  
  Rect<2> subrect;
  ByteOffset offsets[2];  

  double *ptr =  acc.raw_rect_ptr<2>(rect, subrect, offsets);
  assert(rect == subrect);
  assert(ptr  != NULL);

  int nrow = rect.dim_size(0);
  if (ncol == -1)
    ncol = rect.dim_size(1);
  assert(col_beg+ncol <= rect.dim_size(1));
  save_data(ptr, nrow, col_beg, ncol, filename);
}


void
save_solution(HodlrMatrix &matrix, std::string soln_file,
	      Context ctx, HighLevelRuntime *runtime)
{
  Range ru(matrix.get_num_rhs());
  save_HodlrMatrix(matrix.uroot, soln_file, ctx, runtime, ru);
}


void save_HodlrMatrix
  (FSTreeNode * node, std::string filename,
   Context ctx, HighLevelRuntime *runtime, Range rg)
{
  if ( node->is_legion_leaf() ) {
    save_LMatrix(node->lowrank_matrix, filename, ctx, runtime, rg);
  } else {
    save_HodlrMatrix(node->lchild, filename, ctx, runtime, rg);
    save_HodlrMatrix(node->rchild, filename, ctx, runtime, rg);
  }
}


void save_LMatrix
  (const LMatrix *matrix, const std::string filename,
   Context ctx, HighLevelRuntime *runtime, const Range &rg) {

    SaveRegionTask::TaskArgs args;
    int len = filename.size();
    filename.copy(args.filename, len, 0);
    args.filename[len] = '\0';
    args.col_range = rg;
    
    SaveRegionTask launcher(TaskArgument(&args, sizeof(args)));  
    launcher.add_region_requirement(RegionRequirement
				    (matrix->data,
				     READ_ONLY,
				     EXCLUSIVE,
				     matrix->data).
				    add_field(FID_X)
				    );
    Future fm = runtime->execute_task(ctx, launcher);
    fm.get_void_result();
}


/*
void save_Htree
  (FSTreeNode * node, std::string filename,
   Context ctx, HighLevelRuntime *runtime, Range rg)
{
  if ( node->is_legion_leaf() ) {
    save_region(node->lowrank_matrix->data, filename, ctx, runtime, rg);
  } else {
    save_Htree(node->lchild, filename, ctx, runtime, rg);
    save_Htree(node->rchild, filename, ctx, runtime, rg);
  }  
}


void
save_region(LogicalRegion data, std::string filename,
	    Context ctx, HighLevelRuntime *runtime,
	    Range rg) {

    SaveRegionTask::TaskArgs args;
    int len = filename.size();
    filename.copy(args.filename, len, 0);
    args.filename[len] = '\0';
    args.col_range = rg;
    
    SaveRegionTask launcher(TaskArgument(&args, sizeof(args)));  
    launcher.add_region_requirement(
	       RegionRequirement(data,
				 READ_ONLY,
				 EXCLUSIVE,
				 data).
	       add_field(FID_X));
    Future fm = runtime->execute_task(ctx, launcher);
    fm.get_void_result();
}
*/


void
save_data(double *ptr, int nrow, int col_beg, int ncol,
	  std::string filename) {

  std::ofstream outputFile(filename.c_str(), std::ios_base::app);
  for (int i=0; i<nrow; i++) {
    for (int j=0; j<ncol; j++) {
      int row_idx = i;
      int col_idx = j+col_beg;
      double x = ptr[ row_idx + col_idx*nrow ];
      outputFile << std::setprecision(20) << x << '\t';
    }
    outputFile << std::endl;
  }
  outputFile.close();
}



/*
void HodlrMatrix::print_Vmat(FSTreeNode *node, std::string filename) {

  //  if (node == vroot)
  //save_region(node, filename, ctx, runtime); // print big V matrix

  if (node->Hmat != NULL)
    save_region(node->Hmat, filename, ctx, runtime);
  else if (node != vroot)
    return;

  if (node->lchild != NULL && node->rchild != NULL) {
    print_Vmat(node->lchild, filename);
    print_Vmat(node->rchild, filename);
  }
}
*/


void register_output_tasks() {
  SaveRegionTask::register_tasks();
}

