#include <iomanip>
#include "save_region.h"


namespace {
  class SaveRegionTask : public TaskLauncher {
    
  public:
    struct TaskArgs {
      ColRange col_range;
      char filename[50];
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
  : TaskLauncher(TASKID, arg, pred, id, tag)
{
}

/*static*/
void SaveRegionTask::register_tasks(void)
{
  TASKID =
    HighLevelRuntime::register_legion_task<SaveRegionTask::cpu_task>(
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
  TaskArgs* task_args = (TaskArgs *)task->args;
  int col_beg = task_args->col_range.col_beg;
  int ncol    = task_args->col_range.ncol;
  char* filename = task_args->filename;

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
  assert(col_beg+ncol <= rect.dim_size(1));
  //if (ncol == 0)
  //ncol = rect.dim_size(1) - col_beg;


  std::ofstream outputFile(filename, std::ios_base::app);
  //outputFile << nrow << std::endl;
  //outputFile << ncol << std::endl;

  for (int i=0; i<nrow; i++) {
    for (int j=0; j<ncol; j++) {
      int row_idx = i;
      int col_idx = j+col_beg;
      int pnt[] = {row_idx, col_idx};
      double x = ptr[ row_idx + col_idx*nrow ];
      outputFile << std::setprecision(20) << x << '\t';
    }
    outputFile << std::endl;
  }
  outputFile.close();
}


void
save_region(FSTreeNode * node, ColRange rg, std::string filename,
	    Context ctx, HighLevelRuntime *runtime, bool wait) {

  if (node->isLegionLeaf == true) {

    //save_region(node->matrix->data, rg.col_beg, rg.ncol, filename, ctx, runtime);

    //typename
    SaveRegionTask::TaskArgs args;
    int len = filename.size();
    filename.copy(args.filename, len, 0);
    args.filename[len] = '\0';
    args.col_range = rg;
    
    SaveRegionTask launcher(TaskArgument(&args, sizeof(args)));
    
    launcher.add_region_requirement(
	       RegionRequirement(node->matrix->data,
				 READ_ONLY,
				 EXCLUSIVE,
				 node->matrix->data).
	       add_field(FID_X));

    Future fm = runtime->execute_task(ctx, launcher);
    if(wait)
      fm.get_void_result();
    
  } else {
    save_region(node->lchild, rg, filename, ctx, runtime, wait);
    save_region(node->rchild, rg, filename, ctx, runtime, wait);
  }  
}


void
save_solution(LR_Matrix &matrix, std::string &soln_file,
	      Context ctx, HighLevelRuntime *runtime) {

  ColRange ru = {0, matrix.get_num_rhs()};
  save_region(matrix.uroot, ru, soln_file,
	      ctx, runtime, false/*wait*/);
}


void register_output_tasks() {
  SaveRegionTask::register_tasks();
}
