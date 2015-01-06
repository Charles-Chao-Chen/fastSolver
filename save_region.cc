#include <iomanip>
#include "save_region.h"

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
  TaskArgs* task_args = (TaskArgs *)task->args;
  int col_beg    = task_args->col_range.begin;
  int ncol       = task_args->col_range.size;
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

  std::ofstream outputFile(filename, std::ios_base::app);
  //outputFile << nrow << "\t" << ncol << std::endl;

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


void
save_region(FSTreeNode * node, Range rg, std::string filename,
	    Context ctx, HighLevelRuntime *runtime) {

  if (node->isLegionLeaf == true) {

    SaveRegionTask::TaskArgs args;
    int len = filename.size();
    filename.copy(args.filename, len, 0);
    args.filename[len] = '\0';
    args.col_range.begin = rg.begin;
    args.col_range.size  = rg.size;
    
    SaveRegionTask launcher(TaskArgument(&args, sizeof(args)));    
    launcher.add_region_requirement(
	       RegionRequirement(node->matrix->data,
				 READ_ONLY,
				 EXCLUSIVE,
				 node->matrix->data).
	       add_field(FID_X));
    Future fm = runtime->execute_task(ctx, launcher);
    fm.get_void_result();
    
  } else {
    save_region(node->lchild, rg, filename, ctx, runtime);
    save_region(node->rchild, rg, filename, ctx, runtime);
  }  
}


void
save_solution(LR_Matrix &matrix, std::string soln_file,
	      Context ctx, HighLevelRuntime *runtime) {

  Range ru(matrix.get_num_rhs());
  save_region(matrix.uroot, ru, soln_file, ctx, runtime);
}


/*
void LR_Matrix::print_Vmat(FSTreeNode *node, std::string filename) {

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


/* ---- save h-tree functions ----*/

/*
enum {
  SAVE_REGION_TASK_ID = 100,
};


void save_region(FSTreeNode * node, std::string filename, Context ctx, HighLevelRuntime *runtime) {

  if (node->isLegionLeaf == true) {
    
    TaskLauncher save_task(SAVE_REGION_TASK_ID, TaskArgument(&filename[0], filename.size()+1));

    save_task.add_region_requirement(RegionRequirement(node->matrix->data, READ_ONLY,  EXCLUSIVE, node->matrix->data));
    save_task.region_requirements[0].add_field(FID_X);

    runtime->execute_task(ctx, save_task);
    
  } else {
    save_region(node->lchild, filename, ctx, runtime);
    save_region(node->rchild, filename, ctx, runtime);
  }  
}


void save_task(const Task *task, const std::vector<PhysicalRegion> &regions,
	       Context ctx, HighLevelRuntime *runtime) {

  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  char* filename = (char *)task->args;

  IndexSpace is_u = task->regions[0].region.get_index_space();
  Domain dom_u = runtime->get_index_space_domain(ctx, is_u);
  Rect<2> rect_u = dom_u.get_rect<2>();

  RegionAccessor<AccessorType::Generic, double> acc_u =
    regions[0].get_field_accessor(FID_X).typeify<double>();
  
  Rect<2> subrect;
  ByteOffset offsets[2];  

  double *u_ptr = acc_u.raw_rect_ptr<2>(rect_u, subrect, offsets);
  assert(rect_u == subrect);


  int nrow = rect_u.dim_size(0);
  int ncol = rect_u.dim_size(1);

  std::ofstream outputFile(filename, std::ios_base::app);
  if (!outputFile.is_open())
    std::cout << "Error opening file." << std::endl;
  outputFile<<nrow<<std::endl;
  outputFile<<ncol<<std::endl;

  for (int i=0; i<nrow; i++) {
    for (int j=0; j<ncol; j++) {

      int pnt[] = {i, j};
      double x = acc_u.read(DomainPoint::from_point<2>( Point<2>(pnt) ));
      outputFile << x << '\t';
    }
    outputFile << std::endl;
  }
  outputFile.close();
}


void register_save_task() {
  
  HighLevelRuntime::register_legion_task
    <save_task>(SAVE_REGION_TASK_ID,
		Processor::LOC_PROC,
		true, true,
		AUTO_GENERATE_ID,
		TaskConfigOptions(true),
		"save_region");
}
*/

void register_output_tasks() {
  SaveRegionTask::register_tasks();
  //register_save_task();
}

