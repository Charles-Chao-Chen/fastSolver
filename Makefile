
ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

#Flags for directing the runtime makefile what to include
DEBUG=0                   # Include debugging symbols
OUTPUT_LEVEL=LEVEL_DEBUG  # Compile time print level
SHARED_LOWLEVEL=0	  # Use the shared low level
USE_CUDA=0
#ALT_MAPPERS=1		  # Compile the alternative mappers

# Put the binary file name here
OUTFILE	:= main
# List all the application source files here
GEN_SRC	:= main.cc              \
	   test.cc		\
	   fast_solver.cc       \
	   solver_tasks.cc      \
	   gemm.cc              \
	   zero_matrix_task.cc 	\
	   hodlr_matrix.cc 	\
	   htree_helper.cc  	\
	   legion_matrix.cc   	\
	   init_matrix_tasks.cc \
	   save_region_task.cc  \
	   direct_solve.cc 	\
	   range.cc		\
	   timer.cc 	   	\
	   custom_mapper.cc

GEN_GPU_SRC	:=				# .cu files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	:=
CC_FLAGS	:= -g -I ./ 	  \
		   -DLEGION_PROF  \
		   -DLEGION_SPY   \
		   -DNODE_LOGGING \
#		   -g		\
#		   -DDEBUG	  \
#		   -DNDEBUG	  \
#		   -DDEBUG_GEMM
#		   -DDEBUG_NODE_SOLVE

NVCC_FLAGS	:=
GASNET_FLAGS	:=

# gnu blas and lapack
LD_FLAGS	:= -L /usr/lib/	-llapack -lblas -lm

# mkl linking flags
#LD_FLAGS := -L/share/apps/intel/intel-14/mkl/lib/intel64/ \
	-L/share/apps/intel/intel-14/lib/intel64/ \
	-lmkl_intel_lp64 	\
	-lmkl_core		\
	-lmkl_sequential	\
	-lpthread 		\
	-lm

#	-lmkl_intel_thread 	\
	-liomp5 		\
	-lmkl_sequential	\



###########################################################################
#
#   Don't change anything below here
#   
###########################################################################

# All these variables will be filled in by the runtime makefile
LOW_RUNTIME_SRC	:=
HIGH_RUNTIME_SRC:=
GPU_RUNTIME_SRC	:=
MAPPER_SRC	:=

include $(LG_RT_DIR)/runtime.mk

# General shell commands
SHELL	:= /bin/sh
SH	:= sh
RM	:= rm -f
LS	:= ls
MKDIR	:= mkdir
MV	:= mv
CP	:= cp
SED	:= sed
ECHO	:= echo
TOUCH	:= touch
MAKE	:= make
ifndef GCC
GCC	:= g++
endif
ifndef NVCC
NVCC	:= $(CUDA)/bin/nvcc
endif
SSH	:= ssh
SCP	:= scp

common_all : all

.PHONY	: common_all

GEN_OBJS	:= $(GEN_SRC:.cc=.o)
LOW_RUNTIME_OBJS:= $(LOW_RUNTIME_SRC:.cc=.o)
HIGH_RUNTIME_OBJS:=$(HIGH_RUNTIME_SRC:.cc=.o)
MAPPER_OBJS	:= $(MAPPER_SRC:.cc=.o)
# Only compile the gpu objects if we need to 
ifndef SHARED_LOWLEVEL
GEN_GPU_OBJS	:= $(GEN_GPU_SRC:.cu=.o)
GPU_RUNTIME_OBJS:= $(GPU_RUNTIME_SRC:.cu=.o)
else
GEN_GPU_OBJS	:=
GPU_RUNTIME_OBJS:=
endif

ALL_OBJS	:= $(GEN_OBJS) $(GEN_GPU_OBJS) $(LOW_RUNTIME_OBJS) $(HIGH_RUNTIME_OBJS) $(GPU_RUNTIME_OBJS) $(MAPPER_OBJS)

all:
	$(MAKE) $(OUTFILE)

# If we're using the general low-level runtime we have to link with nvcc
$(OUTFILE) : $(ALL_OBJS)

	@echo "---> Linking objects into one binary: $(OUTFILE)"
ifdef SHARED_LOWLEVEL
	$(GCC) -o $(OUTFILE) $(ALL_OBJS) $(LD_FLAGS) $(GASNET_FLAGS)
else
	$(NVCC) -o $(OUTFILE) $(ALL_OBJS) $(LD_FLAGS) $(GASNET_FLAGS)
endif

$(GEN_OBJS) : %.o : %.cc
	$(GCC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)

$(LOW_RUNTIME_OBJS) : %.o : %.cc
	$(GCC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)

$(HIGH_RUNTIME_OBJS) : %.o : %.cc
	$(GCC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)

$(MAPPER_OBJS) : %.o : %.cc
	$(GCC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)

$(GEN_GPU_OBJS) : %.o : %.cu
	$(NVCC) -o $@ -c $< $(INC_FLAGS) $(NVCC_FLAGS)

$(GPU_RUNTIME_OBJS): %.o : %.cu
	$(NVCC) -o $@ -c $< $(INC_FLAGS) $(NVCC_FLAGS)

clean:
	@$(RM) -rf $(OUTFILE) $(GEN_OBJS) *~

cleanall:
	@$(RM) -rf $(ALL_OBJS) *~


newfile:
	mv debug_v0td0_bf.txt debug_v0td0_bf_ref.txt
	mv debug_v1td1_bf.txt debug_v1td1_bf_ref.txt
	mv debug_v0tu0_bf.txt debug_v0tu0_bf_ref.txt
	mv debug_v1tu1_bf.txt debug_v1tu1_bf_ref.txt
	mv debug_umat.txt     debug_umat_ref.txt

test:
	make clean
	make -j 12
	make r2n

tar:	
	tar cvfz fastSolver.tgz 	\
		Makefile Readme TODO 	\
		main.cc 		\
		test.cc			test.h			\
		fast_solver.cc  	fast_solver.h 		\
		solver_tasks.cc		solver_tasks.h		\
		gemm.cc         	gemm.h  		\
					launch_node_task.h	\
		zero_matrix_task.cc	zero_matrix_task.h 	\
		hodlr_matrix.cc 	hodlr_matrix.h 		\
		htree_helper.cc		htree_helper.h		\
		legion_matrix.cc	legion_matrix.h		\
		init_matrix_tasks.cc	init_matrix_tasks.h 	\
		save_region_task.cc	save_region_task.h	\
		range.cc		range.h 		\
		direct_solve.cc		direct_solve.h		\
		timer.cc		timer.h			\
		lapack_blas.h		macros.h		\
		custom_mapper.cc	custom_mapper.h 	\


r1n:
	mpiexec -n 1 \
	-env MV2_SHOW_CPU_BINDING=1 \
	-env MV2_ENABLE_AFFINITY=0  \
	-env GASNET_IB_SPAWNER=mpi  \
	-env GASNET_BACKTRACE=1     \
	./main -cat legion_prof -level 5 \
	-ll:cpu 12 -ll:csize 30000 \
	-hl:sched 8192 -hl:window 8192

r2n:
	mpiexec -n 2 -ppn 1	\
	-env MV2_SHOW_CPU_BINDING=1 \
	-env MV2_ENABLE_AFFINITY=0  \
	-env GASNET_IB_SPAWNER=mpi  \
	-env GASNET_BACKTRACE=1     \
	./main			\
	-np 2 			\
	-level 5 		\
	-ll:cpu    8		\
	-ll:util   4 		\
	-ll:csize  30000 	\
	-hl:sched  8192  	\
	-hl:window 8192

r4n:
	mpiexec -n 4 -ppn 1	\
	-env MV2_SHOW_CPU_BINDING=1 \
	-env MV2_ENABLE_AFFINITY=0  \
	-env GASNET_IB_SPAWNER=mpi  \
	-env GASNET_BACKTRACE=1     \
	./main -level 5 \
	-ll:cpu 12 -ll:csize 10000 \
	-hl:sched 8192 -hl:window 8192

# --- legion profile ---

# nproc : the number of processes, and there should be one
#  process for every node
# numa  : can be set to 'numactl -m 0 -N 0' to use numa node
# ncpu  :
# nutil :
# leaf  : legion leaf size
prof:
	mpiexec -n $(nproc) -ppn 1  \
	-env MV2_SHOW_CPU_BINDING=1 \
	-env MV2_ENABLE_AFFINITY=0  \
	-env GASNET_IB_SPAWNER=mpi  \
	-env GASNET_BACKTRACE=1     \
	./main			\
	-test $(test)		\
	-np $(nproc) 		\
	-leaf $(leaf)		\
	-cat legion_prof	\
	-level 2 		\
	-ll:cpu    $(ncpu)	\
	-ll:util   $(nutil)	\
	-ll:csize  20000 	\
	-hl:sched  8192  	\
	-hl:window 8192

#	-hl:prof   1		\
#	numactl			\
	-m 0 -N 0		\
	$(numa) 		\

# idx : node index
node:
	python ~/legion/tools/legion_prof.py -p node_$(idx).log
#	python ~/legion/tools/legion_prof.py -p node_$(idx).log \
	> legion_node.txt

# --- legion spy ---
#spy2:\
	mpiexec -n 2 -ppn 1	\
	-env OMP_NUM_THREADS=1	\
	-env MV2_SHOW_CPU_BINDING=1 \
	-env MV2_ENABLE_AFFINITY=0  \
	-env GASNET_IB_SPAWNER=mpi  \
	-env GASNET_BACKTRACE=1     \
	./main -cat legion_spy -level 2 \
	-ll:cpu 12 -ll:csize 30000 \
	-hl:sched 8192 -hl:window 8192

# --- Sapling commands ---

# check task execution (data movement)
spy2:
	mpirun -H n0001,n0002 \
	-bind-to none \
	-x GASNET_IB_SPAWNER -x GASNET_BACKTRACE=1 \
	./main \
	-np 2 -leaf 1 \
	-cat legion_spy -level 2 \
	-ll:cpu 11 -ll:util 1 \
	-ll:csize 30000 \
	-hl:sched 8192
#	python ~/legion/tools/legion_spy.py -p node_0.log

prof2:
	mpirun -H n0001,n0002 \
	-bind-to none \
	-x GASNET_IB_SPAWNER -x GASNET_BACKTRACE=1 \
	./main \
	-np 2 -leaf 1 \
	-cat legion_prof -level 2 \
	-ll:cpu 11 -ll:util 1 \
	-ll:csize 30000 \
	-hl:sched 8192

