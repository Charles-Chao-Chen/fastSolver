
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
OUTFILE		:= main
# List all the application source files here
GEN_SRC		:= main.cc              \
		   fast_solver.cc       \
		   solver_tasks.cc      \
		   gemm.cc              \
		   zero_matrix_task.cc 	\
		   hodlr_matrix.cc 	\
		   htree_helper.cc  	\
		   legion_matrix.cc   	\
		   init_matrix_tasks.cc \
		   save_task.cc     	\
		   direct_solve.cc 	\
		   timer.cc 	   	\
		   custom_mapper.cc # .cc files
GEN_GPU_SRC	:=				# .cu files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	:=
CC_FLAGS	:= -g -I ./ 	  \
		   -DLEGION_PROF  \
		   -DLEGION_SPY   \
		   -DNODE_LOGGING \
		   -DDEBUG	  \
		   -DDEBUGGEMM
NVCC_FLAGS	:=
GASNET_FLAGS	:=

#lapack and blas on sapling
#LD_FLAGS	:= -L /usr/lib/	-l :liblapack.so.3 -l :libblas.so.3 -lm
LD_FLAGS	:= -L /usr/lib/	-llapack -lblas -lm

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
	mpiexec -hosts compute-55-2,compute-55-4 \
	-env MV2_SHOW_CPU_BINDING=1 \
	-env MV2_ENABLE_AFFINITY=0  \
	-env GASNET_IB_SPAWNER=mpi  \
	-env GASNET_BACKTRACE=1     \
	./main -level 5 \
	-ll:cpu 12 -ll:csize 30000 \
	-hl:sched 8192 -hl:window 8192

prof1:
	mpirun -H n0000 -bind-to none -x GASNET_IB_SPAWNER -x \
	GASNET_BACKTRACE=1 ./main -cat legion_prof -level 2 \
	-ll:cpu 12 -ll:csize 30000 -hl:sched 8192 -hl:window 8192

prof2:
	mpiexec -hosts compute-55-2,compute-55-4 \
	-env MV2_SHOW_CPU_BINDING=1 \
	-env MV2_ENABLE_AFFINITY=0  \
	-env GASNET_IB_SPAWNER=mpi  \
	-env GASNET_BACKTRACE=1     \
	./main -cat legion_prof -level 2 \
	-ll:cpu 12 -ll:csize 30000 \
	-hl:sched 8192 -hl:window 8192

prof4:
	numactl -m 0 -N 0 mpirun -H n0002 -H n0003 -H n0000 -H n0001 -bind-to none -x GASNET_IB_SPAWNER -x \
	GASNET_BACKTRACE=1 ./main -cat legion_prof -level 2 \
	-ll:cpu 12 -ll:csize 30000 -hl:sched 600

spy2:
	mpirun -H n0001 -H n0002 -bind-to none -x GASNET_IB_SPAWNER -x \
	GASNET_BACKTRACE=1 ./main -cat legion_spy -level 2 \
	-ll:cpu 12 -ll:csize 30000 -hl:sched 600

newfile:
	#mv Umat.txt    Umat_ref.txt
	#mv Ufinish.txt Ufinish_ref.txt
	#mv V0Td0.txt          V0Td0_ref.txt
	#mv V0Td0_finish.txt   V0Td0_finish_ref.txt
	#mv V1Td1.txt          V1Td1_ref.txt
	#mv V1Td1_finish.txt   V1Td1_finish_ref.txt
	#mv V1Tu1.txt          V1Tu1_ref.txt
	#mv V1Tu1_finish.txt   V1Tu1_finish_ref.txt
	mv debug_gemm_bf.txt debug_gemm_bf_ref.txt
	mv debug_gemm_af.txt debug_gemm_af_ref.txt

test:
	make clean
	make -j 12
	make r2n
tar:	
	tar cvfz fastSolver.tgz Makefile Readme main.cc fastSolver.cc fastSolver.h hodlr_matrix.h.cc hodlr_matrix.h.h gemm.cc gemm.h utility.cc utility.h custom_mapper.cc custom_mapper.h
