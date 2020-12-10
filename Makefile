BINDIR = ./
EXECUTABLE := main

CCFILES := util.c 
CUFILES := xmalloc.cu
CUFILES_sm_11 := main.cu 

ROOTDIR := ../../NVIDIA_GPU_Computing_SDK/C/common/

CUDACCFLAGS := --debug --device-debug 0

include $(ROOTDIR)/../common/common.mk
