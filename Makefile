# Compilers

NVCC=$(PREP) nvcc
FC = gfortran
CPP = g++
# Flags

FFLAG = -o
OPTMZ = Grid_efdc -ffixed-line-length-132 -fdefault-real-8

CUDACFLAGS  = -I${CUDA_PATH}/include
CUDALDFLAGS   = -L${CUDA_PATH}/lib64 -lcudart

SW2D-EFDC: SW2D-EFDC.cu gefdc.f Gorp.cpp GorpMain.cpp Makefile
	$(NVCC) $(CUDACFLAGS) $(CUDALDFLAGS) SW2D-EFDC.cu -o SW2D-EFDC
	$(FC) $(FFLAG) $(OPTMZ)  gefdc.f
	$(CPP) Gorp.cpp GorpMain.cpp -o Gorp
	+$(MAKE) -C EFDC_src

print:
	@echo ""
	@echo "Compling..."

done:
	@echo ""
	@echo "Done!!!!"
