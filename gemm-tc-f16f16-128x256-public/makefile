default:
	nvcc gemm.cu -o gemm -arch=compute_86 -code=sm_86 -lcublas --ptxas-options=-v

run:
	./gemm
