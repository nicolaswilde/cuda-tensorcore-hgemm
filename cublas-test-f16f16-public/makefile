default:
	nvcc cublas-test.cu -o cublas-test -arch=compute_86 -code=sm_86 -lcublas --ptxas-options=-v

run:
	./cublas-test
