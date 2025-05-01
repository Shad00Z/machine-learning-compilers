# 02_microkernel -> 03_neon -> submissions -> src ... include
INCLUDE_DIR = ../../../../include

default: microkernel benchmark test
	@echo "Build complete. Run the microkernel with ./build/microkernel, the benchmark with ./build/bench and the test with ./build/test"

microkernel:
	mkdir -p build
	g++ microkernel.cpp ./optimization/v1_matmul_16_6_1.s ./optimization/v2_matmul_16_6_1.s ./optimization/v3_matmul_16_6_1.s -I$(INCLUDE_DIR) -o build/microkernel

benchmark:
	mkdir -p build
	g++ microbench.cpp ./benchmarking/b1_matmul_16_6_1.s ./benchmarking/b2_matmul_16_6_1.s ./benchmarking/b3_matmul_16_6_1.s -I$(INCLUDE_DIR) -o build/bench

test:
	mkdir -p build
	g++ tests.cpp matmul_16_6_1.test.cpp ./optimization/v1_matmul_16_6_1.s ./optimization/v2_matmul_16_6_1.s ./optimization/v3_matmul_16_6_1.s -I$(INCLUDE_DIR) -o build/test

clean:
	rm -rf build