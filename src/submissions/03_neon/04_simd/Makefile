# general_kernel -> 03_neon -> submissions -> src ... include
INCLUDE_DIR = ../../../../include

default: test
	@echo "Build complete. Run the microkernel with ./build/microkernel and the test with ./build/test"

test:
	mkdir -p build
	g++ tests.cpp ./matmul_14_6_64/matmul_14_6_64.test.cpp ./matmul_14_6_64/v1_matmul_14_6_64.s ./matmul_14_6_64/v2_matmul_14_6_64.s ./matmul_14_6_64/v3_matmul_14_6_64.s ./matmul_14_6_64/v4_matmul_14_6_64.s -I$(INCLUDE_DIR) -o build/test_v1
	g++ tests.cpp ./matmul_15_6_64/matmul_15_6_64.test.cpp ./matmul_15_6_64/v1_matmul_15_6_64.s ./matmul_15_6_64/v2_matmul_15_6_64.s ./matmul_15_6_64/v3_matmul_15_6_64.s -I$(INCLUDE_DIR) -o build/test_v2
	g++ tests.cpp ./general_kernel/matmul_M_6_64.test.cpp ./general_kernel/matmul_M_6_64.s -I$(INCLUDE_DIR) -o build/test_general

benchmark:
	mkdir -p build
	g++ microbench.cpp ./matmul_14_6_64/v1_matmul_14_6_64.s ./matmul_14_6_64/v2_matmul_14_6_64.s ./matmul_14_6_64/v3_matmul_14_6_64.s ./matmul_14_6_64/v4_matmul_14_6_64.s ./matmul_15_6_64/v1_matmul_15_6_64.s ./matmul_15_6_64/v2_matmul_15_6_64.s ./matmul_15_6_64/v3_matmul_15_6_64.s ./general_kernel/matmul_M_6_64.s -I$(INCLUDE_DIR) -o build/bench

clean:
	rm -rf build