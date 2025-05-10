# general_kernel -> 03_neon -> submissions -> src ... include
INCLUDE_DIR = ../../../../include

default: test microbench
	@echo "Build complete. Run the microkernel with ./build/microkernel and the test with ./build/test"

test:
	mkdir -p build
	g++ tests/tests.cpp tests/matmul_14_6_64.test.cpp matmul_14_6_64/v1_matmul_14_6_64.s matmul_14_6_64/v2_matmul_14_6_64.s matmul_14_6_64/v3_matmul_14_6_64.s matmul_14_6_64/v4_matmul_14_6_64.s -I$(INCLUDE_DIR) -o build/test_v1
	g++ tests/tests.cpp tests/matmul_15_6_64.test.cpp matmul_15_6_64/v1_matmul_15_6_64.s matmul_15_6_64/v2_matmul_15_6_64.s matmul_15_6_64/v3_matmul_15_6_64.s -I$(INCLUDE_DIR) -o build/test_v2
	g++ tests/tests.cpp tests/matmul_M_6_64.test.cpp general_kernel/matmul_M_6_64.s -I$(INCLUDE_DIR) -o build/test_general

microbench:
	mkdir -p build
	g++ benchmark/microbench.cpp matmul_14_6_64/v1_matmul_14_6_64.s matmul_14_6_64/v2_matmul_14_6_64.s matmul_14_6_64/v3_matmul_14_6_64.s matmul_14_6_64/v4_matmul_14_6_64.s matmul_15_6_64/v1_matmul_15_6_64.s matmul_15_6_64/v2_matmul_15_6_64.s matmul_15_6_64/v3_matmul_15_6_64.s general_kernel/matmul_M_6_64.s -I$(INCLUDE_DIR) -o build/microbench

clean:
	rm -rf build