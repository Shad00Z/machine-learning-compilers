# 02_microkernel -> 03_neon -> submissions -> src ... include
INCLUDE_DIR = ../../../../include

default: microkernel microbench test
	@echo "Build complete. Run the microkernel with ./build/microkernel, the benchmark with ./build/microbench and the test with ./build/test"

microkernel:
	mkdir -p build
	g++ microkernel.cpp optimization/v1_matmul_16_6_1.s optimization/v2_matmul_16_6_1.s optimization/v3_matmul_16_6_1.s -I$(INCLUDE_DIR) -o build/microkernel

microbench:
	mkdir -p build
	g++ benchmark/microbench.cpp benchmark/b1_matmul_16_6_1.s benchmark/b2_matmul_16_6_1.s benchmark/b3_matmul_16_6_1.s -I$(INCLUDE_DIR) -o build/microbench

test:
	mkdir -p build
	g++ tests/tests.cpp tests/matmul_16_6_1.test.cpp optimization/v1_matmul_16_6_1.s optimization/v2_matmul_16_6_1.s optimization/v3_matmul_16_6_1.s -I$(INCLUDE_DIR) -o build/test

clean:
	rm -rf build