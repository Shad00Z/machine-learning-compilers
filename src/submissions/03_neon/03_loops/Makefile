# 02_microkernel -> 03_neon -> submissions -> src ... include
INCLUDE_DIR = ../../../../include

default: test microbench
	@echo "Build complete. Run the test with ./build/test and the benchmark with ./build/microbench"

test:
	mkdir -p build
	g++ tests/tests.cpp tests/matmul_16_6_64.test.cpp loops/matmul_16_6_64.s tests/matmul_64_6_64.test.cpp loops/matmul_64_6_64.s tests/matmul_64_48_64.test.cpp loops/matmul_64_48_64.s -I$(INCLUDE_DIR) -o build/test

microbench:
	mkdir -p build
	g++ benchmark/microbench.cpp loops/matmul_16_6_64.s loops/matmul_64_6_64.s loops/matmul_64_48_64.s -I$(INCLUDE_DIR) -o build/microbench

clean:
	rm -rf build