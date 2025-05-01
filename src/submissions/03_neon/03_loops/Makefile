# 02_microkernel -> 03_neon -> submissions -> src ... include
INCLUDE_DIR = ../../../../include

default: test benchmark
	@echo "Build complete. Run the test with ./build/test"

test:
	mkdir -p build
	g++ tests.cpp matmul_16_6_64.test.cpp matmul_16_6_64.s matmul_64_6_64.test.cpp matmul_64_6_64.s matmul_64_48_64.test.cpp matmul_64_48_64.s -I$(INCLUDE_DIR) -o build/test

benchmark:
	mkdir -p build
	g++ microbench.cpp matmul_16_6_64.s matmul_64_6_64.s matmul_64_48_64.s -I$(INCLUDE_DIR) -o build/bench

clean:
	rm -rf build