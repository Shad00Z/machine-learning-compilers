# general_kernel -> 04_simd -> 03_neon -> submissions -> src ... include
INCLUDE_DIR = ../../../../../include

default: test
	@echo "Build complete. Run the test with ./build/test"

test:
	mkdir -p build
	g++ tests.cpp matmul_M8_6_64.test.cpp matmul_M8_6_64.s -I$(INCLUDE_DIR) -o build/test

clean:
	rm -rf build