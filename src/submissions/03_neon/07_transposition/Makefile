# 07_transposition -> 03_neon -> submissions -> src ... include
INCLUDE_DIR = ../../../../include

default: test microbench
	@echo "Build complete. Run the test with ./build/test"

test:
	mkdir -p build
	g++ tests/tests.cpp tests/trans_neon_4_4.test.cpp tests/trans_neon_8_8.test.cpp optimization/trans_neon_8_8.S optimization/trans_neon_4_4.S -I$(INCLUDE_DIR) -o build/test

microbench:
	mkdir -p build
	g++ benchmark/microbench.cpp optimization/trans_neon_8_8.S optimization/trans_neon_4_4.S -I$(INCLUDE_DIR) -o build/microbench

clean:
	rm -rf build