# 01_bench

default: microbench
	@echo "Build complete. Run the benchmark with ./build/microbench"

microbench:
	mkdir -p build
	g++ benchmark/microbench.cpp fmadd_instr.s fmla_2s_instr.s fmla_4s_dest_lat_instr.s fmla_4s_instr.s fmla_4s_source_lat_instr.s -o build/microbench

clean:
	rm -rf build
