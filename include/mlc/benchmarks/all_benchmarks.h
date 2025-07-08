#ifndef ALL_BENCHMARKS_H
#define ALL_BENCHMARKS_H

#include <mlc/benchmarks/EinsumTree.bench.h>
#include <mlc/benchmarks/TensorOperation.bench.h>
#include <mlc/benchmarks/matmul/Matmul_br_m_n_k.bench.h>
#include <mlc/benchmarks/matmul/Matmul_m_n_k.bench.h>
#include <mlc/benchmarks/unary/fast_sigmoid_primitive.bench.h>
#include <mlc/benchmarks/unary/identity_primitive.bench.h>
#include <mlc/benchmarks/unary/identity_trans_primitive.bench.h>
#include <mlc/benchmarks/unary/reciprocal_primitive.bench.h>
#include <mlc/benchmarks/unary/relu_primitive.bench.h>
#include <mlc/benchmarks/unary/relu_trans_primitive.bench.h>
#include <mlc/benchmarks/unary/sigmoid_interpolation_primitive.bench.h>
#include <mlc/benchmarks/unary/sigmoid_taylor_primitive.bench.h>
#include <mlc/benchmarks/unary/square_primitive.bench.h>
#include <mlc/benchmarks/unary/square_trans_primitive.bench.h>
#include <mlc/benchmarks/unary/zero_eor_primitive.bench.h>
#include <mlc/benchmarks/unary/zero_xzr_primitive.bench.h>

#endif // ALL_BENCHMARKS_H