#ifndef ALL_BENCHMARKS_H
#define ALL_BENCHMARKS_H

#include "matmul/Matmul_br_m_n_k.bench.h"
#include "matmul/Matmul_m_n_k.bench.h"

#include "unary/fast_sigmoid_primitive.bench.h"
#include "unary/identity_primitive.bench.h"
#include "unary/identity_trans_primitive.bench.h"
#include "unary/reciprocal_primitive.bench.h"
#include "unary/relu_primitive.bench.h"
#include "unary/relu_trans_primitive.bench.h"
#include "unary/square_primitive.bench.h"
#include "unary/square_trans_primitive.bench.h"
#include "unary/zero_eor_primitive.bench.h"
#include "unary/zero_xzr_primitive.bench.h"

#include "EinsumTree.bench.h"
#include "TensorOperation.bench.h"

#endif // ALL_BENCHMARKS_H