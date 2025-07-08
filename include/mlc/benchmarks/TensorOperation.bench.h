#ifndef TENSOR_OPERATION_BENCH_H
#define TENSOR_OPERATION_BENCH_H
#include <mlc/TensorOperation.h>
#include <mlc/benchmarks/Benchmark.h>
#include <mlc/types.h>
namespace mini_jit
{
    namespace benchmarks
    {
        class TensorOperationBench : public Benchmark
        {
        public:
            /**
             * @brief Constructor for the benchmark for tensor operations.
             * @param run_time The time to run the benchmark in seconds.
             * @param dtype             Datatype of all tensor elements.
             * @param prim_first_touch  Type of the first touch primitive.
             * @param prim_main         Type of the main primitive.
             * @param prim_last_touch   Type of the last touch primitive.
             * @param dim_types         Dimension type of the loops (c, m, n, or k).
             * @param exec_types        Execution type of the loops (seq, shared, or prim).
             * @param dim_sizes         Sizes of the dimensions.
             * @param strides_in0       Strides of the first input tensor.
             * @param strides_in1       Strides of the second input tensor (ignored if unary).
             * @param strides_out       Strides of the output tensor.
             */
            TensorOperationBench(double                   run_time,
                                 dtype_t                  dtype,
                                 ptype_t                  prim_first_touch,
                                 ptype_t                  prim_main,
                                 ptype_t                  prim_last_touch,
                                 std::span<const dim_t>   dim_types,
                                 std::span<const exec_t>  exec_types,
                                 std::span<const int64_t> dim_sizes,
                                 std::span<const int64_t> strides_in0,
                                 std::span<const int64_t> strides_in1,
                                 std::span<const int64_t> strides_out);
            //! Destructor
            ~TensorOperationBench() override = default;
            //! Runs the benchmark.
            void run() override;

        private:
            double                    m_run_time;
            mini_jit::TensorOperation m_tensor_op;
            std::vector<dim_t>        m_dim_types;
            std::vector<int64_t>      m_dim_sizes;
        };
    } // namespace benchmarks
} // namespace mini_jit

#endif // TENSOR_OPERATION_BENCH_H