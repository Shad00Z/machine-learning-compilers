#ifndef EINSUM_TREE_BENCH_H
#define EINSUM_TREE_BENCH_H
#include <mlc/benchmarks/Benchmark.h>
#include <mlc/einsum/EinsumNode.h>
#include <mlc/einsum/EinsumTree.h>
#include <mlc/types.h>
#include <string>
#include <vector>
namespace mini_jit
{
    namespace benchmarks
    {
        class EinsumTreeBench : public Benchmark
        {
        public:
            EinsumTreeBench(double                              run_time,
                            std::string const&                  einsum_expression,
                            std::vector<int64_t>&               dimension_sizes,
                            mini_jit::dtype_t                   dtype,
                            int64_t                             thread_target,
                            int64_t                             max_kernel_size,
                            int64_t                             min_kernel_size,
                            std::map<std::string, void const*>& tensor_inputs);
            //! Destructor
            ~EinsumTreeBench() override
            {
                if (m_root_node != nullptr)
                {
                    delete m_root_node;
                }
            }
            //! Runs the benchmark.
            void run() override;

        private:
            double                             m_run_time;
            std::vector<int64_t>               m_dimension_sizes;
            std::map<std::string, void const*> m_tensor_inputs;
            mini_jit::einsum::EinsumNode*      m_root_node = nullptr;
        };
    } // namespace benchmarks
} // namespace mini_jit

#endif // EINSUM_TREE_BENCH_H