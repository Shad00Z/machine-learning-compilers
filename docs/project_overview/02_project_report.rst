.. _project-report:

Project Report
===============

In the first two weeks of our project, we focused on ARM AArch64 assembly to build a solid foundation for understanding machine learning compilers. During the first week, we examined the assembly output of a simple C program compiled with both **GCC** and **Clang** to explore the differences in code generation and function call conventions. 

This initial exploration helped us become familiar with compiler behavior, instruction-level representations, and low-level debugging tools such as **GDB**. 

Further details about the specific steps and tasks can be found in the :ref:`assembly section<Assembly>` of our project documentation.

In the second week, we began writing assembly programs from scratch using only base instructions. Specifically, we reimplemented two simple C functions in AArch64 assembly to better understand data movements and control flow at the instruction level. 

After successfully replicating the functionality of the C programs, we developed microbenchmarks to evaluate the **throughput** and **latency** of key instructions such as ``ADD`` and ``MUL``. These benchmarks helped us gain insight into the performance characteristics of modern ARM processors and how instruction-level behavior can impact overall computation speed.

Further details about the specific steps and tasks can be found in the :ref:`base section<Base>` of our project documentation.

After spending the first two weeks experimenting with base instructions and writing simple assembly programs, we advanced to working with **Neon** (**Advanced SIMD**) instructions. In the following weeks, we explored the performance characteristics of Neon operations, an essential step toward mastering the fundamentals required for building our own tensor compiler.

We began by benchmarking the :ref:`throughput and latency<3.1-throughput-latency>` of the ``FMLA`` and ``FMADD`` instructions. This helped us understand the significance of instruction-level parallelism and how much instruction ordering and data dependencies can impact performance. 

After these initial experiments, we shifted our focus toward understanding the role of **microkernels**. We explored key design considerations, like data reuse, register allocation, and memory access patterns, and conducted our first experiments with optimizing :ref:`microkernels<3.2-microkernel>` for performance. 

In the following week, we extended our microkernel by wrapping it in **loops** to support larger matrix dimensions and improve the performance. Starting from our base kernel of ``16x6x1``, we progressively scaled it to handle matrices of size ``64x48x64``. This allowed us to reach the architectural performance limits of a M4 Chip. Further implementation details can be found in the :ref:`loops section <3.3 Loops>` of our project documentation. 

After exploring ideal kernel sizes aligned with our vector register widths, we also experimented with cases where the ``M`` dimension is not a multiple of 4 or 16. In these scenarios, special handling was required to maintain correctness and efficiency. We implemented and optimized dedicated kernels for two such cases, which are documented in detail in the :ref:`SIMD Lanes <3.4 SIMD>` section of our documentation. 

In the same week, we also explored the impact of **accumulator block** shapes for performance reasons. Specifically, we implemented a microkernel for computing ``C+=AB`` with dimensions ``M=64``, ``N=64``, and ``K=64``. This required adapting our existing ``matmul_64_48_64`` kernel to support the extended ``N`` dimension. The details and benchmarking results of this extension are documented in the :ref:`Accumulator Block Shapes<3.5 Accumulator Block Shapes>` section. 

After implementing and optimizing standard matrix multiplication kernels, we extended our work to support :ref:`batch-reduce GEMM<3.6 Batch-Reduce GEMM>` operations. We reused and adapted our existing microkernels to handle **batches of matrices** efficiently. 

The last part of the Neon section, was to explore, how to **transpose** matrices using Neon assembly. Our goal was to handle a ``8x8`` matrix, which we handled by dividing it into four ``4x4`` submatrices. More details and code can be found in the :ref:`transposition section<3.7 Transposition>` of our documentation. 

In week 4 ... (we also started with the jitter)

In week 5 ... (big 8h benchmark)

In week 6 ... (unaries implemented)