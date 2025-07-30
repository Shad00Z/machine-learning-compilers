.. _3-neon:

3. Neon
=========

In this section we explore `Neon <https://developer.arm.com/Architectures/Neon>`_, ARM's advanced SIMD (Single Instruction, Multiple Data) architecture extension. Our goal is to understand how to implement Neon kernels and how to optimize them for maximum performance. 

.. _3.1-throughput-latency:

3.1 Execution Throughput and Latency
-------------------------------------

The first task was to benchmark the execution throughput and latency of some selected FP32 Neon instructions. 
Specifically, we were looking at:

1. `FMLA (vector) <https://developer.arm.com/documentation/ddi0602/2025-03/SIMD-FP-Instructions/FMLA--vector---Floating-point-fused-multiply-add-to-accumulator--vector-->`_ instruction
2. `FMADD (scalar) <https://developer.arm.com/documentation/ddi0602/2025-03/SIMD-FP-Instructions/FMADD--Floating-point-fused-multiply-add--scalar--?lang=en>`_ instruction

3.1.1 Throughput
^^^^^^^^^^^^^^^^^

To analyze the throughput, we compared the performance of the following variants:

1. ``FMLA (vector)`` with arrangement specifier ``4S``
2. ``FMLA (vector)`` with arrangement specifier ``2S``
3. ``FMADD (scalar)``, single-precision variant

To compare the throughput of these variants, we created an assembly program for each of them.
To ensure instruction-level parallelism, we carefully designed the inner loops of these programs to avoid register dependencies between successive instructions.
The calculations and loop structures we used are shown here:

.. literalinclude:: ../../src/submissions/03_neon/01_execution_throughput_and_latency/fmla_4s_instr.s
    :language: asm
    :lines: 31-71
    :lineno-match:
    :caption: ``FMLA (vector)`` with arrangement specifier ``4S``

.. literalinclude:: ../../src/submissions/03_neon/01_execution_throughput_and_latency/fmla_2s_instr.s
    :language: asm
    :lines: 31-71
    :lineno-match:
    :caption: ``FMLA (vector)`` with arrangement specifier ``2S``

.. literalinclude:: ../../src/submissions/03_neon/01_execution_throughput_and_latency/fmadd_instr.s
    :language: asm
    :lines: 39-82
    :lineno-match:
    :caption: ``FMADD (scalar)``, single-precision variant

We then implemented a C++ microbenchmark to evaluate each version. 
For each function, we:

1. Performed a warm-up
2. Measured the execution time
3. Counted the number of operations
4. Calculated the resulting GFLOPs

.. literalinclude:: ../../src/submissions/03_neon/01_execution_throughput_and_latency/benchmark/microbench.cpp
    :language: cpp
    :lines: 61-73
    :lineno-match:
    :caption: Example benchmark for ``FMLA 4S``

Calculations for ``2S`` and ``FMADD (scalar)``:

.. literalinclude:: ../../src/submissions/03_neon/01_execution_throughput_and_latency/benchmark/microbench.cpp
    :language: cpp
    :lines: 85-89
    :lineno-match:
    :caption: Calculations for ``FMLA 2S``

.. literalinclude:: ../../src/submissions/03_neon/01_execution_throughput_and_latency/benchmark/microbench.cpp
    :language: cpp
    :lines: 101-105
    :lineno-match:
    :caption: Calculations for ``FMADD``

The measured throughput results we obtained were as follows:

.. literalinclude:: ../../src/submissions/03_neon/01_execution_throughput_and_latency/benchmark/benchmarking_results.txt
    :language: text
    :lines: 1-23
    :lineno-match:
    :caption: Measured Throughput Results

We observe that:

* ``FMLA 4S`` achieves approximately ``2.5`` times the performance of ``FMLA 2S``
* ``FMLA 2S`` similarly outperforms ``FMADD (scalar)`` by a factor of ``2.5``

These results highlight the benefit of data-level parallelism through vector operations. The higher the vector width, the more operations are performed per instruction, therefore resulting in a significantly improved throughput compared to a scalar execution.

3.1.2 Latency
^^^^^^^^^^^^^^

To analyze the execution latency of the ``FMLA vector`` instruction with arrangement specifier ``4S``, we considered two dependency scenarios:

1. Each instruction depends on the destination register **and** one source register of the previous instruction.

.. literalinclude:: ../../src/submissions/03_neon/01_execution_throughput_and_latency/fmla_4s_source_lat_instr.s
    :language: asm
    :lines: 33-36
    :lineno-match:
    :caption: ``FMLA`` instructions with dependencies on the destination and one source registers

2. Each instruction depends **only** on the destination register of the previous instruction

.. literalinclude:: ../../src/submissions/03_neon/01_execution_throughput_and_latency/fmla_4s_dest_lat_instr.s
    :language: asm
    :lines: 33-36
    :lineno-match:
    :caption: ``FMLA`` instructions with dependency only on the destination register

In both cases, 32 dependent ``FMLA`` instructions were executed in a loop, repeated 100 times. 
The results for both cases are shown below:

.. literalinclude:: ../../src/submissions/03_neon/01_execution_throughput_and_latency/benchmark/benchmarking_results.txt
    :language: text
    :lines: 25-39
    :lineno-match:
    :caption: Latency benchmark results for the two dependency scenarios

We observed that both scenarios produced nearly identical performance results. Therefore, we focused our latency calculations only on the first scenario.

From our measurement, we got :math:`1.16266 \times 10^{10}` instructions per second.
This yields a per-instruction latency of approximately :math:`\frac{1}{1.16266 \times 10^{10}} \approx 8.6 \times 10^{-11}` seconds. 
Assuming a clock frequency of ``4.4`` GHz, we estimated the latency in clock cycles as :math:`8.6 \times 10^{-11} \times 4.4 \times 10^9 = 0.3784` cycles.

This value suggests that the latency of a single ``FMLA 4S`` instruction is well below one clock cycle.

.. _3.2-microkernel:

3.2 Microkernel
--------------------------------

For the second task, we implemented a Neon-based microkernel to perform a matrix-matrix multiplication with the following dimensions:

1. Matrix A: ``16 x 1``
2. Matrix B: ``1 x 6``
3. Matrix C: ``16 x 6``

For the task we were provided with the following C function signature:

.. code-block:: c
    :caption: Function Signature

    /**
     * @brief GEMM that computes: C+=AB.
     * @param a    Pointer to column-major matrix A.
     * @param b    Pointer to column-major matrix B.
     * @param c    Pointer to column-major matrix C.
     * @param ld_a Leading dimension of A.
     * @param ld_b Leading dimension of B.
     * @param ld_c Leading dimension of C.
     **/
    void matmul_16_6_1( float   const * a,
                        float   const * b,
                        float         * c,
                        int64_t         ld_a,
                        int64_t         ld_b,
                        int64_t         ld_c );

3.2.1 Neon Microkernel
^^^^^^^^^^^^^^^^^^^^^^^^

We developed three different versions of this microkernel. With each version, we wanted to compare different data-loading, register usage and data reuse strategies:

The **first version**:

1. Load the entire Matrix A (16 x 1)
2. Load **three** individual elements (1 x 1) of Matrix B
3. Load the entire Matrix C (16 x 6)

.. code-block:: asm
    :caption: `b1_matmul_16_6_1.s <https://github.com/Shad00Z/machine-learning-compilers/blob/main/src/submissions/03_neon/02_microkernel/benchmark/b1_matmul_16_6_1.s>`_

    /*
     * Load 3 elements of B
     */
    mov x6, x1              // current column of B

    ldr s28, [x6]           // Column B(0)
    add x6, x6, x4

    ldr s29, [x6]           // Column B(1)
    add x6, x6, x4

    ldr s30, [x6]           // Column B(2)
    add x6, x6, x4
    
    /*
     * Multiply and accumulate (1 / 2)
     */ 
    fmla v4.4s, v0.4s, v28.s[0]
    fmla v5.4s, v1.4s, v28.s[0]
    fmla v6.4s, v2.4s, v28.s[0]
    fmla v7.4s, v3.4s, v28.s[0]

    fmla v8.4s,  v0.4s, v29.s[0]
    fmla v9.4s,  v1.4s, v29.s[0]
    fmla v10.4s, v2.4s, v29.s[0]
    fmla v11.4s, v3.4s, v29.s[0]

    fmla v12.4s, v0.4s, v30.s[0]
    fmla v13.4s, v1.4s, v30.s[0]
    fmla v14.4s, v2.4s, v30.s[0]
    fmla v15.4s, v3.4s, v30.s[0]

The **second version**:

1. Load the entire Matrix A (16 x 1)
2. Load **one** element of Matrix B
3. Load the entire Matrix C (16 x 6)

.. code-block:: asm
    :caption: `b2_matmul_16_6_1.s <https://github.com/Shad00Z/machine-learning-compilers/blob/main/src/submissions/03_neon/02_microkernel/benchmark/b2_matmul_16_6_1.s>`_

    /*
     * Load column of B (1 / 6)
     */
    mov x6, x1              // current column of B

    ldr s28, [x6]           // Column B(0)
    add x6, x6, x4
    
    /*
     * Multiply and accumulate (1 / 6)
     */ 
    fmla v4.4s, v0.4s, v28.s[0]
    fmla v5.4s, v1.4s, v28.s[0]
    fmla v6.4s, v2.4s, v28.s[0]
    fmla v7.4s, v3.4s, v28.s[0]

The **third version**:

1. Load the entire Matrix A (16 x 1)
2. Load **one** column of Matrix B
3. Load **one** column of Matrix C (16 x 1)

.. code-block:: asm
    :caption: `b3_matmul_16_6_1.s <https://github.com/Shad00Z/machine-learning-compilers/blob/main/src/submissions/03_neon/02_microkernel/benchmark/b3_matmul_16_6_1.s>`_

    /*
     * Matrix C: Column 0
     */
    // Load column of B
    ldr s8, [x6]

    // Load column of C
    ldp q4, q5, [x7]
    ldp q6, q7, [x7, #32]

    // Multiply and accumulate
    fmla v4.4s, v0.4s, v8.s[0]
    fmla v5.4s, v1.4s, v8.s[0]
    fmla v6.4s, v2.4s, v8.s[0]
    fmla v7.4s, v3.4s, v8.s[0]

    // Store column of C
    stp q4, q5, [x7]
    stp q6, q7, [x7, #32]

3.2.2 Testing and Benchmarking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To validate and compare our implementations, we took the following steps:

1. We developed a basic `kernel driver <https://github.com/Shad00Z/machine-learning-compilers/blob/main/src/submissions/03_neon/02_microkernel/benchmark/microbench.cpp>`_ to inspect our output correctness visually
2. We used Catch2 to verify the correctness of our implementations
3. We implemented a benchmark to measure ``GFLOPs`` for all three versions

The ``GFLOPs`` were calculated with the following formula:

.. code-block:: cpp
    :caption: ``GFLOPs`` calculation

    double totalOps = ( 6 * 16 ) * 2;
    double opsPerIteration = totalOps * loopIterations;

    double opsPerSec = opsPerIteration / elapsedTime;
    double gflops = opsPerIteration / ( elapsedTime * 1e9 );

Each kernel was executed with ``50,000`` warmup iterations to reduce variability and ensure fair comparisons. 
The benchmark produced the following performance results:

.. literalinclude:: ../../src/submissions/03_neon/02_microkernel/benchmark/benchmarking_results.txt
    :language: text
    :caption: ``GFLOPs`` results for all three versions

The results show that performance improved incrementally with each version. The best-performing kernel outperformed the least-performing by approximately ``1.7 GFLOPs``, highlighting the importance of careful memory and register management.

.. note::
    Even though the third implementation achieved the best performance, it is tailored specifically to the given matrix dimensions. As the ``K`` dimension increases, the kernel would repeatedly reload columns of matrix ``C``, leading to significant performance degradation.

.. _3.3 Loops:

3.3 Loops
------------

After implementing and benchmarking our initial ``16x6x1`` Neon microkernel, the next step was to scale this kernel for the use with larger matrices. To achieve this, we extended the kernel along the three matrix dimensions ``K``, ``M``, and ``N``, by introducing loops around our base kernel.

3.3.1 Loop Implementations 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our **first** step was to handle larger ``K`` dimensions. Therefore, we transformed our kernel into a ``16x6x64`` kernel. The core of the microkernel remained mostly unchanged, except that we updated the input pointers for matrices ``A`` and ``B`` in each iteration:

* ``A`` is advanced by the given stride to move to the next column.
* ``B`` is advanced row-by-row, with a 4-byte step for each 32-bit float value.

The updated loop body is shown below:

.. literalinclude:: ../../src/submissions/03_neon/03_loops/loops/matmul_16_6_64.s
    :language: asm
    :lines: 67-133
    :lineno-match:
    :caption: Looping the ``matmul_16_6_1`` kernel over the ``K`` dimension

The ``matmul_16_6_1`` kernel mostly stayed the same, except that for each K loop, we now need to adjust the pointers to the input matrices A and B. At the end of each loop, we move the pointers to A to the next column by adding the given stride. In B, we need to move the pointer to the next row. Therefore, we jump by 4 Bytes (since we are using 32-bit floats) from the starting address of B. To keep jumping to the next row in each loop, we accumulate the offset of 4 Bytes in the register ``x9``.

In the **second** step we added a loop over the ``M`` dimension to build a ``64x6x64`` kernel. In this version, we reused the kernel for processing 16 rows of ``M`` at a time and iterated four times to cover all 64 rows. That means, at the end of the ``M`` loop, we advance the pointers of ``A`` and ``C`` to the next block.

.. literalinclude:: ../../src/submissions/03_neon/03_loops/loops/matmul_64_6_64.s
    :language: asm
    :lines: 45-92
    :lineno-match:
    :caption: First part of looping over the ``M`` dimension

.. literalinclude:: ../../src/submissions/03_neon/03_loops/loops/matmul_64_6_64.s
    :language: asm
    :lines: 153-193
    :lineno-match:
    :caption: Second part of looping over the ``M`` dimension

The **third** step was to implement a loop in the N dimension, extending the kernel to handle a ``64x48x64`` matrix multiplication. This required dividing ``N`` into 8 blocks of 6 columns, resulting in 8 loop iterations. For each ``N`` loop, it is important to first reset the pointer of ``A`` to the original address. After each iteration, we need to move the pointers of ``B`` and ``C`` to the next block: 

.. literalinclude:: ../../src/submissions/03_neon/03_loops/loops/matmul_64_48_64.s
    :language: asm
    :lines: 49-66
    :lineno-match:
    :caption: First part of looping over the ``N`` dimension

.. literalinclude:: ../../src/submissions/03_neon/03_loops/loops/matmul_64_48_64.s
    :language: asm
    :lines: 205-220
    :lineno-match:
    :caption: Second part of looping over the ``N`` dimension

3.3.2 Testing and Benchmarking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To ensure correctness, we wrote unit tests for all three of our kernels. To execute the tests, we need to step in the correct directory (``src/submissions/03_neon/03_loops``) and compile the code by invoking ``make``. This will create an executable that can be run with ``./build/test``.

We also benchmarked each kernel to measure their performance in ``GFLOPs``, using the standard formula:

.. math:: M \cdot N \cdot K \cdot \text{Ops Per FMLA}

The benchmarking results that we obtained are:

.. literalinclude:: ../../src/submissions/03_neon/03_loops/benchmark/benchmarking_results.txt
    :language: text
    :lineno-match:
    :caption: ``GFLOPs`` calculations for the MatMul kernels

Our results indicate that the number of ``GFLOPs`` is very consistent, even when scaling the size of our matrices.

.. _3.4 SIMD:

3.4 SIMD Lanes
----------------

In this task, our goal was to implement two kernels capable of handling cases where the ``M`` dimension is not a multiple of 4. Specifically, we focused on the following matrix shapes:

1. the ``M=14``, ``N=6`` and ``K=64``, and
2. the ``M=15``, ``N=6`` and ``K=64``

3.4.1 Matmul_14_6_64
^^^^^^^^^^^^^^^^^^^^^^

For the case ``M=14``, we explored four different implementations:

In our **first approach** we used two loops. The first loop computes a ``12 x 64`` block of matrix C, while the second loop handles the remaining ``2 x 64`` block.

.. literalinclude:: ../../src/submissions/03_neon/04_simd/matmul_14_6_64/v1_matmul_14_6_64.s
    :language: asm
    :lines: 157-200
    :lineno-match:
    :caption: Second loop for the ``2 x 64`` matrix calculation

For our **second approach** we used a single loop. Here, we load the entire matrix ``C`` and process each column of ``A`` in a loop iteration using three ``FMLA (4s)`` instructions and one ``FMLA (2s)`` instruction.

.. literalinclude:: ../../src/submissions/03_neon/04_simd/matmul_14_6_64/v2_matmul_14_6_64.s
    :language: asm
    :lines: 86-145
    :lineno-match:
    :caption: Calculating matrix ``C`` with a single loop using different calculations

In our **third approach** we were also using the single loop version, but this time we padded a register of ``A``, that holds the remaining two values with two zero values using ``mov v27.s[2], wzr`` and ``mov v27.s[3], wzr``. This allows us to use four ``FMLA (4s)`` instructions.

.. literalinclude:: ../../src/submissions/03_neon/04_simd/matmul_14_6_64/v3_matmul_14_6_64.s
    :language: asm
    :lines: 88-150
    :lineno-match:
    :caption: Using zero-padding to use four ``FMLA (4s)`` instructions

In our **fourth approach** we simply copied the second version and changed our loads for matrix A and C. We used ``ld1`` instead of ``ldp``.

.. literalinclude:: ../../src/submissions/03_neon/04_simd/matmul_14_6_64/v4_matmul_14_6_64.s
    :language: asm
    :lines: 73-131
    :lineno-match:
    :caption: Single loop version using ``ld1`` loads

To compare our different versions, we performed benchmarks on each kernel. Our benchmarking results are as follows:

.. literalinclude:: ../../src/submissions/03_neon/04_simd/benchmark/benchmarking_results.txt
    :language: text
    :lines: 1-31
    :lineno-match:
    :caption: Benchmarking results for ``matmul_14_6_64`` approaches

Our results indicate that the second version, using three ``FMLA (4s)`` instructions and one ``FMLA (2s)`` instruction, achieved the best performance. The version using ``ld1`` loads achieved a similar ``GFLOPs``. 

3.4.2 Matmul_15_6_64
^^^^^^^^^^^^^^^^^^^^^^

For the case ``M=15``, we implemented and tested three kernels:

In our **first approach** we similarly to the ``M=14`` case, split the computation into two loops. The first loop handles a ``12 x 64`` block, and the second loop processed the remaining ``3 x 64`` block of matrix ``C``.

.. literalinclude:: ../../src/submissions/03_neon/04_simd/matmul_15_6_64/v1_matmul_15_6_64.s
    :language: asm
    :lines: 211-256
    :lineno-match:
    :caption: Second loop for the ``3 x 64`` matrix calculation

In the **second approach** we implement the kernel using a single loop. We load matrix ``A`` column-wise and calculate parts of matrix ``C`` using four ``FMLA (4s)`` instructions. We zeroed out the last element in the final vector register of matrix ``A`` with ``mov v27.s[3], wzr`` to safely operate on a full register.

.. literalinclude:: ../../src/submissions/03_neon/04_simd/matmul_15_6_64/v2_matmul_15_6_64.s
    :language: asm
    :lines: 103-166
    :lineno-match:
    :caption: Single loop version using ``ldp`` loads

In the **third approach** we again changed the load instructions from ``ldp`` to ``ld1``.

.. literalinclude:: ../../src/submissions/03_neon/04_simd/matmul_15_6_64/v3_matmul_15_6_64.s
    :language: asm
    :lines: 85-147
    :lineno-match:
    :caption: Single loop version using ``ld1`` loads

For these kernels we also executed benchmarks:

.. literalinclude:: ../../src/submissions/03_neon/04_simd/benchmark/benchmarking_results.txt
    :language: text
    :lines: 41-63
    :lineno-match:
    :caption: Benchmarking results for ``matmul_15_6_64`` approaches

Similar to the benchmarks for the ``matmul_14_6_64``, the single loop approach significantly outperformed the two loop implementation. In this case, the difference was even bigger, with a gap of approximately ``20 GFLOPs``. However, unlike before, changing the load instruction from ``ldp`` to ``ld1`` had no impact on the overall performance.

.. _generic-kernel:

3.4.3 Generic Approach
^^^^^^^^^^^^^^^^^^^^^^^^

As a proof of concept, we also implemented a generic matrix multiplication kernel capable of handling any ``M > 0``. The core idea is to write specific kernels for ``M = 1, 2, ..., 8``. For input sizes larger than ``M = 8``, we then divide ``M`` by 8 (shift right by 3) and use that to loop the ``M = 8`` kernel, which is basically a ``matmul_8_6_64`` kernel. Any remaining elements (``1 <= M % 8 <= 7``) are handled by using on of the smaller specialized remainder kernels. To enable this dynamic selection, we employ a jump table that maps the remainder values to their respective kernel entry points.

We also benchmarked the performance of this **generic kernel**:

.. literalinclude:: ../../src/submissions/03_neon/04_simd/benchmark/benchmarking_results.txt
    :language: text
    :lines: 33-39
    :lineno-match:
    :caption: Benchmarking results for ``matmul_M_6_64`` (``M = 14``) approach

.. literalinclude:: ../../src/submissions/03_neon/04_simd/benchmark/benchmarking_results.txt
    :language: text
    :lines: 65-71
    :lineno-match:
    :caption: Benchmarking results for ``matmul_M_6_64`` (``M = 15``) approach

Compared to our best fixed-size implementations, the generic kernel shows a slightly lower performance, approximately ``20 GFLOPs`` lower for ``M = 14``, and around ``45 GFLOPs`` lower for ``M = 15``. This performance gap is expected due to the overhead of dynamic branching and the generalization of memory access patterns. 

.. _3.5 Accumulator Block Shapes:

3.5 Accumulator Block Shapes
-----------------------------

In this task, we were supposed to implement a microkernel that computes ``C+=AB`` for ``M=64``, ``N=64`` and ``K=64``. 

Starting from our previous ``matmul_64_48_64`` kernel, we adapted the implementation to support ``N=64``. Internally, this kernel relies on a smaller microkernel. In our previous version, we used ``matmul_16_6_64``, which we replaced with ``matmul_16_4_64``. Reducing ``N`` from 6 to 4 allows us to split the ``N`` dimension into 16 blocks of 4 columns. We found that using ``N=8`` caused issues due to the limited number of available SIMD registers, which made register allocation and performance tuning more difficult. 

Since this kernel is very similar to our earlier ``matmul_64_48_64`` implementation, we chose to not include the code for this kernel here. 

The benchmarking results for the new kernel are shown below:

.. literalinclude:: ../../src/submissions/03_neon/05_accumulator_block_shapes/benchmark/benchmarking_results.txt
    :language: text
    :lineno-match:
    :caption: Benchmarking results for ``matmul_64_64_64`` approaches

Version V1was directly derived from the ``matmul_64_48_64`` kernel. Trying to improve its performance, we introduced minor optimizations to the stride calculations and removed unnecessary loads and stores of callee-saved registers that were not used. These adjustments led to a consistent performance improvement of ``2-3 GFLOPs``, resulting in version V2.

Below, we compare the naive and the optimized stride calculations:

.. literalinclude:: ../../src/submissions/03_neon/05_accumulator_block_shapes/optimization/v1_matmul_64_64_64.s
    :language: asm
    :lines: 39-47
    :lineno-match:
    :caption: Naive stride calculations

.. literalinclude:: ../../src/submissions/03_neon/05_accumulator_block_shapes/optimization/v2_matmul_64_64_64.s
    :language: asm
    :lines: 37-44
    :lineno-match:
    :caption: Optimized stride calculations

.. _3.6 Batch-Reduce GEMM:

3.6 Batch-Reduce GEMM
-----------------------------

Based on the previous tasks, we now implement a batch-reduce GEMM (BRGEMM) kernel. 
The goal is to implement a kernel that computes the operation :math:`C+=\sum_i A_i B_i` for batched matrix inputs, with ``M=64``, ``N=48``, and ``K=64``. The kernel should be able to handle batches of matrices. For now, we restrict the implementation to the case where the batch size is 16.

Similar to the previous tasks, we developed and benchmarked multiple versions of the kernel to optimize for performance.

In our **first version** we used our ``matmul_64_48_64`` kernel from our :ref:`loops section<3.3 Loops>`. We wrapped it inside a loop that runs 16 times, once for each matrix pair in the batch.
Two key aspects that we addressed were the following:

.. code-block:: asm
    :caption: Setting the batch counter

    // Batch counter
    mov x24, #16

    _n_batch:

    ... 

    sub x24, x24, #1

    cbnz x24, _n_batch
    // END N BATCH

.. literalinclude:: ../../src/submissions/03_neon/06_batch_reduce_gemm/optimization/v1_matmul_64_48_64_16.S
    :language: asm
    :lines: 230-244
    :lineno-match:
    :caption: Jumping to the next matrix ``A`` and ``B`` in the batch

In our **second version**, we applied some optimizations to the kernel. The changes we made were:

.. literalinclude:: ../../src/submissions/03_neon/06_batch_reduce_gemm/optimization/v2_matmul_64_48_64_16.S
    :language: asm
    :lines: 39-44
    :lineno-match:
    :caption: Replacing ``MUL``'s with ``LSL``'s

.. literalinclude:: ../../src/submissions/03_neon/06_batch_reduce_gemm/optimization/v2_matmul_64_48_64_16.S
    :language: asm
    :lines: 78-96
    :lineno-match:
    :caption: Replacing all ``LDP``'s with ``LD1``'s and ``STP``'s with ``ST1``'s

These optimizations resulted in a performance improvement of about ``3-4 GFLOPs``.

.. literalinclude:: ../../src/submissions/03_neon/06_batch_reduce_gemm/benchmark/benchmarking_results_64_48_64_16.txt
    :language: text
    :lineno-match:
    :caption: Benchmarking results for the batch-reduce GEMM kernels

.. _3.7 Transposition:

3.7 Transposition
-----------------------------

In this task, we explored how to transpose an ``8x8`` matrix using Neon assembly instructions. Our approach was to first develop a solution for the simpler ``4x4`` case. 

3.7.1 Transposition Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To begin, we loaded all 4 columns of matrix A using ``ldr qX, [x0]``, so that the entire matrix is placed in our registers. The second step would be to transpose the matrix: 

.. literalinclude:: ../../src/submissions/03_neon/07_transposition/optimization/trans_neon_4_4.S
    :language: asm
    :lines: 56-74
    :lineno-match:
    :caption: ``trans_4_4`` implementation

The idea of ``trn1`` and ``trn2`` is to prepare the elements for each column, so that 
we can then leverage their new structure using ``zip1`` and ``zip2``.

To scale this to an ``8x8`` matrix, we divided the matrix into four ``4x4`` submatrices:

.. image:: ../_static/submatrices.png
  :width: 400
  :alt: Matrix divided in four quadrants

Each quadrant was transposed independently using our ``trans_4_4`` kernel:

1. The upper-left matrix (in the image A) was transposed and stored at the same position.
2. The upper-right matrix (in the image B) was transposed and stored into the position originally occupied by the bottom-left matrix.
3. The bottom-left matrix (in the image C) would be transposed and stored into the position originally occupied by the upper-right matrix.
4. The bottom-right matrix (in the image D) was transposed and stored at the same position.

.. literalinclude:: ../../src/submissions/03_neon/07_transposition/optimization/trans_neon_8_8.S
    :language: asm
    :lines: 89-184
    :lineno-match:
    :caption: Transposing and swapping upper-right and bottom-left submatrices

To optimize our initial implementation, we removed the PCS for all regsiters that we didn't use. 
We also restructured our code for clarity and compactness.

.. literalinclude:: ../../src/submissions/03_neon/07_transposition/optimization/v2_trans_neon_8_8.S
    :language: asm
    :lines: 40-121
    :lineno-match:
    :caption: Optimized second version of the transposition kernel

3.7.2 Performance Measuring
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We measured the throughput of our transposition kernel in terms of memory transfer speed, since the core performance factor in this case is loading and storing elements efficiently.

.. literalinclude:: ../../src/submissions/03_neon/07_transposition/benchmark/benchmarking_results.txt
    :language: text
    :lineno-match:
    :caption: ``trans_8_8`` performance in ``GiB/s``

Our benchmarking results show that the initial version achieved approximately ``81 GiB/s``. With our optimizations, we increased this to about ``113 GiB/s``.
