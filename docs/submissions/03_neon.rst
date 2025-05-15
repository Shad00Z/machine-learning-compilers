3. Neon
=========

3.1 Execution Throughput and Latency
-------------------------------------

For this task, we were benchmarking the execution throughput and latency for FP32 neon instructions.
Specifically, we were looking at:

1. `FMLA (vector) <https://developer.arm.com/documentation/ddi0602/2025-03/SIMD-FP-Instructions/FMLA--vector---Floating-point-fused-multiply-add-to-accumulator--vector-->`_ instruction
2. `FMADD (scalar) <https://developer.arm.com/documentation/ddi0602/2025-03/SIMD-FP-Instructions/FMADD--Floating-point-fused-multiply-add--scalar--?lang=en>`_ instruction

3.1.1 Throughput
^^^^^^^^^^^^^^^^^

As a first step we were comparing the throughput of:

1. FMLA (vector) with arrangement specifier ``4S``
2. FMLA (vector) with arrangement specifier ``2S``
3. FMADD (scalar), single-precision variant

To compare the throughput, we created three assembly programs, that were executing
several of these operations. To get proper results we were looking for any dependencies
regarding the source or destination registers of the operations.
The calculations that we were ending up with were:

.. literalinclude:: ../../src/submissions/03_neon/01_execution_throughput_and_latency/fmla_4s_instr.s
    :language: asm
    :lines: 31-71
    :caption: FMLA (vector) with arrangement specifier ``4S``

.. literalinclude:: ../../src/submissions/03_neon/01_execution_throughput_and_latency/fmla_2s_instr.s
    :language: asm
    :lines: 31-71
    :caption: FMLA (vector) with arrangement specifier ``2S``

.. literalinclude:: ../../src/submissions/03_neon/01_execution_throughput_and_latency/fmadd_instr.s
    :language: asm
    :lines: 39-82
    :caption: FMADD (scalar), single-precision variant

In order to measure the throughput of these instructions we developed a C++ microbenchmark.
For each instruction we firstly performed a warm up, measured the time, counted the operations and then calculated the GFLOPs.

.. literalinclude:: ../../src/submissions/03_neon/01_execution_throughput_and_latency/benchmark/microbench.cpp
    :language: cpp
    :lines: 61-73
    :caption: Example benchmark for FMLA (vector) with arrangement specifier ``4S``

For the ``2S`` and the FMADD (scalar) instructions, we simply adjusted the calculations for the operations slightly:

.. literalinclude:: ../../src/submissions/03_neon/01_execution_throughput_and_latency/benchmark/microbench.cpp
    :language: cpp
    :lines: 85-89
    :caption: Calculations for FMLA (vector) with arrangement specifier ``2S``

.. literalinclude:: ../../src/submissions/03_neon/01_execution_throughput_and_latency/benchmark/microbench.cpp
    :language: cpp
    :lines: 101-105
    :caption: Calculations for FMLA (vector) with arrangement specifier ``2S``

For this benchmarking task we obtained the following results:

.. literalinclude:: ../../src/submissions/03_neon/01_execution_throughput_and_latency/benchmark/benchmarking_results.txt
    :language: text
    :lines: 1-23
    :caption: Throughput results for the three instructions

It can be seen that the FMLA (vector) with arrangement specifier ``4S`` instruction performs approximately 
2.5 times as many floating point operations than the FMLA (vector) with arrangement specifier ``2S`` instruction. 
Further the FMLA (vector) with arrangement specifier ``2S`` instructions performs at approximately 2.5 times more
floating point operations than the FMADD (scalar) instruction. 

This shows that leveraging data-level parallelism (vector-based) can yield a much higher throughput, than using only scalar operations.

3.1.2 Latency
^^^^^^^^^^^^^^

To measure the execution latency for FMLA (vector) instructions with arrangement specifier ``4S``, we examined two scenarios:

1. Each instruction depends on the destination register and one of the source register of the previous instruction

.. literalinclude:: ../../src/submissions/03_neon/01_execution_throughput_and_latency/fmla_4s_source_lat_instr.s
    :language: asm
    :lines: 33-36
    :caption: fmla instructions with dependencies on the destination register and one of the source registers

2. Each instruction depends only on the destination register of the previous instruction

.. literalinclude:: ../../src/submissions/03_neon/01_execution_throughput_and_latency/fmla_4s_dest_lat_instr.s
    :language: asm
    :lines: 33-36
    :caption: fmla instructions with dependencies on the destination register

Both files contain 32 fmla instructions each, which are executed 100 times. The results of our benchmark is shown below:

.. literalinclude:: ../../src/submissions/03_neon/01_execution_throughput_and_latency/benchmark/benchmarking_results.txt
    :language: text
    :lines: 25-39
    :caption: Latency results for the two scenarios

We can see that both scenarios have similar results, which is why we computed the latency only for the first scenario.

We measured :math:`1.16266 \times 10^{10}` instructions per second, which means that the latency of the FMLA (vector) instruction with arrangement specifier ``4S`` is approximately :math:`\frac{1}{1.16266 \times 10^{10}} \approx 8.6 \times 10^{-11}` seconds. Using a known clock frequency of 4.4 GHz, we computed the latency as :math:`8.6 \times 10^{-11} \times 4.4 \times 10^9 = 0.3784` cycles.

3.2 Microkernel
--------------------------------

For the second task we were implementing a microkernel to execute a matrix multiplication for matrices with the dimensions:

1. Matrix A: 16 x 1
2. Matrix B: 1 x 6
3. Matrix C: 16 x 6

3.2.1 Neon Microkernel
^^^^^^^^^^^^^^^^^^^^^^^^

We developed three different versions of this microkernel in order to optimize its performance.

In the **first version** we:

1. Load matrix A (16 x 1)
2. Load three columns (1 x 1) of matrix B
3. Load matrix C (16 x 6)

In the **second version** we:

1. Load matrix A (16 x 1)
2. Load one column of matrix B
3. Load matrix C (16 x 6)

In the **third version** we:

1. Load matrix A (16 x 1)
2. Load one column of matrix B
3. Load one column of matrix C (16 x 1)

3.2.2 Testing and Benchmarking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To test and compare our versions with one another we:

1. implemented a microkernel that would give us a visual indication of the results
2. implemented a test using Catch2 to test the correctness of our implementations
3. implemented a microbenchmark that would calculate the GFLOPs for each of the three versions

The GFLOPs were calculated using the following formula:

.. literalinclude:: ../../src/submissions/03_neon/02_microkernel/benchmark/microbench.cpp
    :language: cpp
    :lines: 138-143
    :caption: GFLOPs calculations

For each version we would perform ``50,000`` iterations as a warmup to guarantee similar results for each execution of the benchmark.
Using this approach we obtained the following results:

.. literalinclude:: ../../src/submissions/03_neon/02_microkernel/benchmark/benchmarking_results.txt
    :language: text
    :caption: GFLOPs calculations

The GLFOPs results indicate that with every version we obtained slightly better results, resulting in 
about ``1.7`` GLOPs in difference comparing our best with our worst approach.

.. _3.3 Loops:

3.3 Loops
------------

In this task, we had to add loops to the matrix multiplication kernel which we wrote in the previous task. The goal was to enable the 16x6x1 kernel to be used for larger matrices.

The first step was to implement a loop in the K dimension, resulting in a 16x6x64 kernel. The loading and storing of matrix C was left unchanged. The relevant code is shown below:

.. literalinclude:: ../../src/submissions/03_neon/03_loops/loops/matmul_16_6_64.s
    :language: asm
    :linenos:
    :lines: 67-133
    :caption: Looping matmul_16_6_1 over K dimension

The ``matmul_16_6_1`` kernel mostly stayed the same, except that for each K loop, we now need to adjust the pointers to the input matrices A and B. At the end of each loop, we move the pointers to A to the next column by adding the given stride. In B, we need to move the pointer to the next row. Therefore, we jump by 4 Bytes (since we are using 32-bit floats) from the starting address of B. To keep jumping to the next row in each loop, we accumulate the offset of 4 Bytes in the register ``x9``.

The second step was to implement a loop in the M dimension, resulting in a 64x6x64 kernel. To keep the code examples shorter, we exclude the K loop from the code snippets. The relevant code is shown below:

.. literalinclude:: ../../src/submissions/03_neon/03_loops/loops/matmul_64_6_64.s
    :language: asm
    :linenos:
    :lines: 45-92
    :caption: First part of looping over M dimension

.. literalinclude:: ../../src/submissions/03_neon/03_loops/loops/matmul_64_6_64.s
    :language: asm
    :linenos:
    :lines: 153-193
    :caption: Second part of looping over M dimension

The M loop needs only 4 iterations, since we are extending the kernel from 16 to 64 in the M dimension by dividing the M dimension into 4 blocks of 16 elements. At the end of the M loop, we move the pointers of A and C to the next block. We jump by 16 elements in the M dimension, which means adding 16*4 Bytes to the pointer of A and C.

The last step was to implement a loop in the N dimension, resulting in a 64x48x64 kernel. The relevant code is shown below:

.. literalinclude:: ../../src/submissions/03_neon/03_loops/loops/matmul_64_48_64.s
    :language: asm
    :linenos:
    :lines: 49-66
    :caption: First part of looping over N dimension

.. literalinclude:: ../../src/submissions/03_neon/03_loops/loops/matmul_64_48_64.s
    :language: asm
    :linenos:
    :lines: 205-220
    :caption: Second part of looping over N dimension

Since we are extending the kernel from 6 to 48 in the N dimension, we need to divide the N dimension into 8 blocks of 6 elements. This means that the loop will have 8 iterations. For each N loop, it is important to first reset the pointer of A to the original address. After each iteration, we need to move the pointers of B and C to the next block. To do this, we jump by elements in the N dimension, that is specifically 6 columns of B and C. We do this by adding 6 times the stride of B and C to the pointers.

3.3.1 Testing and Benchmarking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For all three kernels we have written unit tests. To execute the tests, one first needs to compile the code by invoking ``make`` from within the ``src/submissions/03_neon/03_loops`` directory. This will create an executable that can be run with ``./build/test``.

We also calculated the GFLOPs for each of these matrix multiplications.
To calculate them we followed the simple formula:

.. math:: M \cdot N \cdot K \cdot \text{Ops Per FMLA}

The results that we obtained were:

.. literalinclude:: ../../src/submissions/03_neon/03_loops/benchmark/benchmarking_results.txt
    :language: text
    :caption: GFLOPs calculations for MatMuls

Our results indicate that the number of GFLOPs is very consistent, even when scaling the size of our matrix.


3.4 SIMD Lanes
----------------

For this task we were supposed to create two kernels, that should be able 
to function, even if we don't have a multiple of 4 for the ``M`` dimension.
We created several versions for both:

1. the ``M=14``, ``N=6`` and ``K=64``, and
2. the ``M=15``, ``N=6`` and ``K=64``

3.4.1 Matmul_14_6_64
^^^^^^^^^^^^^^^^^^^^^^

For the case ``M=14`` we considered four different versions:

Our **first approach** was to use two loops. The first loop was used to calculate 
a (12 x 64) block of matrix C. That means, we would load 12 column elements of matrix A. 
The second loop was then used to calculate the remaining (2 x 64) block of matrix C.

.. literalinclude:: ../../src/submissions/03_neon/04_simd/matmul_14_6_64/v1_matmul_14_6_64.s
    :language: asm
    :linenos:
    :lines: 157-200
    :caption: Second loop for the (2 x 64) matrix calculation

The **second approach** was to use a single loop. We would load the whole matrix C, and matrix A 
column-wise using one ``ldp qXX, qXX, [x7]``, one ``ldr qXX, [x7, #32]`` and one ``ldr dXX, [x7, #48]`` instruction. 

.. literalinclude:: ../../src/submissions/03_neon/04_simd/matmul_14_6_64/v2_matmul_14_6_64.s
    :language: asm
    :linenos:
    :lines: 86-144
    :caption: Calculate matrix C with a single loop using four loads

Our **third approach** was again to use a single loop. But this time we would load matrix A
column-wise using two ``ldp qXX, qXX, [x7]`` instructions and then set the last two elements
to zero using ``mov v27.s[2], wzr`` and ``mov v27.s[3], wzr``.

.. literalinclude:: ../../src/submissions/03_neon/04_simd/matmul_14_6_64/v3_matmul_14_6_64.s
    :language: asm
    :linenos:
    :lines: 88-148
    :caption: Calculate matrix C with a single loop using ``ldp`` loads

In our **fourth approach** we simply copied the second version and changed
our loads for matrix A and C. We used ``ld1`` instead of ``ldp``.

.. literalinclude:: ../../src/submissions/03_neon/04_simd/matmul_14_6_64/v4_matmul_14_6_64.s
    :language: asm
    :linenos:
    :lines: 73-130
    :caption: Calculate matrix C with a single loop and ``ld1`` loads

When benchmarking our approaches we obtained the following results:

.. literalinclude:: ../../src/submissions/03_neon/04_simd/benchmark/benchmarking_results.txt
    :language: text
    :linenos:
    :lines: 1-31
    :caption: Benchmarking results for matmul_14_6_64 approaches

The results indicate that the version with three different loads performed
best, with an increase of about ``10 GFLOPs``. The switch from ``ldp`` to ``ld1`` however, didn't show 
any significant changes in the number of GFLOPs.

3.4.2 Matmul_15_6_64
^^^^^^^^^^^^^^^^^^^^^^

For the case ``M=15`` we considered three different versions:

For our **first approach** we again considered two loops. Again, the first loop was used to calculate 
a (12 x 64) block of matrix C. 
The second loop was then used to calculate the remaining (3 x 64) block of matrix C.

.. literalinclude:: ../../src/submissions/03_neon/04_simd/matmul_15_6_64/v1_matmul_15_6_64.s
    :language: asm
    :linenos:
    :lines: 211-256
    :caption: Second loop for the (3 x 64) matrix calculation

In the **second approach** we use one loop. We load matrix A column-wise using two 
two ``ldp qXX, qXX, [x7]`` instructions and then set the last element
to zero using ``mov v27.s[3], wzr``.

.. literalinclude:: ../../src/submissions/03_neon/04_simd/matmul_15_6_64/v2_matmul_15_6_64.s
    :language: asm
    :linenos:
    :lines: 99-158
    :caption: Calculate matrix C with a single loop using ``ldp`` loads

In the **third approach** we again changed the load instructions from ``ldp`` to
``ld1``.

.. literalinclude:: ../../src/submissions/03_neon/04_simd/matmul_15_6_64/v3_matmul_15_6_64.s
    :language: asm
    :linenos:
    :lines: 81-139
    :caption: Calculate matrix C with a single loop using ``ld1`` loads

Again, we performed some benchmarks:

.. literalinclude:: ../../src/submissions/03_neon/04_simd/benchmark/benchmarking_results.txt
    :language: text
    :linenos:
    :lines: 41-63
    :caption: Benchmarking results for matmul_15_6_64 approaches

Similar to the benchmarks for the ``matmul_14_6_64`` the approach with the single loop
performs much better than the other approach. This time, we even gain about 
``23 GFLOPs`` with this approach.

.. _generic-kernel:

3.4.3 Generic Approach
^^^^^^^^^^^^^^^^^^^^^^^^

Simply as a proof of concept we also implemented a generic approach for the ``matmul_14_6_64`` and ``matmul_15_6_64`` kernels. This kernel works for any ``M > 0``. The idea is to write specific kernels for ``M = 1, 2, ..., 8``. We then divide M by 8 (shift right by 3) and use that to loop the kernel for ``M = 8``. Basically we split the M dimension into blocks of 8 elements and compute the result using a ``matmul_8_6_64`` kernel. If there is a remainder, it is ``>=1 and <=7``, which we handle with specific kernels. The selection of the specific kernels is done using a jump table.

We also benchmarked the performance of this **generic kernel**:

.. literalinclude:: ../../src/submissions/03_neon/04_simd/benchmark/benchmarking_results.txt
    :language: text
    :linenos:
    :lines: 33-39
    :caption: Benchmarking results for ``matmul_M_6_64`` (M = 14) approach

.. literalinclude:: ../../src/submissions/03_neon/04_simd/benchmark/benchmarking_results.txt
    :language: text
    :linenos:
    :lines: 65-71
    :caption: Benchmarking results for ``matmul_M_6_64`` (M = 15) approach

Compared to our other approaches our obtained GFLOPs are slightly worse, losing
about ``30 GFLOPs`` to our best approach for the ``matmul_14_6_64`` and about 
``40 GFLOPs`` to our best approach for the ``matmul_15_6_64``.

3.5 Accumulator Block Shapes
-----------------------------

In this task we were supposed to implement a microkernel that computes C+=AB for M=64, N=64 and K=64. Recalling our ``matmul_64_48_64`` kernel, we only need to change the N dimension to 64. This kernel uses the ``matmul_16_6_64`` internally, which we changed to ``matmul_16_4_64``. Changing N from 6 to 4 allows us to divide the N dimension into 16 blocks of 4 elements. N = 8 was not suitable, as we ran into issues with the number of available SIMD lanes. We do not think it is necessary to show the code for this kernel, as it is very similar to the ``matmul_64_48_64`` kernel. The only difference is that we removed the logic for 2 of the 6 columns and increased the loop counter constant.

Benchmarking this kernel we obtained the following results:

.. literalinclude:: ../../src/submissions/03_neon/05_accumulator_block_shapes/benchmark/benchmarking_results.txt
    :language: text
    :linenos:
    :caption: Benchmarking results for matmul_64_64_64 approaches

V1 is the first version which we obtained by converting our best performing ``matmul_64_48_64`` kernel. Trying to squeeze out more performance, we made some minor changes to the computations of the strides (as shown below). We also removed loads and stores of callee-saved registers that were not used. This resulted in a performance increase of about 2-3 GFLOPs in V2 across multiple runs.

.. literalinclude:: ../../src/submissions/03_neon/05_accumulator_block_shapes/optimization/v1_matmul_64_64_64.s
    :language: asm
    :linenos:
    :lines: 39-47
    :caption: Naive stride calculations

.. literalinclude:: ../../src/submissions/03_neon/05_accumulator_block_shapes/optimization/v2_matmul_64_64_64.s
    :language: asm
    :linenos:
    :lines: 37-44
    :caption: Optimized stride calculations

3.6 Batch-Reduce GEMM
-----------------------------

Based on the previous tasks, we are now implementing a batch-reduce GEMM kernel. 
The goal is to implement a kernel that computes :math:`C+=\sum_i A_i B_i` for M=64, N=48 and K=64 matrices. 
The kernel should be able to handle batches of matrices. 
For now we are only implementing the case where the batch size is 16.

Similar to the previous tasks we implemented several versions of this kernel to optimize the performance.

In our **first version** we simply used our ``matmul_64_48_64`` kernel from our :ref:`loops <3.3 Loops>` task and looped 16 times around that kernel.
Key points that we needed to consider were the following:

.. literalinclude:: ../../src/submissions/03_neon/06_batch_reduce_gemm/optimization/v1_matmul_64_48_64_16.S
    :language: asm
    :linenos:
    :lines: 56-59
    :caption: Setting the batch counter

.. literalinclude:: ../../src/submissions/03_neon/06_batch_reduce_gemm/optimization/v1_matmul_64_48_64_16.S
    :language: asm
    :linenos:
    :lines: 230-244
    :caption: Jumping to the next matrix A and B in the batch

In our **second version** we made some optimizations to the kernel.
The changes we made were:

.. literalinclude:: ../../src/submissions/03_neon/06_batch_reduce_gemm/optimization/v2_matmul_64_48_64_16.S
    :language: asm
    :linenos:
    :lines: 39-44
    :caption: Replacing ``MUL``'s with ``LSL``'s

.. literalinclude:: ../../src/submissions/03_neon/06_batch_reduce_gemm/optimization/v2_matmul_64_48_64_16.S
    :language: asm
    :linenos:
    :lines: 78-96
    :caption: Replacing all ``LDP``'s with ``LD1``'s and ``STP``'s with ``ST1``'s

These optimizations resulted in a performance increase of about ``3-4 GFLOPs``.

.. literalinclude:: ../../src/submissions/03_neon/06_batch_reduce_gemm/benchmark/benchmarking_results_64_48_64_16.txt
    :language: text
    :linenos:
    :caption: Benchmarking results for the batch-reduce GEMM kernels
