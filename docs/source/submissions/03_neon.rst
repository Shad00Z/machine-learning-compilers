Neon
======

Execution Throughput and Latency
--------------------------------

For this task, we were benchmarking the execution throughput and latency for FP32 neon instructions.
Specifically, we were looking at:

1. `FMLA (vector) <https://developer.arm.com/documentation/ddi0602/2025-03/SIMD-FP-Instructions/FMLA--vector---Floating-point-fused-multiply-add-to-accumulator--vector-->`_ instruction
2. `FMADD (scalar) <https://developer.arm.com/documentation/ddi0602/2025-03/SIMD-FP-Instructions/FMADD--Floating-point-fused-multiply-add--scalar--?lang=en>`_ instruction

Throughput
^^^^^^^^^^^

As a first step we were comparing the throughput of:

1. FMLA (vector) with arrangement specifier ``4S``
2. FMLA (vector) with arrangement specifier ``2S``
3. FMADD (scalar), single-precision variant

To compare the throughput, we created three assembly programs, that were executing
several of these operations. To get proper results we were looking for any dependencies
regarding the source or destination registers of the operations.
The calculations that we were ending up with were:

.. literalinclude:: ../../../src/submissions/03_neon/01_execution_throughput_and_latency/fmla_4s_instr.s
    :language: asm
    :lines: 31-71
    :caption: FMLA (vector) with arrangement specifier ``4S``

.. literalinclude:: ../../../src/submissions/03_neon/01_execution_throughput_and_latency/fmla_2s_instr.s
    :language: asm
    :lines: 31-71
    :caption: FMLA (vector) with arrangement specifier ``2S``

.. literalinclude:: ../../../src/submissions/03_neon/01_execution_throughput_and_latency/fmadd_instr.s
    :language: asm
    :lines: 39-82
    :caption: FMADD (scalar), single-precision variant

In order to measure the throughput of these instructions we developed a C++ microbenchmark.
For each instruction we firstly performed a warm up, measured the time, counted the operations and then calculated the GFLOPs.

.. literalinclude:: ../../../src/submissions/03_neon/01_execution_throughput_and_latency/microbench.cpp
    :language: cpp
    :lines: 61-73
    :caption: Example benchmark for FMLA (vector) with arrangement specifier ``4S``

For the ``2S`` and the FMADD (scalar) instructions, we simply adjusted the calculations for the operations slightly:

.. literalinclude:: ../../../src/submissions/03_neon/01_execution_throughput_and_latency/microbench.cpp
    :language: cpp
    :lines: 85-89
    :caption: Calculations for FMLA (vector) with arrangement specifier ``2S``

.. literalinclude:: ../../../src/submissions/03_neon/01_execution_throughput_and_latency/microbench.cpp
    :language: cpp
    :lines: 101-105
    :caption: Calculations for FMLA (vector) with arrangement specifier ``2S``

For this benchmarking task we obtained the following results:

.. literalinclude:: ../../../src/submissions/03_neon/01_execution_throughput_and_latency/benchmarking_results.txt
    :language: text
    :lines: 1-23
    :caption: Throughput results for the three instructions

It can be clearly seen that the FMLA (vector) with arrangement specifier ``4S`` instruction performs approximately 
2.5 times as many floating point operations than the FMLA (vector) with arrangement specifier ``2S`` instruction. 
Further the FMLA (vector) with arrangement specifier ``2S`` instructions performs at approximately 2.5 times more
floating point operations than the FMADD (scalar) instruction. 

This clearly shows that leveraging data-level parallelism (vector-based) can yield a much higher throughput, than
using only scalar operations.

Latency
^^^^^^^^

To measure the execution latency for FMLA (vector) instructions with arrangement specifier ``4S``, we examined two scenarios:

1. Each instruction depends on the destination register and one of the source register of the previous instruction

.. literalinclude:: ../../../src/submissions/03_neon/01_execution_throughput_and_latency/fmla_4s_source_lat_instr.s
    :language: asm
    :lines: 33-36
    :caption: fmla instructions with dependencies on the destination register and one of the source registers

2. Each instruction depends only on the destination register of the previous instruction

.. literalinclude:: ../../../src/submissions/03_neon/01_execution_throughput_and_latency/fmla_4s_dest_lat_instr.s
    :language: asm
    :lines: 33-36
    :caption: fmla instructions with dependencies on the destination register

Both files contain 32 fmla instructions each, which are executed 100 times. The results of our benchmark is shown below:

.. literalinclude:: ../../../src/submissions/03_neon/01_execution_throughput_and_latency/benchmarking_results.txt
    :language: text
    :lines: 25-39
    :caption: Latency results for the two scenarios

We can see that both scenarios have similar results, which is why we computed the latency only for the first scenario.

We measured :math:`1.16266 \times 10^{10}` instructions per second, which means that the latency of the FMLA (vector) instruction with arrangement specifier ``4S`` is approximately :math:`\frac{1}{1.16266 \times 10^{10}} \approx 8.6 \times 10^{-11}` seconds. Using a known clock frequency of 4.4 GHz, we computed the latency as :math:`8.6 \times 10^{-11} \times 4.4 \times 10^9 = 0.3784` cycles.

Microkernel
--------------------------------

For the second task we were implementing a microkernel to execute a matrix multiplication for matrices with the dimensions:

1. Matrix A: 16 x 1
2. Matrix B: 1 x 6
3. Matrix C: 16 x 6

Neon Microkernel
^^^^^^^^^^^^^^^^^

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

Testing and Benchmarking
^^^^^^^^^^^^^^^^^^^^^^^^^

To test and compare our versions with one another we:

1. implemented a microkernel that would give us a visual indication of the results
2. implemented a test using Catch2 to test the correctness of our implementations
3. implemented a microbenchmark that would calculate the GFLOPs for each of the three versions

The GFLOPs were calculated using the following formula:

.. literalinclude:: ../../../src/submissions/03_neon/02_microkernel/microbench.cpp
    :language: cpp
    :lines: 138-143
    :caption: GFLOPs calculations

For each version we would perform ``50,000`` iterations as a warmup to guarantee similar results for each execution of the benchmark.
Using this approach we obtained the following results:

.. literalinclude:: ../../../src/submissions/03_neon/02_microkernel/benchmarking_results.txt
    :language: text
    :caption: GFLOPs calculations

The GLFOPs results indicate that with every version we obtained slightly better results, resulting in 
about ``1.7`` GLOPs in difference comparing our best with our worst approach.

Loops
------

In this task, we had to add loops to the matrix multiplication kernel which we wrote in the previous task. The goal was to enable the 16x6x1 kernel to be used for larger matrices.

The first step was to implement a loop in the K dimension, resulting in a 16x6x64 kernel. The loading and storing of matrix C was left unchanged. The relevant code is shown below:

.. literalinclude:: ../../../src/submissions/03_neon/03_loops/matmul_16_6_64.s
    :language: asm
    :linenos:
    :lines: 67-133
    :caption: Looping matmul_16_6_1 over K dimension

The ``matmul_16_6_1`` kernel mostly stayed the same, except that for each K loop, we now need to adjust the pointers to the input matrices A and B. At the end of each loop, we move the pointers to A to the next column by adding the given stride. In B, we need to move the pointer to the next row. Therefore, we jump by 4 Bytes (since we are using 32-bit floats) from the starting address of B. To keep jumping to the next row in each loop, we accumulate the offset of 4 Bytes in the register ``x9``.

The second step was to implement a loop in the M dimension, resulting in a 64x6x64 kernel. To keep the code examples shorter, we exclude the K loop from the code snippets. The relevant code is shown below:

.. literalinclude:: ../../../src/submissions/03_neon/03_loops/matmul_64_6_64.s
    :language: asm
    :linenos:
    :lines: 45-92
    :caption: First part of looping over M dimension

.. literalinclude:: ../../../src/submissions/03_neon/03_loops/matmul_64_6_64.s
    :language: asm
    :linenos:
    :lines: 153-193
    :caption: Second part of looping over M dimension

The M loop needs only 4 iterations, since we are extending the kernel from 16 to 64 in the M dimension by dividing the M dimension into 4 blocks of 16 elements. At the end of the M loop, we move the pointers of A and C to the next block. We jump by 16 elements in the M dimension, which means adding 16*4 Bytes to the pointer of A and C.

The last step was to implement a loop in the N dimension, resulting in a 64x48x64 kernel. The relevant code is shown below:

.. literalinclude:: ../../../src/submissions/03_neon/03_loops/matmul_64_48_64.s
    :language: asm
    :linenos:
    :lines: 49-66
    :caption: First part of looping over N dimension

.. literalinclude:: ../../../src/submissions/03_neon/03_loops/matmul_64_48_64.s
    :language: asm
    :linenos:
    :lines: 205-220
    :caption: Second part of looping over N dimension

Since we are extending the kernel from 6 to 48 in the N dimension, we need to divide the N dimension into 8 blocks of 6 elements. This means that the loop will have 8 iterations. For each N loop, it is important to first reset the pointer of A to the original address. After each iteration, we need to move the pointers of B and C to the next block. To do this, we jump by elements in the N dimension, that is specifically 6 columns of B and C. We do this by adding 6 times the stride of B and C to the pointers.

Testing and Benchmarking
^^^^^^^^^^^^^^^^^^^^^^^^^

For all three kernels we have written unit tests. To execute the tests, one first needs to compile the code by invoking ``make`` from within the ``src/submissions/03_neon/03_loops`` directory. This will create an executable that can be run with ``./build/test``.

We also calculated the GFLOPs for each of these matrix multiplications.
To calculate them we followed the simple formula:

.. math:: M \cdot N \cdot K \cdot \text{Ops Per FMLA}

The results that we obtained were:

.. literalinclude:: ../../../src/submissions/03_neon/03_loops/benchmarking_results.txt
    :language: text
    :caption: GFLOPs calculations for MatMuls

Our results indicate that the number of GFLOPs is very consistent, even when scaling the size of our matrix.

