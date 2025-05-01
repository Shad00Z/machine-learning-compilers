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
