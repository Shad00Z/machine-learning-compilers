.. _Base:

2. Base
=========

After exploring the fundamentals of assembly, we moved on to applying that knowledge by implementing small programs using only AArch64 base instructions.

2.1 Copying Data
------------------

In the first task, we were provided with a `C++ driver <https://github.com/Shad00Z/machine-learning-compilers/blob/main/src/submissions/02_base/01_copying_data/copy_driver.cpp>`_ that calls both C and assembly functions. 

The goal was to replicate the behavior of the two C functions using assembly.
The first C function copies seven 32-bit integers from one array to another. 
The second C function performs the same operation for a variable number of integers, with the number of elements passed as the first argument. 
The corresponding assembly functions were pre-defined but not yet implemented.

For context, the two C functions are as follows:

.. literalinclude:: ../../src/submissions/02_base/01_copying_data/copy_c.c
    :language: c
    :linenos:
    :caption: `copy_c.c <https://github.com/Shad00Z/machine-learning-compilers/blob/main/src/submissions/02_base/01_copying_data/copy_c.c>`_

Task 2.1.1 & 2.1.2
^^^^^^^^^^^^^^^^^^^

For the first function, which copies seven 32-bit integers, we used ``ldr`` and ``str`` instructions to load from the source and store to the destination registers.
The memory addresses were accessed using immediate offsets, incremented by 4 bytes for each successive element (since each 32-bit integer occupies 4 bytes).

For the second function, which supports a variable number of elements, we implemented a loop. 
We used two registers to accomplish a loop:

* one register to track the number of copied elements

* one register to maintain the current byte offset

Our loop performs the following steps:
1. Load a 32-bit value from the source register using the current offset.
2. Store the value at the corresponding destination offset.
3. Increment both the element counter and the byte offset by 1 and 4, respectively.
4. Use the ``cmp`` instruction to check whether the target count has been reached.
5. If not, branch back to the top of the loop. Otherwise, return from the function.

The implementation is as follows:

.. literalinclude:: ../../src/submissions/02_base/01_copying_data/copy_asm.s
    :language: asm
    :linenos:
    :caption: `copy_asm.s <https://github.com/Shad00Z/machine-learning-compilers/blob/main/src/submissions/02_base/01_copying_data/copy_asm.s>`_

After compiling and running the driver, the output confirms that both assembly implementations behave identically to their C function counterparts:

.. code-block:: bash

    $ ./copy_driver
    copy_c_0: copy succeeded
    copy_c_1: copy succeeded
    copy_asm_0: copy succeeded
    copy_asm_1: copy succeeded


2.2 Instruction Throughput and Latency
----------------------------------------

In this task, we wrote a micro-benchmarking script to measure the throughput and the latency of two instructions:

* ``ADD`` (shifted register)

*  ``MUL``.

2.2.1 Throughput
^^^^^^^^^^^^^^^^^

To measure the **throughput** of the instructions, we developed an assembly function for each instruction. The idea was to construct a loop, where each instruction is independent of the previous one, thereby allowing the processor to execute them in parallel and also to avoid any dependencies between the instructions. 

.. literalinclude:: ../../src/submissions/02_base/02_instruction_throughput_and_latency/add_instr.s
    :language: asm
    :linenos:
    :lines: 28-60
    :caption: `add_instr.s <https://github.com/Shad00Z/machine-learning-compilers/blob/main/src/submissions/02_base/02_instruction_throughput_and_latency/add_instr.s>`_ loop

.. literalinclude:: ../../src/submissions/02_base/02_instruction_throughput_and_latency/mul_instr.s
    :language: asm
    :linenos:
    :lines: 28-60
    :caption: `mul_instr.s <https://github.com/Shad00Z/machine-learning-compilers/blob/main/src/submissions/02_base/02_instruction_throughput_and_latency/mul_instr.s>`_ loop

2.2.2 Latency
^^^^^^^^^^^^^^

To measure the **latency** of the two instructions, we implemented two additional assembly functions. In contrast to the throughput benchmarks, where instructions were independent, the idea here was to create data dependencies between instructions.

Each instruction in the loop depends on the result of the previous one, forcing the processor to wait for the output of one instruction before executing the next:

.. literalinclude:: ../../src/submissions/02_base/02_instruction_throughput_and_latency/add_lat_instr.s
    :language: asm
    :linenos:
    :lines: 28-39
    :caption: `add_lat_instr.s <https://github.com/Shad00Z/machine-learning-compilers/blob/main/src/submissions/02_base/02_instruction_throughput_and_latency/add_lat_instr.s>`_ loop

.. literalinclude:: ../../src/submissions/02_base/02_instruction_throughput_and_latency/mul_lat_instr.s
    :language: asm
    :linenos:
    :lines: 27-38
    :caption: `mul_lat_instr.s <https://github.com/Shad00Z/machine-learning-compilers/blob/main/src/submissions/02_base/02_instruction_throughput_and_latency/mul_lat_instr.s>`_ loop

2.2.3 Results
^^^^^^^^^^^^^^^

To test our assembly functions, we implemented a benchmark in C++ that:

1. calls each function multiple times,
2. measures the total execution time,
3. calculates the GOPS (Giga Operations Per Second) obtained by these calculations.

.. literalinclude:: ../../src/submissions/02_base/02_instruction_throughput_and_latency/microbench.cpp
    :dedent:
    :language: cpp
    :lines: 33-36
    :caption: time measurement for add_instr in `microbench.cpp <https://github.com/Shad00Z/machine-learning-compilers/blob/main/src/submissions/02_base/02_instruction_throughput_and_latency/microbench.cpp>`_

.. literalinclude:: ../../src/submissions/02_base/02_instruction_throughput_and_latency/microbench.cpp
    :dedent:
    :language: cpp
    :lines: 46-48
    :caption: GOPS calculation for add_instr

To compile and execute the benchmark, we ran:

.. code-block:: bash

    $ g++ microbench.cpp add_instr.s add_lat_instr.s mul_instr.s mul_lat_instr.s -o microbench.o
    $ ./microbench.o

We obtained the following results:

.. literalinclude:: ../../src/submissions/02_base/02_instruction_throughput_and_latency/results.txt
    :language: none
    :caption: Benchmarking results (GOPS)

To interpret these results, we first looked up the `clock speed of the Apple M4 chip <https://everymac.com/systems/apple/mac_mini/specs/mac-mini-m4-10-core-cpu-10-core-gpu-2024-specs.html#:~:text=This%20model%20is%20powered%20by,speed%20clockspeed%20around%204.4%20GHz.>`_, which is about 4.4 GHz

Further, we needed information about the M4 architecture. We could find, that the Apple M4 has 8 ALUs, from which 3 are able to perform ``MUL`` instructions.

Looking at a these numbers, we can assume a clock cycle speed of:

.. math:: \frac{29.0414 \text{ GOPS}}{8} = 3.63 \text{ GHz}

.. math:: \frac{13.2584 \text{ GOPS}}{3} = 4.42 \text{ GHz}

For the ``ADD`` instruction, we are slightly below the specified clock cycle speed of 4.4 GHz.
Looking at the ``MUL`` instruction on the other hand, our results closely align with the given clock speed.

For the **latency** we can make a similar calculation:

.. math:: \frac{4.4 \text{ GHz}}{4.37951 \text{ GOPS}} ≈ 1 \text{ clock cycles per instr}

.. math:: \frac{4.4 \text{ GHz}}{1.46244 \text{ GOPS}} ≈ 3 \text{ clock cycles per instr}

The ``ADD`` latency matches the theoretical value. The ``MUL`` is slightly higher than expected (3 vs. 2 cylces).
