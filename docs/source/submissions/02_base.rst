Base
======

Copying Data
----------------

For this task, we have been given a C++ driver that makes calls to C and assembly functions. The first C function copies 7 32-bit integers from an array to another. The second C function does the same, but with a variable number of integers which is passed as the first argument. The two assembly functions are supposed to have the same functionality as the C functions, however they were not implemented yet.

For context, the C functions are as follows:

.. literalinclude:: ../../../src/submissions/02_base/01_copying_data/copy_c.c
    :language: c
    :linenos:
    :caption: copy_c.c

Task 1 & 2
^^^^^^^^^^^^

Given the C functions, our task is to implement the assembly functions that match the C functions in functionality by using only base instructions. For the first function, we simply load and store the 7 32-bit integers from the source to the destination using the `ldr` and `str` instructions. We use the given addresses and an immediate offset that is incremented by 4 for each 32-bit integer (4 bytes = 32 bit).

For the second function, we need to use a loop to copy the values from the source to the destination. We use two registers to keep track of number of elements copied and the current byte offset. The loop then starts by copying the first 32-bit integer, and then increase the number of elements copied by 1 and the byte offset by 4 (since each integer is 4 bytes). Next, we use the `cmp` instruction to check if we have copied the specified number of integers. If not, we go back to the beginning of the loop and repeat the process. If we have copied the specified number of integers, we exit the loop and return from the function.

.. literalinclude:: ../../../src/submissions/02_base/01_copying_data/copy_asm.s
    :language: asm
    :linenos:
    :caption: copy_asm.s

After compiling and running the driver, we can see that the assembly functions work as expected. The output of the driver is as follows:

.. code-block:: bash

    $ ./copy_driver
    copy_c_0: copy succeeded
    copy_c_1: copy succeeded
    copy_asm_0: copy succeeded
    copy_asm_1: copy succeeded


Instruction Throughput and Latency
--------------------------------------

For this task, we were supposed to write a micro-benchmarking script to measure the
throughput and the latency of the ADD (shifted register) and the MUL instruction.

Throughput
^^^^^^^^^^

To measure the **throughput** of the instructions we have developed an assembly function
for each of the two instructions. The idea for measuring the throughput is that for 
every consecutive add instructions there shouldn't be any dependencies on the add 
instruction from before. The loop that was executed several times for the ADD (shifted register)
and the MUL instruction was:

.. literalinclude:: ../../../src/submissions/02_base/02_instruction_throughput_and_latency/add_instr.s
    :language: asm
    :linenos:
    :lines: 28-60
    :caption: loop in add_instr.s

.. literalinclude:: ../../../src/submissions/02_base/02_instruction_throughput_and_latency/mul_instr.s
    :language: asm
    :linenos:
    :lines: 28-60
    :caption: loop in mul_instr.s

Latency
^^^^^^^

To measure the **latency** of the two instructions we have developed two more assembly
functions. The idea for measuring the latency is to have consecutive instructions that 
dependent on the result from the last executed instruction. That meant we had to slightly 
adjust the loops from the throughput programs, resulting in the following loops for the
ADD (shifted register) and the MUL instruction: 

.. literalinclude:: ../../../src/submissions/02_base/02_instruction_throughput_and_latency/add_lat_instr.s
    :language: asm
    :linenos:
    :lines: 28-39
    :caption: loop in add_lat_instr.s

.. literalinclude:: ../../../src/submissions/02_base/02_instruction_throughput_and_latency/mul_lat_instr.s
    :language: asm
    :linenos:
    :lines: 27-38
    :caption: loop in mul_lat_instr.s

Results
^^^^^^^

To test our assembly functions we implemented a microbenchmark in C++ that would 
1. call these functions several times
2. measure the time it took to complete these computations
3. calculate the GOPS obtained by these calculations

.. literalinclude:: ../../../src/submissions/02_base/02_instruction_throughput_and_latency/microbench.cpp
    :language: cpp
    :lines: 33-36
    :caption: time measurement for add_instr in microbench.cpp

.. literalinclude:: ../../../src/submissions/02_base/02_instruction_throughput_and_latency/microbench.cpp
    :language: cpp
    :lines: 46-48
    :caption: GOPS calculation for add_instr in microbench.cpp

To compile and execute the microbenchmark we simply used:

.. code-block:: bash

    $ g++ microbench.cpp add_instr.s add_lat_instr.s mul_instr.s mul_lat_instr.s -o microbench.o
    $ ./microbench.o

When executing our benchmarking script we obtained the following results:

.. literalinclude:: ../../../src/submissions/02_base/02_instruction_throughput_and_latency/results.txt
    :language: none
    :caption: results

To understand the results, we did two things:
1. we searched for the `clock speed <https://everymac.com/systems/apple/mac_mini/specs/mac-mini-m4-10-core-cpu-10-core-gpu-2024-specs.html#:~:text=This%20model%20is%20powered%20by,speed%20clockspeed%20around%204.4%20GHz.>`_  which is about 4.4 GHz
2. we looked at the `ARM Neoverse V2 Software Optimization Guide <https://developer.arm.com/documentation/109898/latest/>`_ for the base instructions.

The guide specifies the following theoretical instruction **throughput** per cycle:

* ADD: 6 instructions per cycle
* MUL: 2 instructions per cycle

and the latency (until a result is available) of:

* ADD: 1 clock cycle
* MUL: 2 clock cycles

Looking at a these numbers, we can assume a clock cycle speed of:

.. math:: \frac{29.0414 \text{ GOPS}}{6} = 4.84 \text{ GHz}

.. math:: \frac{13.2584 \text{ GOPS}}{2} = 6.63 \text{ GHz}

For the ADD instruction, this aligns closely with the *known* clock speed of 4.4 GHz.
For the MUL instruction on the other hand the clock speed it higher than expected.

For the **latency** we can make a similar calculation:

.. math:: \frac{4.4 \text{ GHz}}{4.37951 \text{ GOPS}} ≈ 1 \text{ clock cycles per instr}

.. math:: \frac{4.4 \text{ GHz}}{1.46244 \text{ GOPS}} ≈ 3 \text{ clock cycles per instr}

This shows that our obtained results for the ADD instruction correspond with the 
number that we obtained from the guide, whereas the latency for the MUL instruction 
is slightly higher than expected.
