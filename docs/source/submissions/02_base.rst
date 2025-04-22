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

Given the C functions, our task is to implement the assembly functions that match the C functions in functionality by using only base instructions. For the first function, we simply load and store the 7 32-bit integers from the source to the destination using the `ldr` and `str` instructions. We start with the given addresses and increment them by 4 bytes for each further 32-bit integer.

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