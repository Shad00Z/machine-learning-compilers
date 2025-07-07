.. _Assembly:

1. Assembly
===========

To begin the machine learning compilers project, we explored the fundamentals of the assembly language and compiler behavior to establish a fundamental understanding. 

1.1 Hello Assembly
-------------------

We started with a simple C program:

.. code-block:: cpp
    :caption: `hello_assembly.c <https://github.com/Shad00Z/machine-learning-compilers/blob/main/src/submissions/01_assembly/data/hello_assembly.c>`_

    #include <stdio.h>

    void hello_assembly() 
    {
        printf( "Hello Assembly Language!\n");
    }

Our goal was to compile this program using both ``GCC`` and ``Clang`` and analyze the differences in the generated assembly code to understand how compiler implementations can vary at the machine level.

Task 1.1.1
^^^^^^^^^^^^

To generate assembly code from the :code:`hello_assembly.c` program using both compilers, we first identified the appropriate commands:

.. code-block:: bash
    :caption: ``GCC`` compiler

    gcc -S hello_assembly.c

.. code-block:: bash
    :caption: ``Clang`` compiler

    clang -S hello_assembly.c


Task 1.1.2
^^^^^^^^^^^^

After compiling the C program, we analyzed the generated assembly code by:

1. Locating the "Hello Assembly Language!" string.
2. Identifying the instructions inserted by the compiler insert to conform to the procedure call standard (PCS).
3. Identifying the function call to the C standard library (libc) that prints the string.

Task 1.1.2.1 GCC
"""""""""""""""""

The ``GCC``-generated file:

.. literalinclude:: ../../src/submissions/01_assembly/01_task/gcc_hello_assembly.s 
    :language: asm
    :linenos:
    :caption: `gcc_hello_assembly.s <https://github.com/Shad00Z/machine-learning-compilers/blob/main/src/submissions/01_assembly/01_task/gcc_hello_assembly.s>`_

1. The string "Hello Assembly Language!" appears at **line 7**.
2. The instructions for the procedure call standard are:

.. code-block:: asm

    stp	x29, x30, [sp, -16]!
    mov	x29, sp

    ... 

    ldp x29, x30, [sp], 16
    ret

3. The function call used to print the string is:

.. code-block:: asm

    bl	puts

Task 1.1.2.2 Clang
"""""""""""""""""""

The ``Clang``-generated file:

.. literalinclude:: ../../src/submissions/01_assembly/01_task/clang_hello_assembly.s 
    :language: asm
    :linenos:
    :caption: `clang_hello_assembly.s <https://github.com/Shad00Z/machine-learning-compilers/blob/main/src/submissions/01_assembly/01_task/clang_hello_assembly.s>`_

1. The string "Hello Assembly Language!" appears at **line 31**.
2. The instructions for the procedure call standard are:

.. code-block:: asm

    stp	x29, x30, [sp, -16]!
    mov	x29, sp

    ...

    ldp x29, x30, [sp], 16
    ret

3. The function call used to print the string is:

.. code-block:: asm

    bl	printf

This analysis illustrates that while both compilers conform to the same calling standard, they differ in aspects, such as the string placement and the choice of standard library function used for the output. 

Task 1.1.3
^^^^^^^^^^^^

After analyzing the generated assembly code, we implemented a simple C++ driver that calls the ``hello_assembly`` function:

.. literalinclude:: ../../src/submissions/01_assembly/01_task/driver_hello_assembly.cpp 
    :language: cpp
    :linenos:
    :caption: `driver_hello_assembly.cpp <https://github.com/Shad00Z/machine-learning-compilers/blob/main/src/submissions/01_assembly/01_task/driver_hello_assembly.cpp>`_

To compile the driver along with the ``GCC``-generated assembly code and execute the program, we used the following commands:

.. code-block:: bash

    g++ driver_hello_assembly.cpp gcc_hello_assembly.s -o hello_assembly
    ./hello_assembly

The output of the program:

.. code-block:: bash

    Calling hello_assembly ...
    Hello Assembly Language!
    ... returned from function call!


1.2 Assembly Function
----------------------

Next, we worked with the :code:`add_values.s` file.

.. code-block:: asm
    :caption: `add_values.s <https://github.com/Shad00Z/machine-learning-compilers/blob/main/src/submissions/01_assembly/data/add_values.s>`_

        .text
        .type add_values, %function
        .global add_values
    add_values:
        stp fp, lr, [sp, #-16]!
        mov fp, sp

        ldr w3, [x0]
        ldr w4, [x1]
        add w5, w3, w4
        str w5, [x2]

        ldp fp, lr, [sp], #16

        ret


Task 1.2.1
^^^^^^^^^^^

As a first step, we assembled the file into an object file using:

.. code-block:: bash

    as add_values.s -o add_values.o

Task 1.2.2
^^^^^^^^^^^

With the :code:`add_values.o`, we performed different file generations to understand its structure:

.. code-block:: bash
    :caption: Generating a hexadecimal dump

    hexdump add_values.o > hexdump_add_values.hex

.. literalinclude:: ../../src/submissions/01_assembly/02_task/hexdump_add_values.hex
    :language: none
    :linenos:
    :caption: `hexdump_add_values.hex <https://github.com/Shad00Z/machine-learning-compilers/blob/main/src/submissions/01_assembly/02_task/hexdump_add_values.hex>`_

.. code-block:: bash
    :caption: Generating section headers

    readelf -S add_values.o > sec_headers_add_values.relf

.. literalinclude:: ../../src/submissions/01_assembly/02_task/sec_headers_add_values.relf
    :language: none
    :linenos:
    :caption: `sec_headers_add_values.relf <https://github.com/Shad00Z/machine-learning-compilers/blob/main/src/submissions/01_assembly/02_task/sec_headers_add_values.relf>`_

.. code-block:: bash
    :caption: Generating disassembled file

    objdump --syms -S -d add_values.o > dis_add_values.dis

.. literalinclude:: ../../src/submissions/01_assembly/02_task/dis_add_values.dis
    :language: none
    :linenos:
    :caption: `dis_add_values.dis <https://github.com/Shad00Z/machine-learning-compilers/blob/main/src/submissions/01_assembly/02_task/dis_add_values.dis>`_

Task 1.2.3
^^^^^^^^^^^

The next step was to determine the **size** of the :code:`.text` section and understand its content. This information is available in the section headers file in *line 9*:

.. literalinclude:: ../../src/submissions/01_assembly/02_task/sec_headers_add_values.relf
    :language: none
    :lines: 8-9
    :caption: Lines 8 and 9

The ``.text`` section has a size of :code:`0000000000000020` or :code:`0x20` bytes, which corresponds to 32 bytes in decimal. Since each AArch64 instruction is 4 bytes, the function ``add_values`` consists of exactly 8 instructions.

We confirmed this observation by inspecting the disassembled output:

.. literalinclude:: ../../src/submissions/01_assembly/02_task/dis_add_values.dis
    :language: none
    :lines: 15-22
    :caption: Lines 8 and 9

The instructions start at :code:`0x00` and proceed in 4-byte increments, ending at ``0x1c`` with the ``ret`` instruction. This confirms that our there are exactly 8 instructions present, that match the ``.text`` section size. 

Task 1.2.4
^^^^^^^^^^^

Similar to the first task, we now tested the functionality of the :code:`add_values` function by implementing a C++ driver:

.. literalinclude:: ../../src/submissions/01_assembly/02_task/driver_add_values.cpp
    :language: cpp
    :linenos:
    :caption: `driver_add_values.cpp <https://github.com/Shad00Z/machine-learning-compilers/blob/main/src/submissions/01_assembly/02_task/driver_add_values.cpp>`_

We compiled and linked the driver with the assembly file using:

.. code-block:: bash

    g++ driver_add_values.cpp add_values.s -o driver_add_values

After executing these commands, we received the following results:

.. code-block:: bash

    Calling assembly 'add_value' function ...
    l_data_1 / l_value_2 / l_value_o
    4 / 7 / 11

This confirms that the ``add_values`` correctly adds the two input values together and stores the result at the specified memory location.

Task 1.2.5
^^^^^^^^^^^

To better understand the contents of the general purpose registers during a function call to ``add_values``, we used the GNU Project Debugger (GDB) to step through the function execution. 

We launched GDB with the compiled executable:

.. code-block:: bash

    gdb ./driver_add_values
    lay next

To activate the correct layout view that displays both the assembly instructions and register contents, we ran :code:`lay next`. After pressing ``Enter``, GDB displayed the desired view.
Next, we set a breakpoint at the ``add_values`` function and began the debugging process:

.. code-block:: bash
    
    break add_values
    run

After starting the debugging process, we used ``nexti`` to step through each instruction one at a time:

.. raw:: html

    <embed src="../_static/GDB_add_values.pdf" width="100%" height="600px" type="application/pdf">

