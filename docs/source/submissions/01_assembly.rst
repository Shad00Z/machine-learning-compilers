Assembly
===========

Hello Assembly
--------------

We have been given two files:

.. literalinclude:: ../../../src/submissions/01_assembly/data/hello_assembly.c
    :language: c
    :linenos:
    :caption: hello_assembly.c

.. literalinclude:: ../../../src/submissions/01_assembly/data/add_values.s 
    :language: asm
    :linenos:
    :caption: add_values.s

Task 1
^^^^^^

First we would like to compile :code:`hello_assembly.c` using the GCC and Clang compiler.

The **GCC** compiler can be used with the following statement.

.. code-block:: bash

    gcc -S hello_assembly.c

The **Clang** compiler can be used with the following statement.

.. code-block:: bash

    clang -S hello_assembly.c


Task 2
^^^^^^

After compilation we would like for each generated assembly file to:

1. Find the "Hello Assembly Language!" string
2. Identify the instructions that the compilers insert to conform to the procedure call standard
3. Identify the function call to libc that prints the string

GCC
"""

The **GCC** :code:`gcc_hello_assembly.s` looks like:

.. literalinclude:: ../../../src/submissions/01_assembly/01_task/gcc_hello_assembly.s 
    :language: asm
    :linenos:
    :caption: GCC compiled file

1. The string "Hello Assembly Language!" can be found in *line 7*.
2. The instructions for the procedure call standard:

.. code-block:: asm

    stp	x29, x30, [sp, -16]!
    mov	x29, sp
    ldp x29, x30, [sp], 16
    ret

3. The function call is that prints the string:

.. code-block:: asm

    bl	puts

Clang
"""""

The **Clang** :code:`clang_hello_assembly.s` looks like:

.. literalinclude:: ../../../src/submissions/01_assembly/01_task/clang_hello_assembly.s 
    :language: asm
    :linenos:
    :caption: Clang compiled file

1. The string "Hello Assembly Language!" can be found in *line 31*.
2. The instructions for the procedure call standard:

.. code-block:: asm

    stp	x29, x30, [sp, -16]!
    mov	x29, sp
    ldp x29, x30, [sp], 16
    ret

3. The function call is that prints the string:

.. code-block:: asm

    bl	printf

Task 3
^^^^^^

Now we would like to test the :code:`hello_assembly` function with a C++ driver. 

.. literalinclude:: ../../../src/submissions/01_assembly/01_task/driver_hello_assembly.cpp 
    :language: cpp
    :linenos:
    :caption: C++ driver

To compile and run this code, we can use:

.. code-block:: bash

    g++ driver_hello_assembly.cpp gcc_hello_assembly.s -o hello_assembly
    ./hello_assembly

The output is then:

.. code-block:: bash

    Calling hello_assembly ...
    Hello Assembly Language!
    ... returned from function call!


Assembly Function
-----------------

Now we want to work with the :code:`add_values.s` file.

Task 1
^^^^^^

As a first step we want to assemble the file. We can do that by calling:

.. code-block:: bash

    as add_values.s -o add_values.o

Task 2
^^^^^^

With :code:`add_values.o` we are now creating different files:

To get a hexadecimal dump we used:

.. code-block:: bash

    hexdump add_values.o > hexdump_add_values.hex

.. literalinclude:: ../../../src/submissions/01_assembly/02_task/hexdump_add_values.hex
    :language: none
    :linenos:
    :caption: Hexadecimal dump

To get the section headers we used:

.. code-block:: bash

    readelf -S add_values.o > sec_headers_add_values.relf

.. literalinclude:: ../../../src/submissions/01_assembly/02_task/sec_headers_add_values.relf
    :language: none
    :linenos:
    :caption: Section headers

To get the disassembled file we used:

.. code-block:: bash

    objdump --syms -S -d add_values.o > dis_add_values.dis

.. literalinclude:: ../../../src/submissions/01_assembly/02_task/dis_add_values.dis
    :language: none
    :linenos:
    :caption: Disassembled file

Task 3
^^^^^^

The *size* of the :code:`.text` section can be found in the section headers in *line 9*.

.. literalinclude:: ../../../src/submissions/01_assembly/02_task/sec_headers_add_values.relf
    :language: none
    :lines: 8-9
    :caption: Lines 8 and 9

The *size* of the text section is :code:`0000000000000020` or :code:`0x20` which translates to 32 bytes.
These 32 bytes correspond to 8 assembly instructions (each 4 bytes) that the function :code:`add_values` performs.

.. literalinclude:: ../../../src/submissions/01_assembly/02_task/dis_add_values.dis
    :language: none
    :lines: 15-22
    :caption: Lines 8 and 9

Starting at :code:`0x00`, incrementing always by 4 bytes per instruction (performing 8 of those increments), 
we ultimately finish at the :code:`ret` instruction located at :code:`0xc1`, which marks the end of the :code:`.text` section.
Therefore, from :code:`0x00` to :code:`0xc1` there are exactly 8 instructions covered, which equal a *size* of 32 bytes.

Task 4
^^^^^^

To test the functionality of the :code:`add_values` function we implemented a C++ driver:

.. literalinclude:: ../../../src/submissions/01_assembly/02_task/driver_add_values.cpp
    :language: cpp
    :linenos:
    :caption: driver_add_values.cpp

To test our function we now could call:

.. code-block:: bash

    g++ driver_add_values.cpp add_values.s -o driver_add_values

Where :code:`driver_add_values` results in:

.. code-block:: bash

    Calling assembly 'add_value' function ...
    l_data_1 / l_value_2 / l_value_o
    4 / 7 / 11

Task 5
^^^^^^

To get an idea how the contents of the general purpose registers look like, when calling the :code:`add_values` function
we stepped through a function call using the GNU Project Debugger.

To get the correct view in the GNU Project Debugger we call:

.. code-block:: bash

    gdb ./driver_add_values
    lay next

After :code:`lay next` we pressed enter 3 times to get to the register view in combination with the assembly instructions.
To start the debugging process we call:

.. code-block:: bash
    
    break add_values
    run

After starting the debugging process we then proceeded to skip from instruction to instruction using :code:`nexti`.

.. raw:: html

    <embed src="../_static/GDB_add_values.pdf" width="100%" height="600px" type="application/pdf">

