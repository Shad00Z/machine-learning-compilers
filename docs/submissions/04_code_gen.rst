4. Code Generation
====================

4.1 Microkernel
---------------------

This section sets the foundation for our machine learning compiler.
We are starting off by implementing / generating batch-reduce matrix-matrix multiplications (BRGEMMS).

The first step was to implement a ``generate`` function, which is supposed to be the entry for all BRGEMM code generation:

.. literalinclude:: ../../src/Brgemm.cpp
    :language: cpp
    :lines: 7-74
    :caption: implementation of the ``generate`` function

In this function we are generating the code for the BRGEMM kernels.

Firstly, we needed the instructions which a BRGEMM kernel consists of.
Therefore we started wrapping the assembly code in C++ functions.

.. literalinclude:: ../_static/InstGen.cpp
    :language: cpp
    :lines: 140-173
    :caption: Load instruction for a single general purpose register 

After implementing all necessary instructions, we started implementing our first kernel.
The first kernel that we implemented was a simple matrix multiplication kernel, in the form of a ``matmul_16_6_1``.

.. literalinclude:: ../../src/kernels/matmul_16_6_1.cpp
    :language: cpp
    :lines: 71-131
    :caption: FMLA instructions for the ``matmul_16_6_1`` kernel

After implementing this first kernel, we started implementing a more general version with a ``matmul_16_6_k`` kernel.
For this kernel we needed a loop to iterate over the ``k`` dimension.

.. literalinclude:: ../../src/kernels/matmul_16_6_k.cpp
    :language: cpp
    :lines: 133-146
    :caption: Loop instruction using code generation

As a last step we measured the performance of our generated code, resulting in the following results:

.. literalinclude:: ../../src/benchmark/benchmarking_results_matmul_16_6.txt
    :language: text
    :caption: GFLOPs results of the ``matmul_16_6_1`` and ``matmul_16_6_k`` kernels

Comparing our ``matmul_16_6_1`` kernel to our previous implementations, we are slightly worse, loosing about ``8 GFLOPs``.
However, for the ``matmul_16_6_k`` we reach the same number of GFLOPs that we reached with our best implementations.

4.2 GEMM
-----------------

4.2.1 Implementation of a GEMM kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section extends the previously implemented kernels to a more general GEMM kernel. It should be able to compute C+=AB for arbitrary A, B and C matrices.

At first, we had to decide on how to block the matrices. In the M dimension, we decided to use a block size of 8 and in the N dimension we decided to use a block size of 4. The larger we keep the block size, the more efficiently we can use loads, stores and FMLA instructions. The issue with large block sizes is however that we need to write a lot of specialized kernels for all M and N dimensions smaller or equal to the block size. If the input parameters are not multiples of the block size, we need to write additional code to handle the remaining elements. For a block size of M = 8, we already wrote such a kernel in pure assembly, see :ref:`generic-kernel`.