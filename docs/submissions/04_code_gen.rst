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

At first, we had to decide on how to block the matrices. In the M dimension, we decided to use a block size of 8 and in the N dimension we decided to use a block size of 4. The larger we keep the block size, the more efficiently we can use loads, stores and FMLA instructions. However, the issue with large block sizes is that we need to write a lot of specialized kernels for all M and N dimensions smaller or equal to the block size. If the input parameters are not multiples of the block size, we need to write additional code to handle the remaining elements. 

For a block size of M = 8, we already wrote such a kernel in pure assembly, see :ref:`generic-kernel`. Using this generic kernel as a starting point, we reduced the N dimension from 6 to 4. Our reasoning was that we wanted to reduce the number of specialized kernels we need to write. Additionally, we assumed that most numbers commonly used in practice are multiples of 4 instead of 6, thus not depending on the specialized kernels. With this change, we implemented the ``matmul_m_4_k`` kernel, which can compute C+=AB for any matrices where M and K can be chosen freely, but N is fixed to 4.

The kernel first computes the number of blocks in the M dimension and the remaining elements. 

.. literalinclude:: ../../src/kernels/matmul_m_4_k.cpp
    :language: cpp
    :lines: 22-24
    :caption: Computing the number of blocks in the M dimension

Using these numbers, we can call the specialized kernels:

.. literalinclude:: ../../src/kernels/matmul_m_4_k.cpp
    :language: cpp
    :lines: 53-93
    :caption: Calling specialized kernels for different M dimensions

But what does such a specialized kernel look like? For the most part, they are similar to the microkernels we implemented before. The only difference is that we need to adjust the loads, stores and FMLA instructions for fixed M dimensions. For example in the case of M = 3:

.. literalinclude:: ../../src/kernels/matmul_m_4_k.cpp
    :language: cpp
    :lines: 354-357
    :caption: Loading a column of C with M = 3

While we can simply load a double word when M = 2 or even a quad word when M = 4, we need to divide our loads into two parts when M = 3. First, we load a double word and then the remaining single word. The same applies to the stores:

.. literalinclude:: ../../src/kernels/matmul_m_4_k.cpp
    :language: cpp
    :lines: 416-419
    :caption: Storing a column of C with M = 3

The FMLA instructions are also adjusted to the M dimension. For example, when M = 3, we need to use two FMLA instructions to compute the result:

.. literalinclude:: ../../src/kernels/matmul_m_4_k.cpp
    :language: cpp
    :lines: 381-384
    :caption: FMLA instructions

While one could use an ``fmla`` instruction and zero padding, we decided to use one ``fmla`` instruction for the first two elements and one ``fmadd`` instruction for the last element. We did not evaluate any performance differences between the two approaches, but chose the second one because to us it seems more readable and easier to understand. The other specialized kernels for M = 1, 2, 4, 5, 6 and 7 are implemented in a similar way.

Having implemented the ``matmul_m_4_k`` kernel, we can now turn our attention towards the ``matmul_m_n_k`` kernel. Since we decided to block N by 4, we can use the same approach as before. We first compute the number of blocks in the N dimension and the remaining elements.

.. literalinclude:: ../../src/kernels/matmul_m_n_k.cpp
    :language: cpp
    :lines: 27-29
    :caption: Computing the number of blocks in the N dimension

``nLoopRemainder`` can take any value between 0 and 3, which means that additionally to the ``matmul_m_4_k`` kernel where ``nLoopRemainder`` is 0, we need to implement specialized kernels for ``nLoopRemainder`` = 1, 2 and 3. The specialized kernels are basically the same as the ``matmul_m_4_k`` kernel, but we simply removed some of the loads, stores and FMLA instructions. For the more curious reader, the specialized kernels can be found in the files ``src/kernels/matmul_m_1_k.cpp``, ``src/kernels/matmul_m_2_k.cpp`` and ``src/kernels/matmul_m_3_k.cpp``.

For the whole N loop, we use switch statements to call the specialized kernels. The final implementation looks like this:

.. literalinclude:: ../../src/kernels/matmul_m_n_k.cpp
    :language: cpp
    :lines: 62-151
    :caption: N loop

The full code is available in the file ``src/kernels/matmul_m_n_k.cpp``.

4.2.2 Verification of the GEMM kernel using a reference implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This task requires us to verify the correctness of our ``matmul_m_n_k`` kernel by comparing to a reference implementation for 1≤M≤64, 1≤N≤64, K∈[1,16,32,64,128], and lda=M, ldb=K, ldc=M.
We realized this verification using a ``Catch2`` unit test:

.. literalinclude:: ../../src/kernels/matmul_m_n_k.test.cpp
    :language: cpp
    :lines: 8-55
    :caption: Unit test for the ``matmul_m_n_k`` kernel

The M and N dimensions are generated randomly, while the K dimension is fixed to multiple given values. We compute the expected result using high level C++ code and compare it to the result of our kernel.
