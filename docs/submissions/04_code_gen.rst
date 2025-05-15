4. Code Generation
====================

4.1 Microkernel
---------------------

This section sets the foundation for our machine learning compiler.
We are starting off by implementing / generating batch-reduce matrix-matrix multiplications (BRGEMMS).

The first step was to implement a ``generate`` function, which is supposed to be the entry for all BRGEMM code generation:

.. literalinclude:: ../../src/Brgemm.cpp
    :language: cpp
    :lines: 7-91
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

.. literalinclude:: ../../src/kernels/matmul/subkernels/matmul_16_6_1.cpp
    :language: cpp
    :lines: 71-131
    :caption: FMLA instructions for the ``matmul_16_6_1`` kernel

After implementing this first kernel, we started implementing a more general version with a ``matmul_16_6_k`` kernel.
For this kernel we needed a loop to iterate over the ``k`` dimension.

.. literalinclude:: ../../src/kernels/matmul/subkernels/matmul_16_6_k.cpp
    :language: cpp
    :lines: 133-146
    :caption: Loop instruction using code generation

As a last step we measured the performance of our generated code, resulting in the following results:

.. literalinclude:: ../../src/benchmark/benchmarking_results_matmul_16_6.txt
    :language: text
    :caption: GFLOPs results of the ``matmul_16_6_1`` and ``matmul_16_6_k`` kernels

Comparing our ``matmul_16_6_1`` kernel to our previous implementations, we are slightly worse, loosing about ``8 GFLOPs``.
However, for the ``matmul_16_6_k`` we reach the same number of GFLOPs that we reached with our best implementations.

.. _4.2 GEMM:

4.2 GEMM
-----------------

4.2.1 Implementation of a GEMM kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section extends the previously implemented kernels to a more general GEMM kernel. It should be able to compute C+=AB for arbitrary A, B and C matrices in the range of 1≤M≤1024, 1≤N≤1024, and 1≤K≤2048.

At first, we had to decide on how to block the matrices. In the M dimension, we decided to use a block size of 8 and in the N dimension we decided to use a block size of 4. The larger we keep the block size, the more efficiently we can use loads, stores and FMLA instructions. However, the issue with large block sizes is that we need to write a lot of specialized kernels for all M and N dimensions smaller or equal to the block size. If the input parameters are not multiples of the block size, we need to write additional code to handle the remaining elements. 

For a block size of M = 8, we already wrote such a kernel in pure assembly, see :ref:`generic-kernel`. Using this generic kernel as a starting point, we reduced the N dimension from 6 to 4. Our reasoning was that we wanted to reduce the number of specialized kernels we need to write. Additionally, we assumed that most numbers commonly used in practice are multiples of 4 instead of 6, thus not depending on the specialized kernels. With this change, we implemented the ``matmul_m_4_k`` kernel, which can compute C+=AB for any matrices where M and K can be chosen freely, but N is fixed to 4.

The kernel first computes the number of blocks in the M dimension and the remaining elements. 

.. literalinclude:: ../../src/kernels/matmul/subkernels/matmul_m_4_k.cpp
    :language: cpp
    :lines: 22-24
    :caption: Computing the number of blocks in the M dimension

Using these numbers, we can call the specialized kernels:

.. literalinclude:: ../../src/kernels/matmul/subkernels/matmul_m_4_k.cpp
    :language: cpp
    :lines: 53-93
    :caption: Calling specialized kernels for different M dimensions

But what does such a specialized kernel look like? For the most part, they are similar to the microkernels we implemented before. The only difference is that we need to adjust the loads, stores and FMLA instructions for fixed M dimensions. For example in the case of M = 3:

.. literalinclude:: ../../src/kernels/matmul/subkernels/matmul_m_4_k.cpp
    :language: cpp
    :lines: 354-357
    :caption: Loading a column of C with M = 3

While we can simply load a double word when M = 2 or even a quad word when M = 4, we need to divide our loads into two parts when M = 3. First, we load a double word and then the remaining single word. The same applies to the stores:

.. literalinclude:: ../../src/kernels/matmul/subkernels/matmul_m_4_k.cpp
    :language: cpp
    :lines: 416-419
    :caption: Storing a column of C with M = 3

The FMLA instructions are also adjusted to the M dimension. For example, when M = 3, we need to use two FMLA instructions to compute the result:

.. literalinclude:: ../../src/kernels/matmul/subkernels/matmul_m_4_k.cpp
    :language: cpp
    :lines: 381-384
    :caption: FMLA instructions

While one could use an ``fmla`` instruction and zero padding, we decided to use one ``fmla`` instruction for the first two elements and one ``fmadd`` instruction for the last element. We did not evaluate any performance differences between the two approaches, but chose the second one because to us it seemed more readable and easier to understand. The other specialized kernels for M = 1, 2, 4, 5, 6 and 7 are implemented in a similar way.

Having implemented the ``matmul_m_4_k`` kernel, we can now turn our attention towards the ``matmul_m_n_k`` kernel. Since we decided to block N by 4, we can use the same approach as before. We first compute the number of blocks in the N dimension and the remaining elements.

.. literalinclude:: ../../src/kernels/matmul/matmul_m_n_k.cpp
    :language: cpp
    :lines: 27-29
    :caption: Computing the number of blocks in the N dimension

``nLoopRemainder`` can take any value between 0 and 3, which means that additionally to the ``matmul_m_4_k`` kernel where ``nLoopRemainder`` is 0, we need to implement specialized kernels for ``nLoopRemainder`` = 1, 2 and 3. The specialized kernels are basically the same as the ``matmul_m_4_k`` kernel, but we simply removed some of the loads, stores and FMLA instructions. For the more curious reader, the specialized kernels can be found in the files ``src/kernels/matmul_m_1_k.cpp``, ``src/kernels/matmul_m_2_k.cpp`` and ``src/kernels/matmul_m_3_k.cpp``.

For the whole N loop, we use switch statements to call the specialized kernels. The final implementation looks like this:

.. literalinclude:: ../../src/kernels/matmul/matmul_m_n_k.cpp
    :language: cpp
    :lines: 62-151
    :caption: N loop

The full code is available in the file ``src/kernels/matmul_m_n_k.cpp``.

4.2.2 Verification of the GEMM kernel with lda=M, ldb=K, ldc=M
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This task requires us to verify the correctness of our ``matmul_m_n_k`` kernel by comparing to a reference implementation for 1≤M≤64, 1≤N≤64, K∈[1,16,32,64,128], and lda=M, ldb=K, ldc=M.
We realized this verification using a ``Catch2`` unit test:

.. literalinclude:: ../../src/kernels/matmul/matmul_m_n_k.test.cpp
    :language: cpp
    :lines: 8-64
    :caption: Unit test for the ``matmul_m_n_k`` kernel with lda=M, ldb=K, ldc=M

The M and N dimensions are generated randomly, while the K dimension is fixed to multiple given values. We compute the expected result using high level C++ code and compare it to the result of our kernel.

4.2.3 Verification of the GEMM kernel with lda>M, ldb>K or ldc>M
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This task is very similar to the previous one, but we need to verify the correctness of our ``matmul_m_n_k`` kernel for 1≤M≤64, 1≤N≤64, K∈[1,16,32,64,128], and lda>M, ldb>K or ldc>M. This means that we need to store the matrices in a way that they are not contiguous in memory. We can do this by first choosign strides that are larger than the M, N and K dimensions. Then we can use the strides to compute the addresses of the elements in the matrices. Next, we can use this strides to first allocate memory that is larger than the matrices and then only set the elements that are used in the computation. The other elements, which will be skipped due to the strides, will be set to 0. Lastly, we call our kernel and compare the result to the expected result:

.. literalinclude:: ../../src/kernels/matmul/matmul_m_n_k.test.cpp
    :language: cpp
    :lines: 66-149
    :caption: Unit test for the ``matmul_m_n_k`` kernel with lda>M, ldb>K or ldc>M

4.2.4 Benchmarking the GEMM kernel performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the benchmarking we enhanced our ``benchmarking.cpp`` file that was used for the previous tasks.
Our task was to benchmark the performance of our generated kernels and report the measured
performance for 1≤M≤64, 1≤N≤64, K∈[1,16,32,64,128], lda=M, ldb=K and ldc=M. 

We were also given a baseline CSV file, which gave us a structure, on how to safe our benchmarking performance.
Our idea was run each of these benchmarks for a time of ``1.5s`` in order to guarantee comparable results.
During this time we calculated the number of iterations our ``matmul_m_n_k`` kernel would perform.
Using this metrics we could then calculate the performance in GFLOPs for the respective execution.

.. literalinclude:: ../../src/benchmark.cpp
    :language: cpp
    :lines: 145-162
    :caption: ``matmul_m_n_k`` benchmarking approach for different M, N, and K.

The results that we obtained were safed under ``src/benchmark/gemm_perf.csv``. 

.. literalinclude:: ../../src/benchmark/gemm_perf.csv
    :language: text
    :lines: 1-15
    :caption: Snippet of executed benchmarks for ``matmul_m_n_k``

Looking at the benchmarks we could see that the performance varies a lot for different configurations.
Also as we reduced our "standard" execution from ``16_6`` to ``8_4`` we were not as performant as we could
be, especially for larger matrices. When we compare our results, we get approximately the same performance
as for our :ref:`generic-kernel`.


4.3 Batch-Reduce GEMM
-----------------------

After generating our GEMM kernel for different values for the M, N, and K dimensions, we are now implementing
a batched version of this kernel. That means we are now implementing kernels to comply with matrix multiplications 
of the form: C+=∑AᵢBᵢ.

4.3.1 Support Batch-Reduce GEMMs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We started by altering our ``generate`` function, so that we would now accept a ``batch_size``.

.. literalinclude:: ../../src/Brgemm.cpp
    :language: cpp
    :lines: 57-66
    :caption: handling of invalid values for ``br_size`` in the ``generate`` function

.. literalinclude:: ../../src/Brgemm.cpp
    :language: cpp
    :lines: 77-90
    :caption: implementation of ``br_size`` in the ``generate`` function

We based our implementation for the ``matmul_br_m_n_k`` on our assembly implementation of the :ref:`batch-reduce GEMM <3.6 Batch-Reduce GEMM>`.
As we now had the additional values ``br_stride_a`` and ``br_stride_a`` we needed to slightly adjust the use of our registers.
Apart from that, we were ready to start. 

The first step we took was to initialize the loop counter for the batch dimension.

.. literalinclude:: ../../src/kernels/matmul/matmul_br_m_n_k.cpp
    :language: cpp
    :lines: 66-70
    :caption: initialize loop counter for batch dimension

Our second step was to make sure that after a GEMM has finished, we 
would increment the pointers, to move to the next respective matrices.

.. literalinclude:: ../../src/kernels/matmul/matmul_br_m_n_k.cpp
    :language: cpp
    :lines: 160-176
    :caption: move to the next A and B matrix and restore the position for matrix C

These were the only changes we had to make. Between the initialization of the loop 
and jumping to the next matrices, we would loop over our :ref:`matmul_m_n_k kernel <4.2 GEMM>`.

4.3.2 Verification of the Batch-Reduce GEMM kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Similar to the GEMM kernel, we also tested our implementation of the batch-reduce GEMM.
We executed several initializations of our kernel, using a similar approach to the testing of the GEMM kernel.

.. literalinclude:: ../../src/kernels/matmul/matmul_br_m_n_k.test.cpp
    :language: cpp
    :lines: 8-68
    :caption: Unit test for the ``matmul_br_m_n_k`` kernel

4.3.3 Benchmarking the Batch-Reduce GEMM kernel performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For the benchmarking we, again, enhanced our ``benchmarking.cpp`` file.
We introduced a new function that should handle  1≤M≤64, 1≤N≤64, K∈[1,16,32,64,128], lda=M, ldb=K and ldc=M.

We reduced the time for our benchmarks to ``1.0s``.

Beside the fact, that we would now consider 16 Matrices for A and B, the calculation 
for the GFLOPs was than similar to the normal ``GEMM``.

.. literalinclude:: ../../src/benchmark.cpp
    :language: cpp
    :lines: 206-223
    :caption: ``matmul_br_m_n_k`` benchmarking approach for a batch size of 16 and different M, N, and K values

The results that we obtained were safed under ``src/benchmark/br_gemm_perf.csv``. 