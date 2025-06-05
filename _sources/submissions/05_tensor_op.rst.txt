##############################
5. Tensor Operation Backend
##############################

After experimenting with different neon implementations and developing kernels 
for our gemm and brgemm, and most recently for the unary primitives, it is now time 
to combine all of these kernels together in a backend.

*********************
5.1 User Interface
*********************

The first thing that we need for our backend is a common entry point. 
Our common entry point is our ``setup`` function. Within the setup function we 
parse a number of configuration parameters, from which the corresponding kernels and 
primitives are constructed at runtime.

In our setup function, we check several things:

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 15-34
    :lineno-match:
    :caption: error handling for dimensions and execution types
    :dedent:

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 36-45
    :lineno-match:
    :caption: assigning configuration parameters to member variables
    :dedent:

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 47-73
    :lineno-match:
    :caption: check data type and assign correct first and last touch, and main primitive
    :dedent:

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 114-135
    :lineno-match:
    :caption: find the first ``prim`` and ``seq`` position in ``exec_types``
    :dedent:

We need to know the position of the first prim, to determine when we need to call the main kernel instead of recursively going deeper into the loop structure. 
In other words, we traverse the first sequential loops and as soon as we reach the first primary dimension, we start calling the main kernel.

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 86-112
    :lineno-match:
    :caption: assign the size of the ``prim`` dimensions according to the order in ``dim_types``
    :dedent:

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 149-168
    :lineno-match:
    :caption: assign the size of the ``seq`` dimensions according to the order in ``dim_types``
    :dedent:

After checking all these things, we were then able to create our kernels accordingly.

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 184-238
    :lineno-match:
    :caption: construct kernels based on assigned member variables
    :dedent:

*************************************
5.2 Recursive Loops over Primitives
*************************************

After constructing our kernels, we still needed to build together an ``execution`` function, 
in order to combine our main primitive with our first and last touches.

Our starting point is an ``execute`` function that takes the pointers to our matrices and passes 
them to our ``execute_iter`` function.

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 247-274
    :lineno-match:
    :caption: starting point: ``execute`` function
    :dedent:

The 'real' execution happens in the ``execute_iter`` function. We first check if the current iteration is the first or last access to a block in our output matrix.
Next, we update the pointers to the matrices accordingly.

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 291-301
    :lineno-match:
    :caption: calculate if it is the first or last access in our output matrix and update pointers
    :dedent:

In the following step, we use our ``execute_iter`` function to recursively call the ``execute_iter`` function based on how many ``seq`` dimensions exist in our ``exec_types``.

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 303-312
    :lineno-match:
    :caption: recursive call to ``execute_iter``
    :dedent:

If we have no further recursive call, we can execute the kernels.

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 313-335
    :lineno-match:
    :caption: execute the kernels
    :dedent:

.. _5.3 Sequential Benchmarking:

*******************************
5.3 Performance Benchmarking
*******************************

To test the performance of our at runtime constructed kernels and to see if everything works seamlessly together, 
we were performing some reference benchmarks. 

We were given a number of configuration parameters, that we should check:

.. list-table:: Benchmark Configuration
   :widths: 25 25 25 25
   :header-rows: 1

   * - Variable 
     - 1st Value 
     - 2nd Value
     - 3rd Value
   * - **dtype**
     - FP32
     - FP32
     - FP32
   * - **prim_first_touch**
     - None
     - None
     - Zero
   * - **prim_main**
     - GEMM
     - BRGEMM
     - BRGEMM
   * - **prim_last_touch**
     - None
     - None
     - ReLU
   * - **dim_types**
     - (M, N, K, M, N, K)
     - (M, N, K, M, N, K)
     - (M, N, K, M, N, K)
   * - **exec_types**
     - (Seq, Seq, Seq, Prim, Prim, Prim)
     - (Seq, Seq, Prim, Prim, Prim, Prim)
     - (Seq, Seq, Prim, Prim, Prim, Prim)
   * - **dim_sizes**
     - (32, 32, 8, 32, 32, 32)
     - (32, 32, 8, 32, 32, 32)
     - (32, 32, 8, 32, 32, 32)
   * - **strides_in0**
     - (8192, 0, 1024, 1, 0, 32)
     - (8192, 0, 1024, 1, 0, 32)
     - (8192, 0, 1024, 1, 0, 32)
   * - **strides_in1**
     - (0, 8192, 1024, 0, 32, 1)
     - (0, 8192, 1024, 0, 32, 1)
     - (0, 8192, 1024, 0, 32, 1)
   * - **strides_out**
     - (32768, 1024, 0, 1, 32, 0)
     - (32768, 1024, 0, 1, 32, 0)
     - (32768, 1024, 0, 1, 32, 0)
  
However, when benchmarking our implementation with these strides, we were running into memory leaks. 
For that reason, we decided to adjust the strides of the benchmarks slightly.

.. list-table:: Our Benchmark Configuration (bold are changes)
   :widths: 25 25 25 25
   :header-rows: 1

   * - Variable 
     - 1st Value 
     - 2nd Value
     - 3rd Value
   * - **dim_sizes**
     - (32, 32, 8, 32, 32, 32)
     - (32, 32, 8, 32, 32, 32)
     - (32, 32, 8, 32, 32, 32)
   * - **strides_in0**
     - (**1024**, 0, **32768**, 1, 0, 32)
     - (**1024**, 0, **32768**, 1, 0, 32)
     - (**1024**, 0, **32768**, 1, 0, 32)
   * - **strides_in1**
     - (0, 8192, **32**, 0, **1024**, 1)
     - (0, 8192, **32**, 0, **1024**, 1)
     - (0, 8192, **32**, 0, **1024**, 1)
   * - **strides_out**
     - (**32**, **32768**, 0, 1, **1024**, 0)
     - (**32**, **32768**, 0, 1, **1024**, 0)
     - (**32**, **32768**, 0, 1, **1024**, 0)
  
When benchmarking our configurations we achieved the following ``GFLOP`` performance:

.. literalinclude:: ../../benchmarks/tensor_operation_benchmarks.txt
    :language: text
    :lineno-match:
    :caption: ``GFLOP`` performance of our benchmark configuration
    :dedent:

The results show that we achieve between ``71-73 GFLOPs`` for all our executions. 
These results are somewhat consistent with calling the kernels themselves independently.

.. note::
    Since the submission we made some minor changes to our implementation.
    First, we fixed some errors and were then able to use the strides that we were provided with.
    Secondly, we decided to enhance our ``matmul_m_n_k`` implementation. 
    Afterwards we able to calculate kernels of size ``16x4`` instead of ``8x4``.
    This helped us increase the results from ``71-73 GFLOPs`` to around ``90-91 GFLOPs``.

.. literalinclude:: ../../benchmarks/tensor_operation_benchmarks_2.txt
    :language: text
    :lineno-match:
    :caption: ``GFLOP`` performance initial benchmark configuration with enhanced ``matmul`` kernel
    :dedent:

**********************************
5.4 Shared Memory Parallelization
**********************************

To enable the execution of shared loops, we needed to make a few adjustments to our ``setup`` code:

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 137-147
    :lineno-match:
    :caption: gather shared loop id's and dimension sizes
    :dedent:

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 169-181
    :lineno-match:
    :caption: assign the size of the ``shared`` dimensions according to the order in ``dim_types``
    :dedent:

In our execute function we would just needed to check if our ``m_num_parallel_loops`` variable would be greater 
than zero. If this was the case we would execute our ``execute_iter_parallel`` function:

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 345-350
    :lineno-match:
    :caption: multiply ``shared`` loop sizes to get total number of iterations
    :dedent:

The idea is to get a flat iteration space that can be used to parallelize over.

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 352-363
    :lineno-match:
    :caption: multiply ``shared`` loop sizes to get total number of iterations
    :dedent:

We 'unflatten' the OpenMP iteration index ``l_it_all`` into a set of loop indices, one for each shared loop dimension. 
These indices are then used to compute the offsets for the ``in0``, ``in1``, and ``out`` tensors: 

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 365-379
    :lineno-match:
    :caption: calculate the tensor ``offsets``
    :dedent:

Here we are calculating the offset for the current thread. 
Every shared loop contributes to the calculation with its corresponding stride. 

Lastly, we call our ``execute_iter`` function. 
Depending on whether we have a ``seq`` dimension, we need to be careful, which id we pass to the function:

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 381-387
    :lineno-match:
    :caption: call remaining loops with ``execute_iter``
    :dedent:

We were also executing the benchmark configurations from the :ref:`sequential execution<5.3 Sequential Benchmarking>` task.
We were executing our benchmarks using ``OMP_NUM_THREADS=4``:

.. literalinclude:: ../../benchmarks/shared_tensor_operation_benchmarks.txt
    :language: text
    :lineno-match:
    :caption: ``GFLOP`` performance for ``4 shared`` loop execution
    :dedent:

With the parallelization we achieve about ``360 - 390 GFLOPs``. 

**********************************
5.5 Optimization Passes
**********************************

Our approach to enhancing the performance of the tensor operations was to use a vector of ``struct``'s for each dimension that we have got:

.. literalinclude:: ../../src/ir/Dimension.h
    :language: cpp
    :lines: 17-60
    :lineno-match:
    :caption: call remaining loops with ``execute_iter``
    :dedent:

This ``struct`` is used to store all information about a dimension.

After setting this up, we could create our optimization passes.

5.5.1 Primitive Identification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first optimization that we performed was to find primitive dimensions. 
This optimization would be useful, for cases, where we were given only sequential loops. 
Our approach to this optimization was the following: 

.. literalinclude:: ../../src/ir/Optimizer.cpp
    :language: cpp
    :lines: 33-55
    :lineno-match:
    :caption: finding the ``K2 prim`` dimension for the ``BRGEMM`` case
    :dedent:

We were trying to identify the respective dimensions by looking at the strides of the ``in1`` and ``out`` tensors.
As a starting point we use that for column-major ``BRGEMM``'s we have to have a certain mask for our tensors ``...M, ...K1 -> ...M``, which we were trying to follow.
This means that the ``K2`` that we would need for our ``BRGEMM`` should not have any unit-stride in the first input tensor.

Similarly, we did the same for:

- ``M`` dimension
- ``N`` dimension
- ``K1`` dimension

For the ``N`` dimension, where we did not have any indication about which matrix to choose, we simply choose the ``N`` dimension, with the smallest stride:

.. literalinclude:: ../../src/ir/Optimizer.cpp
    :language: cpp
    :lines: 85-108
    :lineno-match:
    :caption: finding the ``N prim`` dimension with smallest stride
    :dedent:

We did the 'identification' process in the order ``K2, M, N, K1``. 
The reason for this order was that after the identification, we would rotate the respective dimension to the end of the order.
This would then ultimately lead to the structure: ``..., K2, M, N, K1`` for our 'identified' primitive dimensions.

5.5.2 Splitting Dimensions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For our second optimization pass we decided to look at the dimension sizes of our loops. 
That means for the case that a ``prim`` dimension would be larger than ``1024`` we would decide to split it in two dimensions:

.. literalinclude:: ../../src/ir/Optimizer.cpp
    :language: cpp
    :lines: 150-178
    :lineno-match:
    :caption: split large ``prim`` dimensions
    :dedent:

If a ``prim`` dimension would be large enough, we would call our ``findBestSplit`` function. 
Our ``findBestSplit`` function is designed after the dimensions in our ``matmul_m_n_k`` kernel. 
Depending on the dimension we want to split we are here selecting the ideal sizes. 

That means we start with the ``M`` dimension to find an appropriate match: 

.. literalinclude:: ../../src/ir/Optimizer.cpp
    :language: cpp
    :lines: 230-254
    :lineno-match:
    :caption: ``M`` dimension split
    :dedent:

Similarly, we do the same for the ``N`` dimension, where we want multiples of ``4`` and for ``K`` we are flexible.

5.5.3 Shared Memory Parallelization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our third optimization pass was to make all loops that were not a ``prim`` dimension and of the dimension-type ``M`` or ``N`` a ``shared`` loop.
For that we initially check how many loops are already of dimension-type ``shared``:

.. literalinclude:: ../../src/ir/Optimizer.cpp
    :language: cpp
    :lines: 184-200
    :lineno-match:
    :caption: finding the ``N prim`` dimension with smallest stride
    :dedent:

For the case that we already have a high number of ``shared`` loops we do not create any more and simply return. 
Otherwise we check the ``seq`` dimensions for potential candidates:

.. literalinclude:: ../../src/ir/Optimizer.cpp
    :language: cpp
    :lines: 202-214
    :lineno-match:
    :caption: select ``shared`` loop candidates
    :dedent:

As a last step we move all our ``shared`` loops to the front of the order:

.. literalinclude:: ../../src/ir/Optimizer.cpp
    :language: cpp
    :lines: 216-221
    :lineno-match:
    :caption: move ``shared`` loops to the front
    :dedent:

5.5.6 Performance Benchmarks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We also benchmarked the results for two given configurations:

.. literalinclude:: ../../benchmarks/optimized_tensor_operation_benchmarks.txt
    :language: text
    :lineno-match:
    :caption: ``GFLOP`` performance for sample configurations
    :dedent:

For the execution of these configs we receive around ``250 - 260 GFLOPs``. 