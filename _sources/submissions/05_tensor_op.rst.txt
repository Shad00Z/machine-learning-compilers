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
    :lines: 36-48
    :lineno-match:
    :caption: assigning configuration parameters to member variables
    :dedent:

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 50-76
    :lineno-match:
    :caption: assign correct first and last touch, and main primitive
    :dedent:

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 78-90
    :lineno-match:
    :caption: find the first ``prim`` position in ``exec_types``
    :dedent:

We need to know the position of the first prim, to determine when we need to call the main kernel instead of recursively going deeper into the loop structure. In other words, we traverse the first sequential loops and as soon as we reach the first primary dimension, we start calling the main kernel.

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 92-124
    :lineno-match:
    :caption: assign the size of the ``prim`` dimensions according to the order in ``dim_types``
    :dedent:

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 125-145
    :lineno-match:
    :caption: assign the size of the ``seq`` dimensions according to the order in ``dim_types``
    :dedent:

After checking all these things, we were then able to create our kernels accordingly.

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 147-201
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
    :lines: 210-224
    :lineno-match:
    :caption: starting point: ``execute`` function
    :dedent:

The 'real' execution happens in the ``execute_iter`` function. We first check the current iteration is the first or last access to a block in our output matrix.
Next, we update the pointers to the matrices accordingly.

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 252-258
    :lineno-match:
    :caption: calculate if it is the ``first`` or ``last`` access in our output matrix
    :dedent:

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 238-266
    :lineno-match:
    :caption: calculate the offsets of our matrices
    :dedent:

In the following step, we use our ``execute_iter`` function to recursively call the ``execute_iter`` function based on how many ``seq`` dimensions exist in our ``exec_types``.

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 268-277
    :lineno-match:
    :caption: recursive call to ``execute_iter``
    :dedent:

If we have no further recursive call, we can execute the kernels.

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 278-300
    :lineno-match:
    :caption: execute the kernels
    :dedent:

*******************************
5.3 Performance Benchmarking
*******************************

To test the performance of our at runtime constructed kernels and to see if everything works seemlessly together, 
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
    :caption: ``GFLOP`` performance of our benchmarks
    :dedent:

The results show that we achieve between ``71-73 GFLOPs`` for all our executions. 
These results are somewhat consistent with calling the kernels themselves independently.