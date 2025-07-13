##############################
5. Tensor Operation Backend
##############################

After experimenting with different neon implementations and developing kernels 
for our GEMM and BRGEMM, and most recently for the unary primitives, it is now time 
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
    :lines: 17-68
    :lineno-match:
    :caption: error handling for dimensions, execution types and data type
    :dedent:

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 70-90
    :lineno-match:
    :caption: assigning configuration parameters to member variables
    :dedent:

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 92-113
    :lineno-match:
    :caption: find the first ``prim`` and ``seq`` position in ``exec_types``
    :dedent:

We need to know the position of the first prim, to determine when we need to call the main kernel instead of recursively going deeper into the loop structure. 
In other words, we traverse the first sequential loops and as soon as we reach the first primary dimension, we start calling the main kernel.

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 127-153
    :lineno-match:
    :caption: assign the size of the ``prim`` dimensions according to the order in ``dim_types``
    :dedent:

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 155-188
    :lineno-match:
    :caption: assign the size of the ``seq`` and ``shared`` dimensions according to the order in ``dim_types``
    :dedent:

After checking all these things, we were then able to create our kernels accordingly.

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 256-317
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
    :lines: 326-340
    :lineno-match:
    :caption: starting point: ``execute`` function
    :dedent:

The 'real' execution happens in the ``execute_iter`` function. We first check if the current iteration is the first or last access to a block in our output matrix.
Next, we update the pointers to the matrices accordingly.

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 367-378
    :lineno-match:
    :caption: calculate if it is the first or last access in our output matrix and update pointers
    :dedent:

In the following step, we use our ``execute_iter`` function to recursively call the ``execute_iter`` function based on how many ``seq`` dimensions exist in our ``exec_types``.

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 381-389
    :lineno-match:
    :caption: recursive call to ``execute_iter``
    :dedent:

If we have no further recursive call, we can execute the kernels.

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 392-411
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
    :lines: 340-348
    :lineno-match:
    :caption: gather shared loop id's and dimension sizes
    :dedent:

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 175-187
    :lineno-match:
    :caption: assign the size of the ``shared`` dimensions according to the order in ``dim_types``
    :dedent:

In our execute function we would just needed to check if our ``m_num_parallel_loops`` variable would be greater 
than zero. If this was the case we would execute our ``execute_iter_parallel`` function:

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 422-427
    :lineno-match:
    :caption: multiply ``shared`` loop sizes to get total number of iterations
    :dedent:

The idea is to get a flat iteration space that can be used to parallelize over.

We 'unflatten' the OpenMP iteration index ``l_it_all`` into a set of loop indices, one for each shared loop dimension. 
These indices are then used to compute the offsets for the ``in0``, ``in1``, and ``out`` tensors: 

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 434-447
    :lineno-match:
    :caption: calculate the tensor ``offsets``
    :dedent:

Here we are calculating the offset for the current thread. 
Every shared loop contributes to the calculation with its corresponding stride. 

Lastly, we call our ``execute_iter`` function. 
Depending on whether we have a ``seq`` dimension, we need to be careful, which id we pass to the function:

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 461-466
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
    :lines: 209-231
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
    :lines: 261-279
    :lineno-match:
    :caption: finding the ``N prim`` dimension with smallest stride
    :dedent:

We did the 'identification' process in the order ``K2, M, N, K1``. 
The reason for this order was that after the identification, we would rotate the respective dimension to the end of the order.
This would then ultimately lead to the structure: ``..., K2, M, N, K1`` for our 'identified' primitive dimensions.

5.5.2 Dimension Splitting
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For our second optimization pass we decided to look at the dimension sizes of our loops. We introduced a ``max_kernel_size`` parameter,
which specifies the maximum allowed size for a dimension. If a dimension with a size larger than the maximum is found, the dimension splitter
will try to split it into new dimensions with optimized sizes. The entry point for this optimization is the ``splitDimensions`` function:

.. code-block:: cpp
    :caption: splitDimensions function of the Optimizer

    void mini_jit::ir::Optimizer::splitDimensions(std::vector<mini_jit::ir::Dimension> &dimensions,
                                                  int64_t max_kernel_size)
    {
        // Dimensions should be split if they are too large (> max_kernel_size)
        for (size_t i = 0; i < dimensions.size(); i++)
        {
            if (dimensions[i].size > max_kernel_size)
            {
                int64_t l_size_dim_0 = 0;
                int64_t l_size_dim_1 = 0;
                findBestSplit(dimensions[i].size,
                              max_kernel_size,
                              dimensions[i].type,
                              l_size_dim_0,
                              l_size_dim_1);
                if (l_size_dim_0 > 1)
                {
                    // create a new seq dimension
                    mini_jit::ir::Dimension l_dim_new(dimensions[i].type,
                                                      exec_t::seq,
                                                      l_size_dim_0,
                                                      dimensions[i].stride_in0 * l_size_dim_1,
                                                      dimensions[i].stride_in1 * l_size_dim_1,
                                                      dimensions[i].stride_out * l_size_dim_1);
                    // update the original dimension size
                    dimensions[i].size = l_size_dim_1;
                    // insert the new dimension at the back, so it will be checked for a split again
                    dimensions.push_back(l_dim_new);
                }
            }
        }
    }

For each dimension, it finds the bets split for our kernels if the dimension size is too large and creates a new dimension. The size of the original dimension is updated to ``l_size_dim_1``, and it will be smaller than or equal to ``max_kernel_size``. However, the new dimension ``l_dim_new`` might still have a larger dimension size than ``max_kernel_size``, which is why it is inserted at the end of the dimensions vector, where it will be checked for a possible split in a later iteration.

But what does ``findBestSplit`` do?

The way our kernels were implemented makes their execution more efficient for specific dimension sizes. Considering the **M** dimension, a size that is a multiple of **16** is optimal for most kernels, since we manually optimized the kernels for this case. As for the **N** dimension size, a multiple of **4** is optimal for most kernels. 
In the **K** dimension, we do not have such optimizations and the dimension size can be chosen freely, as long as it is smaller than ``max_kernel_size``. 
The following code snippet shows the implementation of ``findBestSplit`` for the **M** and **N** dimensions:

.. code-block:: cpp
    :caption: findBestSplit function of the Optimizer for M and N

    o_size_0 = 1;
    o_size_1 = i_size;
    if (i_type == dim_t::m)
    {
        // multiples of (multiples of) 4 are efficient (LDP, STP)
        for (int64_t i = 16; i > 4; i -= 4)
        {
            findLargestMultipleOfDivisor(i, i_size, i_max_kernel_size, o_size_0, o_size_1);
            if (o_size_0 > 1)
            {
                return;
            }
        }
        // split by 2
        findLargestMultipleOfDivisor(2, i_size, i_max_kernel_size, o_size_0, o_size_1);
        if (o_size_0 > 1)
        {
            return;
        }
    }
    // for n, we want multiples of 4
    else if (i_type == dim_t::n)
    {
        // split by 4
        findLargestMultipleOfDivisor(4, i_size, i_max_kernel_size, o_size_0, o_size_1);
        if (o_size_0 > 1)
        {
            return;
        }
        // split by 2
        findLargestMultipleOfDivisor(2, i_size, i_max_kernel_size, o_size_0, o_size_1);
        if (o_size_0 > 1)
        {
            return;
        }
    }

But what does ``findLargestMultipleOfDivisor`` do?

As the name suggests, this helper function tries to find the largest multiple of a given divisor. Let's say the given divisor is ``16``, the input dimension size is **1600** and the ``i_max_kernel_size`` is **1024**.
Then, ``findLargestMultipleOfDivisor`` will try to find the largest multiple of **16** which divides **1600** and is smaller than or equal to **1024**. The result of this computation is **2** for ``o_size_0`` and **800** for ``o_size_1``.
For the more curious reader, the implementation of ``findLargestMultipleOfDivisor`` is given below:

.. code-block:: cpp
    :caption: ``findLargestMultipleOfDivisor`` function of the Optimizer

    void mini_jit::ir::Optimizer::findLargestMultipleOfDivisor(int64_t i_divisor,
                                                               int64_t i_size,
                                                               int64_t i_max_size,
                                                               int64_t &o_size_0,
                                                               int64_t &o_size_1)
    {
        if (i_divisor <= 0 || i_size <= 0 || i_max_size <= 0 || i_divisor > i_max_size)
        {
            return;
        }

        // start: largest multiple of i_divisor < i_max_size
        int64_t l_max_divisible = (i_max_size / i_divisor) * i_divisor;
        for (int64_t l_m = l_max_divisible; l_m >= i_divisor; l_m -= i_divisor)
        {
            // we found an m that divides i_size! it is also the largest
            if (i_size % l_m == 0)
            {
                o_size_0 = i_size / l_m;
                o_size_1 = l_m;
                return;
            }
        }
    }

5.5.3 Shared Memory Parallelization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our third optimization pass was to make all loops that were not a ``prim`` dimension and of the dimension-type ``M`` or ``N`` a ``shared`` loop.
For that we initially check how many loops are already of dimension-type ``shared``:

.. literalinclude:: ../../src/ir/Optimizer.cpp
    :language: cpp
    :lines: 364-371
    :lineno-match:
    :caption: finding possible iterations for shared loops
    :dedent:

For the case that we already have a high number of ``shared`` loops we do not create any more and simply return. 
Otherwise we check the ``seq`` dimensions for potential candidates:

.. literalinclude:: ../../src/ir/Optimizer.cpp
    :language: cpp
    :lines: 386-398
    :lineno-match:
    :caption: select ``shared`` loop candidates
    :dedent:

As a last step we move all our ``shared`` loops to the front of the order:

.. literalinclude:: ../../src/ir/Optimizer.cpp
    :language: cpp
    :lines: 401-405
    :lineno-match:
    :caption: move ``shared`` loops to the front
    :dedent:


.. _dimension-fusion:

5.5.4 Dimension Fusion
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. note::

    This part of the Optimizer was implemented much later, as part of the :ref:`project-week-2` of our final project phase.

The idea behind **Dimension Fusion** is that when certain dimensions have very small sizes, fusing them can improve cache efficiency and simplify tensor expressions. It also enables our existing dimension splitter to operate more effectively, as it can now split the fused dimensions in ways optimized for our kernels, rather than being constrained by the original tensor structure. In other words, **Dimension Fusion** will be the first step in our optimizer, simplifying the tensor expression upfront so it can then be split in an optimized way and finally, have its primitive dimensions identified.

The first step was to introduce a new ``min_kernel_size`` parameter. It allows the user to specify the minimum dimension size a kernel should have. If a dimension is smaller than that, the dimension fuser will try to look for candidates to fuse with. This process happens in the new ``fuseDimensions`` function of the Optimizer.

.. code-block:: cpp
    :caption: Dimension Fusing in the Optimizer

    void mini_jit::ir::Optimizer::fuseDimensions(std::vector<mini_jit::ir::Dimension> &dimensions,
                                                 int64_t min_kernel_size)
    {
        for (size_t i = 0; i < dimensions.size(); i++)
        {
            mini_jit::ir::Dimension &l_dim_0 = dimensions[i];
            if (l_dim_0.size < min_kernel_size)
            {
                // find a dimension that can be fused with the current one
                for (size_t j = 0; j < dimensions.size(); j++)
                {
                    if (i == j) continue; // skip self

                    mini_jit::ir::Dimension &l_dim_1 = dimensions[j];
                    if (l_dim_0.type == l_dim_1.type &&
                        (l_dim_0.exec_type == l_dim_1.exec_type ||
                        l_dim_0.exec_type == exec_t::undefined ||
                        l_dim_1.exec_type == exec_t::undefined) &&
                        l_dim_1.stride_in0 == l_dim_0.size * l_dim_0.stride_in0 &&
                        l_dim_1.stride_in1 == l_dim_0.size * l_dim_0.stride_in1 &&
                        l_dim_1.stride_out == l_dim_0.size * l_dim_0.stride_out)
                    {
                        // fuse the two dimensions
                        l_dim_0.size *= l_dim_1.size;
                        // remove the fused dimension
                        dimensions.erase(dimensions.begin() + j);
                        j--; // adjust index after erasing
                    }
                }
            }
        }
    }

Here, ``l_dim_0`` is the dimension whose size is smaller than ``min_kernel_size``, meaning that we would like to fuse it with another candidate. However, the candidate (``l_dim_1``) the function looks for needs to fulfill some criteria:

#. Same dimension type as ``l_dim_0`` (**M, N, K, C**)
#. Same execution type as ``l_dim_0``, or either type is undefined
#. The stride of ``l_dim_1`` needs to equal the product of the stride and size of ``l_dim_0`` (Two dimensions X and Y can be fused can be fused if for all tensors: **stride(X) = |Y| ⨉ stride(Y)**)

If a fitting candidate has been found, ``l_dim_0`` and ``l_dim_1`` can be fused. This involves multiplying the dimension sizes and removing the candidate from the dimensions vector. The strides do not need to be adjusted, as the original stride of the small ``l_dim_0`` is still correct.

After implementing dimension fusion, we also had to make adjustments to the dimension splitter. Previously, we would split dimensions by finding the largest possible split for one dimension. For example, if the given dimension size was **1600** and the maximum kernel size **1024**, the function would have returned **2** for ``o_size_0`` and **800** for ``o_size_1``. This is because **800** is the largest multiple of **16** that is less than or equal to **1024**. This was problematic however, because we then had a dimension of size **2**, which was very small and could have lead to inefficiencies. Our solution to this problem was to also introduce the ``min_kernel_size`` parameter to the dimension splitter as well. Specifically, we adjusted the ``findBestSplit`` function, which now returns a split if the ``minimum_kernel_size`` is reached:

.. code-block:: cpp
    :caption: Updated findBestSplit function for **M** dimensions

    if (i_type == dim_t::m)
    {
        // multiples of (multiples of) 4 are efficient (LDP, STP)
        for (int64_t i = 16; i > 4; i -= 4)
        {
            findLargestMultipleOfDivisor(i, i_size, i_max_kernel_size, i_min_kernel_size, o_size_0, o_size_1);
            if (o_size_0 >= i_min_kernel_size)
            {
                return;
            }
        }
        // split by 2
        findLargestMultipleOfDivisor(2, i_size, i_max_kernel_size, i_min_kernel_size, o_size_0, o_size_1);
        if (o_size_0 >= i_min_kernel_size)
        {
            return;
        }
    }

Consequently, ``findLargestMultipleOfDivisor`` had to be adjusted as well, with a simple if-condition:

.. code-block:: cpp
    :caption: Updated findLargestMultipleOfDivisor functionalities

    void mini_jit::ir::Optimizer::findLargestMultipleOfDivisor(int64_t i_divisor,
                                                              int64_t i_size,
                                                              int64_t i_max_size,
                                                              int64_t i_min_size,
                                                              int64_t &o_size_0,
                                                              int64_t &o_size_1)
    {
        if (i_divisor <= 0 || i_size <= 0 || i_max_size <= 0 || i_min_size <= 0 ||
            i_divisor > i_max_size || i_size < i_min_size)
        {
            return;
        }

        // start: largest multiple of i_divisor < i_max_size
        int64_t l_max_divisible = (i_max_size / i_divisor) * i_divisor;
        for (int64_t l_m = l_max_divisible; l_m >= i_divisor; l_m -= i_divisor)
        {
            // we found an m that divides i_size! it is also the largest
            if (i_size % l_m == 0)
            {
                int64_t candidate_size_0 = i_size / l_m;
                int64_t candidate_size_1 = l_m;
                if (candidate_size_0 >= i_min_size && candidate_size_1 >= i_min_size)
                {
                    o_size_0 = candidate_size_0;
                    o_size_1 = candidate_size_1;
                    return;
                }
            }
        }
    }

Candidates for splitting are now only chosen if both dimension sizes are at least as large as the specified minimum kernel size. 
Therefore, the new dimension splitter now outputs **50** and **32** as a split of **1600**, if ``min_kernel_size`` is set to **16**.

.. _5.5.6 Performance Benchmarks:

5.5.6 Performance Benchmarks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We also benchmarked the results for some configurations:

.. literalinclude:: ../../benchmarks/optimized_tensor_operation_benchmarks.txt
    :language: text
    :lineno-match:
    :caption: ``GFLOP`` performance for sample configurations
    :dedent:

Depending on the selected dimensions our results varied massively. The highest performance we achieved was around ``350 GFLOPs``. 

**********************************
5.6 Unary Operations
**********************************

After supporting the standard primitives, with ``GEMM`` and ``BRGEMM`` we would now allow also primitives like 
``copy`` / ``identity``, ``tranposition`` or ``permutation``.

5.6.1 Backend Extension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The distinction to the other primitves here is, that all dimensions are of type ``dim_t::c``. 
To allow this kind of primitives, we would have to make some small adjustments to our ``TensorOperation``:

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 190-210
    :lineno-match:
    :caption: find ``M`` and ``N`` dimensions based on stride in the input
    :dedent:

.. literalinclude:: ../../src/TensorOperation.cpp
    :language: cpp
    :lines: 295-303
    :lineno-match:
    :caption: generate ``identity`` primitive
    :dedent:

5.6.2 Optimization Passes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run our optimization passes on these primitives, we would again have to make some adjustment, this time in our ``Optimizer``.

For our ``identifyPrimitives`` function we would first check if we have a ``dim_t::c`` as a dimension type.
If this was the case, we would continue to find the ``prim`` dimensions:

.. literalinclude:: ../../src/ir/Optimizer.cpp
    :language: cpp
    :lines: 73-81
    :lineno-match:
    :caption: error handling for correct dimension types
    :dedent:

.. literalinclude:: ../../src/ir/Optimizer.cpp
    :language: cpp
    :lines: 82-91
    :lineno-match:
    :caption: exit early, if all ``prim`` dimensions are already set
    :dedent:

.. literalinclude:: ../../src/ir/Optimizer.cpp
    :language: cpp
    :lines: 97-125
    :lineno-match:
    :caption: exit early, if all ``prim`` dimensions are already set
    :dedent:

.. literalinclude:: ../../src/ir/Optimizer.cpp
    :language: cpp
    :lines: 127-155
    :lineno-match:
    :caption: check for ``transposition`` and find dimensions accordingly
    :dedent:

If we do not have a ``transposition`` we would simply look for the smallest stride and set the dimensions accordingly:

.. literalinclude:: ../../src/ir/Optimizer.cpp
    :language: cpp
    :lines: 160-180
    :lineno-match:
    :caption: set dimensions for ``identity`` primitive
    :dedent:

The last step would be to set the remaining undefined dimensions to ``seq``, as the next optimization would be to find ideal ``shared`` loops.

However, in our ``createSharedLoops`` function, we did not have to make any adjustments.

For our ``splitDimensions`` function, we would now also check if we had a ``dim_t::c`` as a dimension type:

.. literalinclude:: ../../src/ir/Optimizer.cpp
    :language: cpp
    :lines: 465-489
    :lineno-match:
    :caption: split dimensions of type ``dim_t::c``
    :dedent:

5.6.3 Reference Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For our reference implementation, we were using an example with 4 dimensions ``trus``. 
We would change this ``trus`` order to ``turs``.

.. literalinclude:: ../../tests/unit/TensorOperation.test.cpp
    :language: cpp
    :lines: 301-318
    :lineno-match:
    :caption: initialize sizes for tensors
    :dedent:

.. literalinclude:: ../../tests/unit/TensorOperation.test.cpp
    :language: cpp
    :lines: 320-342
    :lineno-match:
    :caption: fill the tensors with values
    :dedent:

Then we would prepare the execution, by setting all arguments accordingly:

.. literalinclude:: ../../tests/unit/TensorOperation.test.cpp
    :language: cpp
    :lines: 344-376
    :lineno-match:
    :caption: prepare arguments for execution
    :dedent:

Finally, we would execute our implementation.