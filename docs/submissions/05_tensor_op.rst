##############################
5. Tensor Operation Backend
##############################

After developing kernels for binary primitives such as GEMM and BRGEMM, as well as kernels for unary primitives like Zero and ReLU, 
it was now time to implement common interfaces that let the user create tensor operation objects.
The tensor operation object is not only be responsible for setting up and holding the used kernel objects,
but also to block the input and output matrices and execute the kernels accordingly.

*********************
5.1 User Interface
*********************

The first component of our tensor operation backend is a function that generates and sets up all necessary kernels.
This ``setup`` function parses a number of configuration parameters, from which the corresponding kernels and primitives are constructed at runtime.

The first step of the setup is to validate the input parameters for correctness. That includes checking that

#. All input vectors have the same size
#. The number of primitive dimensions aligns with the given primitive type
#. The chosen primitive types are valid and supported by our backend
#. The given data type is supported

If all supplied parameters are valid, the function scans the execution types of the dimensions for the first primitive dimension and the first sequential dimension.

.. code-block:: cpp
    :caption: Finding the first primitive and sequential dimension

    /////////////////////////////////////////////////////////////////////
    // Find first PRIM and SEQ dimensions in exec types
    /////////////////////////////////////////////////////////////////////
    auto it = std::find(exec_types.begin(), exec_types.end(), exec_t::prim);
    if (it != exec_types.end())
    {
        m_id_first_primitive_loop = std::distance(exec_types.begin(), it);
    }
    else
    {
        m_id_first_primitive_loop = 0;
    }

    it = std::find(exec_types.begin(), exec_types.end(), exec_t::seq);
    if (it != exec_types.end())
    {
        m_id_first_seq_loop = std::distance(exec_types.begin(), it);
    }
    else
    {
        m_id_first_seq_loop = -1;
    }

The function then proceeds to determine the dimension IDs based on their types.

.. code-block:: cpp
    :caption: Assigning the dimension IDs based on the dimension types

    /////////////////////////////////////////////////////////////////////
    // Read PRIM dimensions using dim types (No Copy)
    /////////////////////////////////////////////////////////////////////
    // convert to int so negative values are allowed
    int l_dim_types_size = static_cast<int>(m_dim_types.size());
    for (int i = l_dim_types_size - 1; i >= 0; i--)
    {
        if (m_exec_types[i] == exec_t::prim)
        {
            if (m_dim_id_prim_M == -1 && m_dim_types[i] == dim_t::m)
            {
                m_dim_id_prim_M = i;
            }
            else if (m_dim_id_prim_N == -1 && m_dim_types[i] == dim_t::n)
            {
                m_dim_id_prim_N = i;
            }
            else if (m_dim_id_prim_K == -1 && m_dim_types[i] == dim_t::k)
            {
                m_dim_id_prim_K = i;
            }
            else if (m_dim_id_prim_K != -1 && m_dim_id_prim_BR == -1 && m_dim_types[i] == dim_t::k)
            {
                m_dim_id_prim_BR = i;
            }
        }
    }

    /////////////////////////////////////////////////////////////////////
    // Read SEQ and SHARED dimensions
    /////////////////////////////////////////////////////////////////////
    for (size_t i = 0; i < m_dim_types.size(); ++i)
    {
        if (m_exec_types[i] == exec_t::seq)
        {
            if (m_dim_types[i] == dim_t::m)
            {
                m_dim_id_seq_M = i;
            }
            else if (m_dim_types[i] == dim_t::n)
            {
                m_dim_id_seq_N = i;
            }
            else if (m_dim_types[i] == dim_t::k)
            {
                m_dim_id_seq_K = i;
            }
        }
    }

    /////////////////////////////////////////////////////////////////////
    // Find M and N dimensions for dim_t::c (PRIM, IDENTITY)
    /////////////////////////////////////////////////////////////////////
    if (prim_main == ptype_t::identity)
    {
        // For unary operations with dim_t::c, treat the first primitive dimension as M and the second as N
        for (size_t i = 0; i < m_dim_types.size(); ++i)
        {
            if (m_exec_types[i] == exec_t::prim && m_dim_types[i] == dim_t::c)
            {
                if (m_strides_in0[i] == 1)
                {
                    m_dim_id_prim_M = i;
                }
                else
                {
                    m_dim_id_prim_N = i;
                }
            }
        }
    }

The next step is to check whether the tensor operation includes a transposition, and to adjust all strides accordingly:

.. code-block:: cpp
    :caption: Checking for transposition and adjusting strides

    /////////////////////////////////////////////////////////////////////
    // Check for Transposition
    /////////////////////////////////////////////////////////////////////
    if (m_dim_id_prim_M != -1)
    {
        int64_t l_stride_in0 = m_strides_in0[m_dim_id_prim_M];
        int64_t l_stride_out = m_strides_out[m_dim_id_prim_M];
        // set transpose flag to true if the strides are different
        m_transpose_output = l_stride_in0 != l_stride_out;
    }
    else
    {
        // idk if we can check for transposition without M
        m_transpose_output = false;
    }

    /////////////////////////////////////////////////////////////////////
    // Adjust strides based on primitive type and transposition
    /////////////////////////////////////////////////////////////////////
    if (prim_main == ptype_t::identity)
    {
        if (!m_transpose_output)
        {
            m_adjusted_stride_in0 = m_strides_in0[m_dim_id_prim_N];
            m_adjusted_stride_in1 = 0;
            m_adjusted_stride_out = m_strides_out[m_dim_id_prim_N];
        }
        else
        {
            m_adjusted_stride_in0 = m_strides_in0[m_dim_id_prim_N];
            m_adjusted_stride_in1 = 0;
            m_adjusted_stride_out = m_strides_out[m_dim_id_prim_M];
        }
    }
    else if (prim_main == ptype_t::add || prim_main == ptype_t::sub ||
             prim_main == ptype_t::mul || prim_main == ptype_t::div ||
             prim_main == ptype_t::min || prim_main == ptype_t::max)
    {
        m_adjusted_stride_in0 = m_strides_in0[m_dim_id_prim_N];
        m_adjusted_stride_in1 = m_strides_in1[m_dim_id_prim_N];
        m_adjusted_stride_out = m_strides_out[m_dim_id_prim_N];
    }
    else
    {
        // GEMM & BRGEMM
        m_adjusted_stride_in0 = m_strides_in0[m_dim_id_prim_K];
        m_adjusted_stride_in1 = m_strides_in1[m_dim_id_prim_N];
        m_adjusted_stride_out = m_strides_out[m_dim_id_prim_N];
    }
    m_adjusted_br_size_A = m_dim_id_prim_BR != -1 ? m_strides_in0[m_dim_id_prim_BR] : 1;
    m_adjusted_br_size_B = m_dim_id_prim_BR != -1 ? m_strides_in1[m_dim_id_prim_BR] : 1;

Lastly, we need to generate the kernels. As the process is very similar for most kernels, we will provide only a brief code snippet here.

.. code-block::
    :caption: First touch, gemm and brgemm kernel generation

    if (prim_first_touch != ptype_t::none)
    {
        // no transposition
        m_unary_first_touch.generate(m_dim_sizes[m_dim_id_prim_M],
                                     m_dim_sizes[m_dim_id_prim_N],
                                     0,
                                     dtype,
                                     prim_first_touch);
        m_kernel_first_touch = m_unary_first_touch.get_kernel();
    }
    if (prim_main == ptype_t::gemm)
    {
        // no transposition
        m_brgemm_main.generate(m_dim_sizes[m_dim_id_prim_M],
                               m_dim_sizes[m_dim_id_prim_N],
                               m_dim_sizes[m_dim_id_prim_K],
                               1,
                               0,
                               0,
                               0,
                               dtype);
        m_kernel_gemm_main = m_brgemm_main.get_kernel();
    }
    else if (prim_main == ptype_t::brgemm)
    {
        // no transposition
        m_brgemm_main.generate(m_dim_sizes[m_dim_id_prim_M],
                               m_dim_sizes[m_dim_id_prim_N],
                               m_dim_sizes[m_dim_id_prim_K],
                               m_dim_sizes[m_dim_id_prim_BR],
                               0,
                               0,
                               0,
                               dtype);
        m_kernel_gemm_main = m_brgemm_main.get_kernel();
    }

*************************************
5.2 Recursive Loops over Primitives
*************************************

After generating the kernels, we needed to implement a function to execute the tensor operation.
The entry point is an ``execute`` function that takes the pointers to our matrices and passes 
them to our ``execute_iter`` function.

.. code-block:: cpp
    :caption: Entry point function: execute

    void mini_jit::TensorOperation::execute(void const* tensor_in0,
                                            void const* tensor_in1,
                                            void*       tensor_out)
    {
        if (!m_has_been_setup)
        {
            std::cerr << "TensorOperation has not been setup. Call setup() before execute()." << std::endl;
            return;
        }

        auto ptr_in0 = static_cast<char const*>(tensor_in0);
        auto ptr_in1 = static_cast<char const*>(tensor_in1);
        auto ptr_out = static_cast<char*>(tensor_out);

        execute_iter(0,
                     ptr_in0,
                     ptr_in1,
                     ptr_out,
                     true,y
                     true);
    }

The 'real' execution happens in the ``execute_iter`` function.

First, we determine the strides, retrieve the size of the current dimension and start a loop over the current dimension.

.. code-block:: cpp
    :caption: Determining the strides and looping over the current dimension size

    void mini_jit::TensorOperation::execute_iter(int64_t     id_loop,
                                                 char const* ptr_in0,
                                                 char const* ptr_in1,
                                                 char*       ptr_out,
                                                 bool        first_access,
                                                 bool        last_access)
    {
        // there is only one iteration if the dimension is the first primitive
        const int64_t l_size       = id_loop != m_id_first_primitive_loop ? m_dim_sizes[id_loop] : 1;
        const int64_t dtype_sz     = dtype_size();
        const int64_t l_stride_in0 = m_strides_in0[id_loop] * dtype_sz;
        const int64_t l_stride_in1 = m_strides_in1[id_loop] * dtype_sz;
        const int64_t l_stride_out = m_strides_out[id_loop] * dtype_sz;

        for (int64_t l_iter = 0; l_iter < l_size; l_iter++)
        {
            [See code snippets below]
        }
    }

Inside the loop, we first check if the current iteration is the first or last access to the block in the output matrix.

.. code-block:: cpp
    :caption: Determining if the current iteration is the first or last access to the block in the output matrix

    bool is_first = first_access;
    bool is_last  = last_access;
    // if the size is 1, it is always the first and last access
    if (l_size > 1 && m_dim_types[id_loop] == dim_t::k)
    {
        is_first = first_access && (l_iter == 0);
        is_last  = last_access && (l_iter == m_dim_sizes[id_loop] - 1);
    }

Next, we adjust the pointers to the correct blocks of the matrices and execute the kernels if necessary.
In the case that the current dimension, is a sequential dimension, we recursively call the ``execute_iter`` function again in order to go deeper into the loop structure.

.. code-block:: cpp
    :caption: Stride adjustment and recursive call to execute_iter

    char const* sub_ptr_in0 = ptr_in0 + l_iter * l_stride_in0;
    char const* sub_ptr_in1 = ptr_in1 + l_iter * l_stride_in1;
    char*       sub_ptr_out = ptr_out + l_iter * l_stride_out;

    // Recursive Call
    if (id_loop + 1 < m_id_first_primitive_loop)
    {
        execute_iter(id_loop + 1,
                        sub_ptr_in0,
                        sub_ptr_in1,
                        sub_ptr_out,
                        is_first,
                        is_last);
    }

However, in case the current dimension is the last sequential dimension and the next one the first primitive dimension, we need to execute the actual kernels. Depending on the ``is_first`` and ``is_last`` variables, the first and last touch kernels are also executed.

.. code-block:: cpp
    :caption: Executing the kernels

    else
    {
        if (is_first)
        {
            execute_kernel_first_touch(sub_ptr_out,
                                        m_adjusted_stride_out);
        }
        execute_kernel_main(sub_ptr_in0,
                            sub_ptr_in1,
                            sub_ptr_out,
                            m_adjusted_stride_in0,
                            m_adjusted_stride_in1,
                            m_adjusted_stride_out,
                            m_adjusted_br_size_A,
                            m_adjusted_br_size_B);

        if (is_last)
        {
            execute_kernel_last_touch(sub_ptr_out,
                                        m_adjusted_stride_out);
        }
    }

.. _5.3 Sequential Benchmarking:

*******************************
5.3 Performance Benchmarking
*******************************

To test the performance of our at runtime constructed kernels and to see if everything works seamlessly together, 
we were performing some reference benchmarks. 

For this, we were given a number of configuration parameters:

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
  
When benchmarking the configurations we achieved the following performance in ``GFLOPs``:

.. literalinclude:: ../../benchmarks/tensor_operation_benchmarks.txt
    :language: text
    :caption: ``GFLOP`` performance of the given configurations
    :dedent:

The results show that we achieve between ``71-73 GFLOPs`` for all our executions. 
These results are somewhat consistent with calling the kernels themselves independently.

.. note::
    Since the submission we made some minor changes to our implementation.
    To improve performance, we decided to enhance our ``matmul_m_n_k`` implementation. 
    Specifically, the matmul kernel now computes blocks with a size of ``16x4`` instead of ``8x4``.
    This helped us increase the results from ``71-73 GFLOPs`` to around ``90-91 GFLOPs``.

.. literalinclude:: ../../benchmarks/tensor_operation_benchmarks_2.txt
    :language: text
    :caption: ``GFLOP`` performance of the given configurations using the enhanced ``matmul`` kernel
    :dedent:

**********************************
5.4 Shared Memory Parallelization
**********************************

To enable the execution of shared loops, we needed to make a few adjustments to our ``setup`` code:

.. code-block:: cpp
    :caption: Gathering shared loop IDs and dimension sizes

    /////////////////////////////////////////////////////////////////////
    // Find SHARED dimensions in exec types
    /////////////////////////////////////////////////////////////////////
    m_shared_loop_ids.clear();
    m_shared_loop_sizes.clear();
    for (size_t i = 0; i < m_exec_types.size(); ++i)
    {
        if (m_exec_types[i] == exec_t::shared)
        {
            m_shared_loop_ids.push_back(i);
            m_shared_loop_sizes.push_back(m_dim_sizes[i]);
        }
    }

.. code-block:: cpp
    :caption: Reading the shared dimension IDs

    /////////////////////////////////////////////////////////////////////
    // Read SEQ and SHARED dimensions using dim types
    /////////////////////////////////////////////////////////////////////
    for (size_t i = 0; i < m_dim_types.size(); ++i)
    {
        if (m_exec_types[i] == exec_t::seq)
        {
            if (m_dim_types[i] == dim_t::m)
            {
                m_dim_id_seq_M = i;
            }
            else if (m_dim_types[i] == dim_t::n)
            {
                m_dim_id_seq_N = i;
            }
            else if (m_dim_types[i] == dim_t::k)
            {
                m_dim_id_seq_K = i;
            }
        }
        else if (m_exec_types[i] == exec_t::shared)
        {
            if (m_dim_types[i] == dim_t::m)
            {
                m_dim_id_sha_M = i;
                m_num_parallel_loops++;
            }
            else if (m_dim_types[i] == dim_t::n)
            {
                m_dim_id_sha_N = i;
                m_num_parallel_loops++;
            }
        }
    }

In our execute function we needed to add a check if our ``m_num_parallel_loops`` variable would be greater 
than zero. If this was the case we would then execute our ``execute_iter_parallel`` function:

.. code-block:: cpp
    :caption: Updated execute function

    void mini_jit::TensorOperation::execute(void const* tensor_in0,
                                            void const* tensor_in1,
                                            void*       tensor_out)
    {
        if (!m_has_been_setup)
        {
            std::cerr << "TensorOperation has not been setup. Call setup() before execute()." << std::endl;
            return;
        }

        auto ptr_in0 = static_cast<char const*>(tensor_in0);
        auto ptr_in1 = static_cast<char const*>(tensor_in1);
        auto ptr_out = static_cast<char*>(tensor_out);

        if (m_num_parallel_loops == 0)
        {
            // No shared loops, execute sequentially
            execute_iter(0,
                        ptr_in0,
                        ptr_in1,
                        ptr_out,
                        true,
                        true);
        }
        else
        {
            // Shared loops, execute in parallel
            execute_iter_parallel(ptr_in0,
                                ptr_in1,
                                ptr_out,
                                true,
                                true);
        }
    }

Inside the ``execute_iter_parallel``, we first multiply the shared loop sizes to get the total number of iterations. 
The idea is to get a flat iteration space that can be used to parallelize over.

.. code-block:: cpp
    :caption: multiply shared loop sizes to get the total number of iterations

    // Compute total number of iterations over shared loops
    int64_t l_size_parallel_loops = 1;
    for (auto current_loop_size : m_shared_loop_sizes)
    {
        l_size_parallel_loops *= current_loop_size;
    }

    int64_t l_first_id_loop = (m_id_first_seq_loop != -1) ? m_id_first_seq_loop : m_id_first_primitive_loop;

We unflatten the OpenMP iteration index ``l_it_all`` into a set of loop indices, one for each shared loop dimension. 
These indices are then used to compute the offsets for the ``in0``, ``in1``, and ``out`` tensors: 

.. code-block:: cpp
    :caption: Calculating the tensor offsets

    #pragma omp parallel for
    for (int64_t l_it_all = 0; l_it_all < l_size_parallel_loops; ++l_it_all)
    {
        // Unflatten l_it_all into loop indices
        int64_t              remainder = l_it_all;
        std::vector<int64_t> loop_indices(m_shared_loop_ids.size());

        for (int64_t i = m_shared_loop_ids.size() - 1; i >= 0; --i)
        {
            loop_indices[i] = remainder % m_shared_loop_sizes[i];
            remainder /= m_shared_loop_sizes[i];
        }

        // Compute pointer offsets using strides and loop indices
        char const* sub_ptr_in0 = ptr_in0;
        char const* sub_ptr_in1 = ptr_in1;
        char*       sub_ptr_out = ptr_out;

        const int64_t dtype_sz = dtype_size();
        for (size_t i = 0; i < m_shared_loop_ids.size(); ++i)
        {
            const int64_t dim_id = m_shared_loop_ids[i];
            const int64_t idx    = loop_indices[i];

            sub_ptr_in0 += idx * m_strides_in0[dim_id] * dtype_sz;
            sub_ptr_in1 += idx * m_strides_in1[dim_id] * dtype_sz;
            sub_ptr_out += idx * m_strides_out[dim_id] * dtype_sz;
        }

Here we are calculating the offset for the current thread. 
Every shared loop contributes to the calculation with its corresponding stride. 

Lastly, we call our ``execute_iter`` function. 

.. code-block:: cpp
    :caption: Executing the remaining loops with execute_iter

    // Call remaining loops
    execute_iter(l_first_id_loop,
                    sub_ptr_in0,
                    sub_ptr_in1,
                    sub_ptr_out,
                    first_access,
                    last_access);

After having implemented the shared loop parallelization, we benchmarked the configurations from the :ref:`sequential execution<5.3 Sequential Benchmarking>` task again.
To enable multithreading, we called the executable with ``OMP_NUM_THREADS=4``.

.. literalinclude:: ../../benchmarks/shared_tensor_operation_benchmarks.txt
    :language: text
    :lineno-match:
    :caption: ``GFLOP`` performance for ``4 shared`` loop execution
    :dedent:

With the parallelization we achieved about ``360 - 390 GFLOPs``. 

**********************************
5.5 Optimization Passes
**********************************

To lay the foundation for our Optimizer, we first decided to implement a Dimension ``struct`` that wraps all information on a dimension.

.. code-block:: cpp
    :caption: Dimension struct

    /**
    * @brief The Dimension struct represents a dimension in a tensor operation.
    * It contains information about the type of dimension (M, N, K), execution type (Prim, Seq, Shared),
    * size, and strides for the input and output tensors.
    */
    struct Dimension
    {
        //! Type of the dimension (M, N, K)
        dim_t type = dim_t::m;
        //! Execution type (Prim, Seq, Shared, ...)
        exec_t exec_type = exec_t::undefined;
        //! Dimension size
        int64_t size = 0;
        //! Stride in the first input tensor
        int64_t stride_in0 = 0;
        //! Stride in the second input tensor
        int64_t stride_in1 = 0;
        //! Stride in the output tensor
        int64_t stride_out = 0;

        /**
            * @brief Construct a new Dimension object.
            *
            * @param type Type of the dimension (M, N, K).
            * @param exec_type Execution type (Prim, Seq, Shared, ...).
            * @param size Size of the dimension.
            * @param stride_in0 Stride in the first input tensor.
            * @param stride_in1 Stride in the second input tensor.
            * @param stride_out Stride in the output tensor.
            */
        Dimension(dim_t   type,
                    exec_t  exec_type,
                    int64_t size,
                    int64_t stride_in0,
                    int64_t stride_in1,
                    int64_t stride_out)
            : type(type),
                exec_type(exec_type),
                size(size),
                stride_in0(stride_in0),
                stride_in1(stride_in1),
                stride_out(stride_out)
        {
            if (size <= 0)
            {
                throw std::invalid_argument("Dimension size needs to be greater than 0");
            }
        }
    };

This method seemed simpler to us in comparison to accessing and reordering all information of a Dimension using multiple vectors that only store a single property each.

5.5.1 Primitive Identification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first optimization pass which we implemented was the primitive identification.
This optimization is useful in cases where only sequential and shared loops are given.

First we need to check whether we are optimizing a unary operation (identity, permutation), a binary operation (Add, Sub, Mul; See :ref:`binary-primitives`) or a ternary operation (GEMM, BRGEMM).

.. code-block:: cpp
    :caption: Checking for the existence of C and K dimensions

    auto l_has_c_dim = std::any_of(dimensions.begin(), dimensions.end(), [](const mini_jit::ir::Dimension& dim)
                                   { return dim.type == dim_t::c; });

    auto l_has_k_dim = std::any_of(dimensions.begin(), dimensions.end(), [](const mini_jit::ir::Dimension& dim)
                                   { return dim.type == dim_t::k; });

If we find a ``C`` dimension, we assume that the current operation is a unary operation.

If ``l_has_k_dim`` is true, the operation to be optimized is a ternary operation and if it is false, it is a binary operation. Sine the following code is rather complex, we decided to briefly describe it using words instead.

5.5.1.1 Unary Operation
"""""""""""""""""""""""""""""""""

In this case, we first check if all dimensions are of type ``C``.
Only then it is a valid unary operation and we can start identifying the primitive dimensions.

First, we identify the primitive ``M`` dimension as the dimension which has unit stride in the input tensor.
If that dimension does not have unit stride in the output tensor, we assume that the unary operation uses transposition.

In case of transposition, the primitive ``N`` dimension is the one with unit stride in the output tensor.
In case there is no transposition, we choose the dimension with the smallest stride in the input tensor.
That excludes the previously identified ``M`` dimension.

Lastly, we set all remaining dimensions to ``sequential``.

5.5.1.2 Binary Operation
"""""""""""""""""""""""""""""""""

In binary operations, the primitive ``M`` dimension has unit stride in all tensors and can thus be identified easily.
For the primitive ``N`` dimension, we simply choose the dimension of type ``N`` that has the smallest strides.

5.5.1.3 Ternary Operation
"""""""""""""""""""""""""""""""""

We first check for a second ``K`` dimension, which has no unit stride in the second input tensor and does not appear in the output tensor.
If such a ``K`` dimension exists, we select it as the batch-reduce dimension and conclude that the current operation is a BRGEMM.
If no such ``K`` dimension exists, we are handling a GEMM.

Next, we identify the primitive ``M`` dimension by checking for an ``M`` dimension with unit stride in the first input and output tensor.

The primary ``N`` dimension is then found by choosing the dimension with the smallest stride which appears only in the second input tensor and in the output tensor.

Lastly, we identify the primitive ``K`` dimension by searching for a ``K`` dimension that has unit stride in the second input tensor and does not appear in the output tensor.

5.5.2 Dimension Splitting
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For our second optimization pass we decided to look at the dimension sizes of our loops. 
We introduced a ``max_kernel_size`` parameter, which specifies the maximum allowed size for a dimension.
If a dimension with a size larger than the maximum is found, the dimension splitter will try to split it into new dimensions with optimized sizes. 
The entry point for this optimization is the ``splitDimensions`` function:

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

For each dimension, it finds the bets split for our kernels if the dimension size is too large and creates a new dimension. 
The size of the original dimension is updated to ``l_size_dim_1``, and it will be smaller than or equal to ``max_kernel_size``. However, the new dimension ``l_dim_new`` might still have a larger dimension size than ``max_kernel_size``, which is why it is inserted at the end of the dimensions vector, where it will be checked for a possible split in a later iteration.

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

Our third optimization pass was to make all loops that were not a ``prim`` dimension and of the dimension type ``M`` or ``N`` a ``shared`` loop.
For that we initially check how many loops are already of dimension type ``shared``:

.. code-block:: cpp
    :caption: Count already existing shared iterations

    int64_t l_num_threads = 1;

    // Count the number of existing iterations for shared loops
    for (size_t i = 0; i < dimensions.size(); i++)
    {
        if (dimensions[i].exec_type == exec_t::shared)
        {
            // increase thread number for each existing shared dimension
            l_num_threads *= dimensions[i].size;
        }
    }

For the case that we already have a high number of ``shared`` loops we do not create any more and simply return. 
Otherwise we check the ``seq`` dimensions for potential candidates:

.. code-block:: cpp
    
    if (l_num_threads >= thread_target)
    {
        // make sure that the shared loops are at the front
        std::stable_partition(dimensions.begin(), dimensions.end(), [](const mini_jit::ir::Dimension& dim)
                              { return dim.exec_type == exec_t::shared; });
        // no need to create more shared loops
        return;
    }

    // Creation of new shared loops:
    for (size_t i = 0; i < dimensions.size(); i++)
    {
        // if the dimension can be set to shared and we did not reach the target number of threads yet
        // we set the dimension to shared
        // also dont parallelize the k dimension (see class slides)
        if ((dimensions[i].exec_type == exec_t::seq || dimensions[i].exec_type == exec_t::undefined) &&
            dimensions[i].type != dim_t::k &&
            l_num_threads * dimensions[i].size <= thread_target)
        {
            dimensions[i].exec_type = exec_t::shared;
            l_num_threads *= dimensions[i].size;
        }
    }

As a last step we move all our ``shared`` loops to the front of the order:

.. code-block:: cpp

    // Move all shared loops to the front
    std::stable_partition(dimensions.begin(), dimensions.end(), [](const mini_jit::ir::Dimension& dim)
                          { return dim.exec_type == exec_t::shared; });

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

We performed benchmarks using different parameters to see what effect they have on the performance.
We obtained the following results:

.. literalinclude:: ../../benchmarks/optimized_tensor_operation_benchmarks.txt
    :language: text
    :lineno-match:
    :caption: ``GFLOP`` performance for sample configurations
    :dedent:

Depending on the selected dimensions our results varied massively. The highest performance we achieved was around ``350 GFLOPs``. 

**********************************
5.6 Unary Operations
**********************************

5.6.1 Backend Extension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this task, we were supposed to add support for unary operations, such as permuting a tensor's dimensions, to our tensor operation backend.
Furthermore, we had to implement primitive identification and shared memory parallelization optimization passes for these unary primitives.

Instead of separating the documentation on unary, binary and ternary operations, we decided to merge them.
This means that the code for handling unary operations has already been shown and explained in the previous section(s).

5.6.2 Reference Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For our reference implementation, we used an example with 4 dimensions ``trus`` in which we reorder the dimensions to ``turs``.

.. code-block:: cpp
    :caption: Initialization of the tensor sizes

    const mini_jit::ptype_t first_touch_type = mini_jit::ptype_t::none;
    const mini_jit::ptype_t main_type        = mini_jit::ptype_t::identity;
    const mini_jit::ptype_t last_touch_type  = mini_jit::ptype_t::none;

    const int T = GENERATE(3, 4, 7);
    const int R = GENERATE(3, 4, 7);
    const int U = GENERATE(3, 4, 7);
    const int S = GENERATE(3, 4, 7);

    const int SIZE = T * R * U * S;

    float* A          = new float[SIZE];
    float* C          = new float[SIZE];
    float* C_expected = new float[SIZE];

.. code-block:: cpp
    :caption: Filling the tensors with values

    std::random_device                    rd;
    std::mt19937                          gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (int i = 0; i < SIZE; ++i)
    {
        A[i] = dist(gen);
    }

    // Compute C_expected
    for (int t = 0; t < T; ++t)
    {
        for (int r = 0; r < R; ++r)
        {
            for (int u = 0; u < U; ++u)
            {
                for (int s = 0; s < S; ++s)
                {
                    // Calculate index in output format (t,r,u,s) using strides_out
                    int l_idx_c_exp = t * (U * R * S) + r * S + u * (R * S) + s;
                    // Calculate index in input format (t,u,r,s) using strides_in0
                    int l_idx_a             = t * (R * U * S) + u * S + r * (U * S) + s;
                    C_expected[l_idx_c_exp] = A[l_idx_a];
                }
            }
        }
    }

Then we prepared the execution by setting all arguments accordingly:

.. code-block:: cpp
    :caption: Prepare arguments for execution

    std::vector<mini_jit::dim_t> dim_types = {
        mini_jit::dim_t::c, // t
        mini_jit::dim_t::c, // r
        mini_jit::dim_t::c, // u
        mini_jit::dim_t::c  // s
    };

    std::vector<mini_jit::exec_t> exec_types = {
        mini_jit::exec_t::seq,  // t
        mini_jit::exec_t::seq,  // r
        mini_jit::exec_t::prim, // u
        mini_jit::exec_t::prim  // s
    };

    std::vector<int64_t> dim_sizes = {
        T, R, U, S};

    std::vector<int64_t> strides_in0 = {
        R * U * S, // t
        U * S,     // r
        S,         // u
        1          // s
    };

    std::vector<int64_t> strides_in1 = {0, 0, 0, 0};

    std::vector<int64_t> strides_out = {
        U * R * S, // t
        S,         // r
        R * S,     // u
        1          // s
    };

This code can be found in the ``TensorOperation.test.cpp`` file. Running the test resulted in a successful pass.
