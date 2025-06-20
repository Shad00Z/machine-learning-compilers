#####################
4. Code Generation
#####################

**********************
4.1 BRGEMM Primitive
**********************

In this section, we explain how we implemented the BRGEMM primitive for our machine learning compiler.

.. _4-1-1-microkernel:

4.1.1 Microkernel
===================

Before we began implementing the planned batch-reduce general matrix-matrix multiplication, we first needed to take a look at how we could multiply matrices using C++ code generator for assembly. To simplify things further, we first put our focus on generating the fixed size microkernels which we had already implemented in A64 ARM assembly (see: :ref:`3-neon`).

4.1.1.1 Instruction Generation
----------------------------------

The first step to generating a kernel is to create C++ mappings to the A64 ARM assembly instructions we need. That is, for each instruction we use in our assembly kernel, we require a C++ function that can generate the instruction for us, with support for various input parameters. The output of such a function is a ``uint32_t`` value, representing the 32-bits of the assembly instruction.

.. note::

    Instead of working with binary numbers directly, we decided to use hexadecimal representation.

While we won't show all instructions that we had to implement, here are some examples:

**LDR instruction (unsigned offset)**

.. code:: cpp

    /**
    * @brief Generates a base LDR (12-bit immediate) instruction using unsigned offset encoding.
    *
    * @param reg_dest destination register.
    * @param reg_src source register (base address).
    * @param imm12 12-bit immediate value.
    */
    constexpr uint32_t ldr(gpr_t reg_dest,
                        gpr_t reg_src,
                        uint32_t imm)
    {
        uint32_t l_ins = 0xB9400000;

        // set size
        uint32_t l_sf = reg_dest & 0x20;
        l_ins |= l_sf << 25; // set bit 30

        // set destination register id
        l_ins |= (reg_dest & 0x1f);

        // set first source register id
        l_ins |= (reg_src & 0x1f) << 5;

        // check if immediate can be encoded
        uint32_t scale = (l_sf) ? 8 : 4;
        if (imm % scale != 0)
        {
            throw std::invalid_argument("Immediate offset must be a multiple of 4 (32-bit) or 8 (64-bit)");
        }

        // scale the immediate for encoding (right-shift)
        uint32_t scaleShift = (l_sf) ? 3 : 2; // 64-bit? then /8 (>>3); else /4 (>>2)
        uint32_t l_imm = (imm >> scaleShift) & 0xFFF;

        // set 12 bit immediate value
        l_ins |= l_imm << 10;
        return l_ins;
    }

**ADD instruction (immediate)**

.. code:: cpp

    /**
    * @brief Generates an ADD (immediate) instruction.
    *
    * @param reg_dest destination register.
    * @param reg_src1 source register.
    * @param imm12 12-bit immediate value.
    * @param shift shift value.
    *
    * @return instruction.
    */
    constexpr uint32_t add(gpr_t reg_dest,
                           gpr_t reg_src,
                           uint32_t imm12,
                           uint32_t shift)
    {
        uint32_t l_ins = 0x11000000;

        // set size
        uint32_t l_sf = reg_dest & 0x20;
        l_ins |= l_sf << 26; // set bit 31

        // set destination register id
        uint32_t l_reg_id = reg_dest & 0x1f;
        l_ins |= l_reg_id;

        // set first source register id
        l_reg_id = reg_src & 0x1f;
        l_ins |= l_reg_id << 5;

        // set immediate value
        uint32_t l_imm = imm12 & 0xfff;
        l_ins |= l_imm << 10;

        // set shift value
        uint32_t l_shift = shift & 0x1;
        l_ins |= l_shift << 22;

        return l_ins;
    }

For more information on the instructions, please refer to :ref:`API: mini_jit:instructions <api_mini_jit_instructions>`.

4.1.1.2 Microkernel Generation
------------------------------------

Having implemented all necessary functions for generating the instructions, we then turned our attention to the microkernel generation. Here, the first kernel we tackled was the ``matmul_16_6_1`` kernel. The process here was to copy the assembly code line by line and replace all instructions with our C++ bindings. A part of the result can be seen in the following code snippet:

**Loading of inputs section of the matmul_16_6_1 kernel using C++ JIT code generation**

.. code:: cpp

    // Load Matrix A
    kernel.add_instr( simd_fp::ldp(simd_fp_t::v0, simd_fp_t::v1, gpr_t::x0, 0, neon_size_spec_t::q) );
    kernel.add_instr( simd_fp::ldp(simd_fp_t::v2, simd_fp_t::v3, gpr_t::x0, 32, neon_size_spec_t::q) );

    // Load Matrix C
    kernel.add_instr( base::mov(gpr_t::x7, gpr_t::x2) );
    kernel.add_instr( simd_fp::ldp(simd_fp_t::v4, simd_fp_t::v5, gpr_t::x7, 0, neon_size_spec_t::q) );
    kernel.add_instr( simd_fp::ldp(simd_fp_t::v6, simd_fp_t::v7, gpr_t::x7, 32, neon_size_spec_t::q) );
    kernel.add_instr( base::add(gpr_t::x7, gpr_t::x7, gpr_t::x5, 0, 0) );

    kernel.add_instr( simd_fp::ldp(simd_fp_t::v8, simd_fp_t::v9, gpr_t::x7, 0, neon_size_spec_t::q) );
    kernel.add_instr( simd_fp::ldp(simd_fp_t::v10, simd_fp_t::v11, gpr_t::x7, 32, neon_size_spec_t::q) );
    kernel.add_instr( base::add(gpr_t::x7, gpr_t::x7, gpr_t::x5, 0, 0) );

    kernel.add_instr( simd_fp::ldp(simd_fp_t::v12, simd_fp_t::v13, gpr_t::x7, 0, neon_size_spec_t::q) );
    kernel.add_instr( simd_fp::ldp(simd_fp_t::v14, simd_fp_t::v15, gpr_t::x7, 32, neon_size_spec_t::q) );
    kernel.add_instr( base::add(gpr_t::x7, gpr_t::x7, gpr_t::x5, 0, 0) );

    kernel.add_instr( simd_fp::ldp(simd_fp_t::v16, simd_fp_t::v17, gpr_t::x7, 0, neon_size_spec_t::q) );
    kernel.add_instr( simd_fp::ldp(simd_fp_t::v18, simd_fp_t::v19, gpr_t::x7, 32, neon_size_spec_t::q) );
    kernel.add_instr( base::add(gpr_t::x7, gpr_t::x7, gpr_t::x5, 0, 0) );

    kernel.add_instr( simd_fp::ldp(simd_fp_t::v20, simd_fp_t::v21, gpr_t::x7, 0, neon_size_spec_t::q) );
    kernel.add_instr( simd_fp::ldp(simd_fp_t::v22, simd_fp_t::v23, gpr_t::x7, 32, neon_size_spec_t::q) );
    kernel.add_instr( base::add(gpr_t::x7, gpr_t::x7, gpr_t::x5, 0, 0) );

    kernel.add_instr( simd_fp::ldp(simd_fp_t::v24, simd_fp_t::v25, gpr_t::x7, 0, neon_size_spec_t::q) );
    kernel.add_instr( simd_fp::ldp(simd_fp_t::v26, simd_fp_t::v27, gpr_t::x7, 32, neon_size_spec_t::q) );

**FMLA section of the matmul_16_6_1 kernel using C++ JIT code generation**

.. code:: cpp

    // Load Column of Matrix B
    kernel.add_instr( base::mov(gpr_t::x6, gpr_t::x1) );
    kernel.add_instr( simd_fp::ldr(simd_fp_t::v28, gpr_t::x6, 0, neon_size_spec_t::s) );
    kernel.add_instr( base::add(gpr_t::x6, gpr_t::x6, gpr_t::x4, 0, 0) );

    // 1st Multiplication
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v4, simd_fp_t::v0, simd_fp_t::v28, arr_spec_t::s4) );
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v5, simd_fp_t::v1, simd_fp_t::v28, arr_spec_t::s4) );
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v6, simd_fp_t::v2, simd_fp_t::v28, arr_spec_t::s4) );
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v7, simd_fp_t::v3, simd_fp_t::v28, arr_spec_t::s4) );

    // Load Column of Matrix B
    kernel.add_instr( simd_fp::ldr(simd_fp_t::v29, gpr_t::x6, 0, neon_size_spec_t::s) );
    kernel.add_instr( base::add(gpr_t::x6, gpr_t::x6, gpr_t::x4, 0, 0) );

    // 2nd Multiplication
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v8, simd_fp_t::v0, simd_fp_t::v29, arr_spec_t::s4) );
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v9, simd_fp_t::v1, simd_fp_t::v29, arr_spec_t::s4) );
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v10, simd_fp_t::v2, simd_fp_t::v29, arr_spec_t::s4) );
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v11, simd_fp_t::v3, simd_fp_t::v29, arr_spec_t::s4) );

    // Load Column of Matrix B
    kernel.add_instr( simd_fp::ldr(simd_fp_t::v30, gpr_t::x6, 0, neon_size_spec_t::s) );
    kernel.add_instr( base::add(gpr_t::x6, gpr_t::x6, gpr_t::x4, 0, 0) );

    // 3rd Multiplication
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v12, simd_fp_t::v0, simd_fp_t::v30, arr_spec_t::s4) );
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v13, simd_fp_t::v1, simd_fp_t::v30, arr_spec_t::s4) );
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v14, simd_fp_t::v2, simd_fp_t::v30, arr_spec_t::s4) );
    kernel.add_instr( simd_fp::fmlaElem(simd_fp_t::v15, simd_fp_t::v3, simd_fp_t::v30, arr_spec_t::s4) );

.. note::

    All instructions are added to a ``kernel`` object. This code structure was already given to us, so we will not explain it in detail here. Basically, the ``kernel`` object is responsible for holding all instructions in a buffer, allocating the necessary memory, writing the instructions to the memory and then making the allocated memory executable. The ``kernel`` object is also able to later release the allocated memory again.

Towards the goal of implementing a ``GEMM`` kernel, we now had to start supporting arbitrary dimension sizes. We decided to start implementing a loop over the ``k`` dimension, thus extending the ``matmul_16_6_1`` kernel to ``matmul_16_6_k``.

**K-Loop section of the matmul_16_6_k kernel using C++ JIT code generation**

.. code:: cpp

    // Setup for Loop
    kernel.add_instr( base::mov(gpr_t::x6, k) ); // K loop counter
    kernel.add_instr( base::mov(gpr_t::x7, gpr_t::x0) ); // Matrix A pointer
    kernel.add_instr( base::mov(gpr_t::x8, gpr_t::x1) ); // Matrix B pointer
    kernel.add_instr( base::mov(gpr_t::x9, 0) ); // Row index for Matrix B

    [matmul_16_6_1 kernel]

    // Decrement K
    // move to next column of A
    kernel.add_instr( base::add(gpr_t::x7, gpr_t::x7, gpr_t::x3, 0, 0) ); 
    // move to next row of B
    kernel.add_instr( base::mov(gpr_t::x8, gpr_t::x1) );
    kernel.add_instr( base::add(gpr_t::x9, gpr_t::x9, 4, 0) );
    kernel.add_instr( base::add(gpr_t::x8, gpr_t::x8, gpr_t::x9, 0, 0) );
    // edit K and jump to start of the kernel
    kernel.add_instr( base::sub(gpr_t::x6, gpr_t::x6, 1, 0) );
    kernel.add_instr( base::cbnz(gpr_t::x6, -168) );

4.1.1.3 Microkernel Benchmark
------------------------------------

The last step of the task was to run benchmarks. We obtained the following results:

.. code:: text

    Benchmarking Matmul_16_6_1 throughput ...
    -----------------------------------------------
    Measuring throughput for Instruction
    Total time (s):   1.19943
    Instructions per Second:   2.40114e+10
    Estimated GFLOPS:   24.0114 GFLOPS/sec
    -----------------------------------------------

    Benchmarking Matmul_16_6_64 throughput ...
    -----------------------------------------------
    Measuring throughput for Instruction
    Total time (s):   1.82951
    Instructions per Second:   1.34331e+11
    Estimated GFLOPS:   134.331 GFLOPS/sec
    -----------------------------------------------

.. _4.1.2 GEMM:

4.1.2 GEMM
==================

4.1.2.1 Implementation of a GEMM kernel
----------------------------------------

This section extends the in :ref:`4-1-1-microkernel` implemented kernel to a more general GEMM kernel. It should be able to compute C+=AB for arbitrary A, B and C matrices in the range of 1≤M≤1024, 1≤N≤1024, and 1≤K≤2048.

At first, we had to decide on how to block the matrices. In the M dimension, we decided to use a block size of 16 and in the N dimension we decided to use a block size of 4. The larger we keep the block size, the more efficiently we can use loads, stores and FMLA instructions. However, the issue with large block sizes is that we need to write a lot of specialized kernels for all M and N dimensions smaller or equal to the block size. If the input parameters are not multiples of the block size, we need to write additional code to handle the remaining elements. 

For a block size of M = 8, we already wrote a kernel in pure assembly, see :ref:`generic-kernel`. Using this generic kernel as a starting point, we reduced the N dimension from 6 to 4. Our reasoning was that we wanted to reduce the number of specialized kernels we need to write. Additionally, we assumed that most numbers commonly used in practice are multiples of 4 instead of 6, thus not depending on the specialized kernels. Nevertheless, we made the decision to increase M from 8 to 16 to increase performance. With this change, we implemented the ``matmul_m_4_k`` kernel, which can compute C+=AB for any matrices where M and K can be chosen freely, but N is fixed to 4.

The kernel first computes the number of blocks in the M dimension and the remaining elements. 

**matmul_m_4_k: Computing the number of blocks in the M dimension**

.. code:: cpp

    int mLoopIterations = m / 16;
    int mLoopRemainder = m % 16;

Using these numbers, we can call the specialized kernels:

**matmul_m_4_k: Calling specialized kernels for different M dimensions**

.. code:: cpp

    if (mLoopIterations > 0)
    {
        mini_jit::kernels::matmul::subkernels::internal::generateM16N4Loop(kernel, mLoopIterations, k);
    }

    if (mLoopRemainder > 0)
    {
        // set up k loop counter
        kernel.add_instr(base::mov(gpr_t::x14, k));
        // save base matrix pointers
        kernel.add_instr(base::mov(gpr_t::x15, gpr_t::x8)); // A
        kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x9)); // B
        kernel.add_instr(base::mov(gpr_t::x17, 0));         // row count B

        switch (mLoopRemainder)
        {
        case 1:
            mini_jit::kernels::matmul::subkernels::internal::generateM1N4Loop(kernel);
            break;
        case 2:
            mini_jit::kernels::matmul::subkernels::internal::generateM2N4Loop(kernel);
            break;
        case 3:
            mini_jit::kernels::matmul::subkernels::internal::generateM3N4Loop(kernel);
            break;
        case 4:
            mini_jit::kernels::matmul::subkernels::internal::generateM4N4Loop(kernel);
            break;
        case 5:
            mini_jit::kernels::matmul::subkernels::internal::generateM5N4Loop(kernel);
            break;
        case 6:
            mini_jit::kernels::matmul::subkernels::internal::generateM6N4Loop(kernel);
            break;
        case 7:
            mini_jit::kernels::matmul::subkernels::internal::generateM7N4Loop(kernel);
            break;
        case 8:
            mini_jit::kernels::matmul::subkernels::internal::generateM8N4Loop(kernel);
            break;
        case 9:
            mini_jit::kernels::matmul::subkernels::internal::generateM9N4Loop(kernel);
            break;
        case 10:
            mini_jit::kernels::matmul::subkernels::internal::generateM10N4Loop(kernel);
            break;
        case 11:
            mini_jit::kernels::matmul::subkernels::internal::generateM11N4Loop(kernel);
            break;
        case 12:
            mini_jit::kernels::matmul::subkernels::internal::generateM12N4Loop(kernel);
            break;
        case 13:
            mini_jit::kernels::matmul::subkernels::internal::generateM13N4Loop(kernel);
            break;
        case 14:
            mini_jit::kernels::matmul::subkernels::internal::generateM14N4Loop(kernel);
            break;
        case 15:
            mini_jit::kernels::matmul::subkernels::internal::generateM15N4Loop(kernel);
            break;
        default:
            break;
        }
    }

But what does such a specialized kernel look like? For the most part, they are similar to the microkernels we implemented before. The only difference is that we need to adjust the loads, stores and FMLA instructions for fixed M dimensions. For example in the case of M = 3:

**matmul_m_4_k: Loading a column of C with M = 3**

.. code:: cpp

    // first column
    kernel.add_instr(base::mov(gpr_t::x24, gpr_t::x12));
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v0, gpr_t::x24, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v1, gpr_t::x24, 0, neon_size_spec_t::s));

While we can simply load a double word when M = 2 or even a quad word when M = 4, we need to divide our loads into two parts when M = 3. First, we load a double word and then the remaining single word. The same applies to the stores:

**matmul_m_4_k: Storing a column of C with M = 3**

.. code:: cpp

    // first column
    kernel.add_instr(base::mov(gpr_t::x24, gpr_t::x12));
    kernel.add_instr(simd_fp::strPost(simd_fp_t::v0, gpr_t::x24, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::str(simd_fp_t::v1, gpr_t::x24, 0, neon_size_spec_t::s));

The FMLA instructions are also adjusted to the M dimension. For example, when M = 3, we need to use two FMLA instructions to compute the result:

**matmul_m_4_k: FMLA instructions with M = 3**

.. code:: cpp

    // B: COLUMN 0
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v0, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s2));
    kernel.add_instr(simd_fp::fmadd(simd_fp_t::v1, simd_fp_t::v25, simd_fp_t::v29, simd_fp_t::v1, neon_size_spec_t::s));

While one could use an ``fmla`` instruction and zero padding, we decided to use one ``fmla`` instruction for the first two elements and one ``fmadd`` instruction for the last element. We did not evaluate any performance differences between the two approaches, but chose the second one because to us it seemed more readable and easier to understand. The other specialized kernels for M = 1, 2, 4, 5, 6 and 7 are implemented similarly.

Having implemented the ``matmul_m_4_k`` kernel, we can now turn our attention towards the ``matmul_m_n_k`` kernel. Since we decided to block N by 4, we can use the same approach as before. We first compute the number of blocks in the N dimension and the remaining elements.

**matmul_m_n_k: Computing the number of blocks in the N dimension**

.. code::

    int nLoopIterations = n / 4;
    int nLoopRemainder = n % 4;

``nLoopRemainder`` can take any value between 0 and 3, which means that additionally to the ``matmul_m_4_k`` kernel where ``nLoopRemainder`` is 0, we need to implement specialized kernels for ``nLoopRemainder`` = 1, 2 and 3. The specialized kernels are basically the same as the ``matmul_m_4_k`` kernel, but we simply removed some of the loads, stores and FMLA instructions. For the more curious reader, we recommend viewing :ref:`API: mini_jit:kernels <api_mini_jit_kernels>`.

For the whole N loop, we use switch statements to call the specialized kernels. The final implementation looks like this:

**matmul_m_n_k: Calling kernels for different N**

.. code:: cpp

    if (nLoopIterations > 0)
    {
        // n_loop:
        kernel.add_label("n_loop");

        // Save base matrix pointers
        kernel.add_instr(base::mov(gpr_t::x8, gpr_t::x0));   // A
        kernel.add_instr(base::mov(gpr_t::x9, gpr_t::x20));  // B
        kernel.add_instr(base::mov(gpr_t::x10, gpr_t::x21)); // C

        if (mLoopIterations > 0)
        {
            internal_subkernels::generateM16N4Loop(kernel, mLoopIterations, k);
        }

        if (mLoopRemainder > 0)
        {
            // set up k loop counter
            kernel.add_instr(base::mov(gpr_t::x14, k));
            // save base matrix pointers
            kernel.add_instr(base::mov(gpr_t::x15, gpr_t::x8)); // A
            kernel.add_instr(base::mov(gpr_t::x16, gpr_t::x9)); // B
            kernel.add_instr(base::mov(gpr_t::x17, 0));         // row count B

            switch (mLoopRemainder)
            {
            case 1:
                internal_subkernels::generateM1N4Loop(kernel);
                break;
            case 2:
                internal_subkernels::generateM2N4Loop(kernel);
                break;
            case 3:
                internal_subkernels::generateM3N4Loop(kernel);
                break;
            case 4:
                internal_subkernels::generateM4N4Loop(kernel);
                break;
            case 5:
                internal_subkernels::generateM5N4Loop(kernel);
                break;
            case 6:
                internal_subkernels::generateM6N4Loop(kernel);
                break;
            case 7:
                internal_subkernels::generateM7N4Loop(kernel);
                break;
            case 8:
                internal_subkernels::generateM8N4Loop(kernel);
                break;
            case 9:
                internal_subkernels::generateM9N4Loop(kernel);
                break;
            case 10:
                internal_subkernels::generateM10N4Loop(kernel);
                break;
            case 11:
                internal_subkernels::generateM11N4Loop(kernel);
                break;
            case 12:
                internal_subkernels::generateM12N4Loop(kernel);
                break;
            case 13:
                internal_subkernels::generateM13N4Loop(kernel);
                break;
            case 14:
                internal_subkernels::generateM14N4Loop(kernel);
                break;
            case 15:
                internal_subkernels::generateM15N4Loop(kernel);
                break;
            default:
                break;
            }
        }

        // increase B and C pointers for next block
        // (jump 4 columns) 4*x4, 4*x5
        kernel.add_instr(base::add(gpr_t::x20, gpr_t::x20, gpr_t::x22, 0, 0));
        kernel.add_instr(base::add(gpr_t::x21, gpr_t::x21, gpr_t::x23, 0, 0));
        // decrement n loop counter
        kernel.add_instr(base::sub(gpr_t::x19, gpr_t::x19, 1, 0));

        // check if loop counter is zero
        int l_nLoopInstrCount = kernel.getInstrCountFromLabel("n_loop");
        kernel.add_instr(base::cbnz(gpr_t::x19, -l_nLoopInstrCount * 4));
        // END N LOOP
    }

    if (nLoopRemainder > 0)
    {
        // Save base matrix pointers
        kernel.add_instr(base::mov(gpr_t::x8, gpr_t::x0));   // A
        kernel.add_instr(base::mov(gpr_t::x9, gpr_t::x20));  // B
        kernel.add_instr(base::mov(gpr_t::x10, gpr_t::x21)); // C

        switch (nLoopRemainder)
        {
        case 1:
            mini_jit::kernels::matmul::internal::generateN1Loop(kernel, mLoopIterations, mLoopRemainder, k);
            break;
        case 2:
            mini_jit::kernels::matmul::internal::generateN2Loop(kernel, mLoopIterations, mLoopRemainder, k);
            break;
        case 3:
            mini_jit::kernels::matmul::internal::generateN3Loop(kernel, mLoopIterations, mLoopRemainder, k);
            break;
        default:
            break;
        }
    }

The full code is available in the file ``matmul_m_n_k.cpp``.

4.1.2.2 Calling the GEMM kernel
----------------------------------------

Having implemented the code for the ``matmul_m_n_k``, we now had to find a way to call it. For this, we use a ``Brgemm`` class that contains an ``execute`` function. Since we used the same function for calling our ``matmul_br_m_n_k`` BRGEMM kernel, which we will explain in the following chapter, please refer to X which will explain the ``Brgemm`` class in greater detail.

4.1.2.3 Verification of the GEMM kernel with lda=M, ldb=K, ldc=M
-------------------------------------------------------------------

This task requires us to verify the correctness of our ``matmul_m_n_k`` kernel by comparing to a reference implementation for 1≤M≤64, 1≤N≤64, K∈[1,16,32,64,128], and lda=M, ldb=K, ldc=M.
We realized this verification using a ``Catch2`` unit test:

.. code:: cpp

    TEST_CASE("Reference test for matmul kernel with variable M, N, K", "[matmul][parameterized]")
    {
        const int M = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32);
        const int N = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32);
        const int K = GENERATE(1, 16, 32, 64, 128);

        float *A = new float[M * K];
        float *B = new float[K * N];
        float *C = new float[M * N];
        float *C_expected = new float[M * N];

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-0.5f, 100.0f);

        for (int i = 0; i < M * K; ++i)
        {
            A[i] = dist(gen);
        }

        for (int i = 0; i < K * N; ++i)
        {
            B[i] = dist(gen);
        }

        for (int i = 0; i < M * N; ++i)
        {
            C[i] = C_expected[i] = dist(gen);
        }

        // Reference GEMM calculation
        for (int col = 0; col < N; ++col)
        {
            for (int row = 0; row < M; ++row)
            {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k)
                {
                    sum += A[row + k * M] * B[k + col * K];
                }
                C_expected[row + col * M] += sum;
            }
        }

        mini_jit::Kernel l_kernel;
        mini_jit::kernels::matmul::matmul_m_n_k(l_kernel, M, N, K);
        mini_jit::Brgemm::kernel_t l_kernel_t = reinterpret_cast<mini_jit::Brgemm::kernel_t>(const_cast<void *>(l_kernel.get_kernel()));
        l_kernel_t(A, B, C, M, K, M, 0, 0);

        for (int i = 0; i < M * N; ++i)
        {
            REQUIRE(C[i] == Approx(C_expected[i]).margin(FLOAT_ERROR_MARGIN));
        }

        delete[] A;
        delete[] B;
        delete[] C;
        delete[] C_expected;
    }

The M and N dimensions are generated randomly, while the K dimension is fixed to multiple given values. We compute the expected result using high level C++ code and compare it to the result of our kernel.

4.1.2.4 Verification of the GEMM kernel with lda>M, ldb>K or ldc>M
-------------------------------------------------------------------

This task is very similar to the previous one, but we need to verify the correctness of our ``matmul_m_n_k`` kernel for 1≤M≤64, 1≤N≤64, K∈[1,16,32,64,128], and lda>M, ldb>K or ldc>M. This means that we need to store the matrices in a way that they are not contiguous in memory. We can do this by first choosing strides that are larger than the M, N and K dimensions. Then we can use the strides to compute the addresses of the elements in the matrices. Next, we can use this strides to first allocate memory that is larger than the matrices and then only set the elements that are used in the computation. The other elements, which will be skipped due to the strides, will be set to 0. Lastly, we call our kernel and compare the result to the expected result:

.. code:: cpp

    TEST_CASE("Reference test for matmul kernel with variable M, N, K and lda>M, ldb>K or ldc>M", "[matmul][parameterized][larger strides]")
    {
        const int M = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32);
        const int N = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32);
        const int K = GENERATE(1, 16, 32, 64, 128);

        std::random_device rd;
        std::mt19937 gen(rd());

        std::uniform_int_distribution<int> strideDist(1, 10);

        // Set strides larger than dimensions
        const int lda = M + strideDist(gen);
        const int ldb = K + strideDist(gen);
        const int ldc = M + strideDist(gen);

        // Allocate space for matrices larger than M, N, K
        float *A = new float[lda * K];
        float *B = new float[ldb * N];
        float *C = new float[ldc * N];
        float *C_expected = new float[ldc * N];

        std::uniform_real_distribution<float> dist(-0.5f, 100.0f);

        // Initialize A
        for (int k = 0; k < K; ++k)
        {
            for (int m = 0; m < lda; ++m)
            {
                A[m + k * lda] = (m < M) ? dist(gen) : 0.0f;
            }
        }

        // Initialize B
        for (int n = 0; n < N; ++n)
        {
            for (int k = 0; k < ldb; ++k)
            {
                B[k + n * ldb] = (k < K) ? dist(gen) : 0.0f;
            }
        }

        // Initialize C and C_expected
        for (int n = 0; n < N; ++n)
        {
            for (int m = 0; m < ldc; ++m)
            {
                float value = (m < M) ? dist(gen) : 0.0f;
                C[m + n * ldc] = value;
                C_expected[m + n * ldc] = value;
            }
        }

        // Reference GEMM calculation
        for (int col = 0; col < N; ++col)
        {
            for (int row = 0; row < M; ++row)
            {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k)
                {
                    sum += A[row + k * lda] * B[k + col * ldb];
                }
                C_expected[row + col * ldc] += sum;
            }
        }

        mini_jit::Kernel l_kernel;
        mini_jit::kernels::matmul::matmul_m_n_k(l_kernel, M, N, K);
        mini_jit::Brgemm::kernel_t l_kernel_t = reinterpret_cast<mini_jit::Brgemm::kernel_t>(const_cast<void *>(l_kernel.get_kernel()));
        l_kernel_t(A, B, C, lda, ldb, ldc, 0, 0);

        for (int n = 0; n < N; ++n)
        {
            for (int m = 0; m < M; ++m)
            {
                REQUIRE(C[m + n * ldc] == Approx(C_expected[m + n * ldc]).margin(FLOAT_ERROR_MARGIN));
            }
        }

        delete[] A;
        delete[] B;
        delete[] C;
        delete[] C_expected;
    }

4.1.2.5 Benchmarking the GEMM kernel performance
-------------------------------------------------------------------

For the benchmarking we enhanced our ``benchmarking.cpp`` file that was used for the previous tasks.
Our task was to benchmark the performance of our generated kernels and report the measured
performance for 1≤M≤64, 1≤N≤64, K∈[1,16,32,64,128], lda=M, ldb=K and ldc=M. 

We were also given a baseline CSV file, which gave us a structure, on how to safe our benchmarking performance.
Our idea was run each of these benchmarks for a time of ``1.5s`` in order to guarantee comparable results.
During this time we calculated the number of iterations our ``matmul_m_n_k`` kernel would perform.
Using this metrics we could then calculate the performance in GFLOPs for the respective execution.

**matmul_m_n_k benchmarking approach for different M, N, and K**

.. code:: cpp

    // Generate and get the kernel function
    mini_jit::Kernel l_kernel;
    mini_jit::kernels::matmul::matmul_m_n_k(l_kernel, m_M, m_N, m_K);
    mini_jit::Brgemm::kernel_t l_kernel_t = reinterpret_cast<mini_jit::Brgemm::kernel_t>(const_cast<void *>(l_kernel.get_kernel()));

    // RUN
    long l_num_reps = 0;
    auto l_start_time = std::chrono::high_resolution_clock::now();
    double l_elapsed = 0.0;
    double l_runTimeMs = m_run_time * 1e6;
    do
    {
        l_kernel_t(m_A, m_B, m_C, m_M, m_K, m_M, 0, 0);
        ++l_num_reps;
        auto l_now = std::chrono::high_resolution_clock::now();
        l_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(l_now - l_start_time).count();
    } while (l_elapsed < l_runTimeMs);
    l_elapsed /= 1e6; // Convert to seconds
    // END RUN

    // Calculate metrics
    long l_totalOperations = 2.0 * m_M * m_N * m_K * l_num_reps;
    double l_gflops = ((double)l_totalOperations) / (l_elapsed * 1e9);

The results that we obtained were saved under ``benchmarks/gemm_perf.csv``. 

**Snippet of executed benchmarks for matmul_m_n_k**

.. code:: text

    m,n,k,br_size,trans_a,trans_b,trans_c,ld_a,ld_b,ld_c,br_stride_a,br_stride_b,num_reps,time,gflops
    1,1,1,1,0,0,0,0,0,0,0,0,54127879,1.5,0.0721705
    1,1,16,1,0,0,0,0,0,0,0,0,44228413,1.5,0.943539
    1,1,32,1,0,0,0,0,0,0,0,0,30326543,1.5,1.29393
    1,1,64,1,0,0,0,0,0,0,0,0,19160608,1.5,1.63504
    1,1,128,1,0,0,0,0,0,0,0,0,10973115,1.5,1.87274

4.1.3 Batch-Reduce GEMM
=========================

After generating our GEMM kernel for different values for the M, N, and K dimensions, we then implemented
a batched version of this kernel. This means we now had to implement kernels that support matrix multiplications 
of the form: C+=∑AᵢBᵢ.

4.1.3.1 Support Batch-Reduce GEMMs
------------------------------------

We started by altering our ``generate`` function, so that we would now accept a ``batch_size``.

.. literalinclude:: ../../src/Brgemm.cpp
    :language: cpp
    :lines: 57-66
    :lineno-match:
    :caption: handling of invalid values for ``br_size`` in the ``generate`` function
    :dedent:

.. literalinclude:: ../../src/Brgemm.cpp
    :language: cpp
    :lines: 77-90
    :lineno-match:
    :caption: implementation of ``br_size`` in the ``generate`` function
    :dedent:

We based our implementation for the ``matmul_br_m_n_k`` on our assembly implementation of the :ref:`batch-reduce GEMM <3.6 Batch-Reduce GEMM>`.
As we now had the additional values ``br_stride_a`` and ``br_stride_a`` we needed to slightly adjust the use of our registers.
Apart from that, we were ready to start. 

The first step we took was to initialize the loop counter for the batch dimension.

.. literalinclude:: ../../src/kernels/matmul/matmul_br_m_n_k.cpp
    :language: cpp
    :lines: 66-70
    :lineno-match:
    :caption: initialize loop counter for batch dimension
    :dedent:

Our second step was to make sure that after a GEMM has finished, we 
would increment the pointers, to move to the next respective matrices.

.. literalinclude:: ../../src/kernels/matmul/matmul_br_m_n_k.cpp
    :language: cpp
    :lines: 160-176
    :lineno-match:
    :caption: move to the next A and B matrix and restore the position for matrix C
    :dedent:

These were the only changes we had to make. Between the initialization of the loop 
and jumping to the next matrices, we would loop over our :ref:`matmul_m_n_k kernel <4.1.2 GEMM>`.

4.1.3.2 Verification of the Batch-Reduce GEMM kernel
----------------------------------------------------

Similar to the GEMM kernel, we also tested our implementation of the batch-reduce GEMM.
We executed several initializations of our kernel, using a similar approach to the testing of the GEMM kernel.

.. literalinclude:: ../../tests/unit/kernels/matmul/matmul_br_m_n_k.test.cpp
    :language: cpp
    :lines: 8-69
    :lineno-match:
    :caption: Unit test for the ``matmul_br_m_n_k`` kernel
    :dedent:

4.1.3.3 Benchmarking the Batch-Reduce GEMM kernel performance
--------------------------------------------------------------------------
For the benchmarking we, again, enhanced our ``benchmarking.cpp`` file.
We introduced a new function that should handle 1≤M≤64, 1≤N≤64, K∈[1,16,32,64,128], lda=M, ldb=K and ldc=M.

We reduced the time for our benchmarks to ``1.0s``.

Beside the fact, that we would now consider 16 Matrices for A and B, the calculation 
for the GFLOPs was than similar to the normal ``GEMM``.

.. literalinclude:: ../../src/benchmarks/matmul/Matmul_br_m_n_k.bench.cpp
    :language: cpp
    :lines: 47-69
    :lineno-match:
    :caption: ``matmul_br_m_n_k`` benchmarking approach for a batch size of 16 and different M, N, and K values
    :dedent:

The results that we obtained were saved under ``src/benchmark/br_gemm_perf.csv``. 

.. literalinclude:: ../../benchmarks/brgemm_perf.csv
    :language: text
    :lines: 1-15
    :lineno-match:
    :caption: Snippet of executed benchmarks for ``matmul_br_m_n_k``
    :dedent:

Evaluating our GFLOP performance, we can see that we achieve a similar performance as in our ``matmul_m_n_k`` benchmark.

**********************
4.2 Unary Primitives
**********************

.. note::
    For this submission, we overhauled our benchmarking framework. After compilation, the main entry point can be called using ``./build/<OS_NAME>/benchmarks``, but that will not actually run any benchmarks. Which benchmark types should be run is specified using command like arguments, such as ``matmul`` or ``unary``. Multiple benchmarks can be run at once, for example by running: ``./build/OS_NAME/benchmarks matmul unary``. The results are saved in the ``benchmarks`` folder in text files.

4.2.1 Zero Primitive
===========================

4.2.1.1 Zero Primitive Implementation
---------------------------------------

The first unary primitive we implemented was the zero primitive. This kernel is supposed to set all elements of the output matrix to zero, while ignoring the input matrix. This functionality can be implemented in many different ways, but we started with the arm instruction which we had already implemented: ``STR``. We called this version the ``XZR`` approach, because we are using the ``XZR`` (and sometimes ``WZR``) register to store zeroes in the output matrix. The limitation here is that the ``XZR`` is only 64 bits wide, which means that we can only set 2 FP32 values to zero at once. To improve this, we implemented a second version that uses ``Neon`` instructions. We first create a zero register using the ``EOR`` instruction (eg. ``eor v31.16b, v31.16b, v31.16b`` sets ``v31`` to zero) and then use ``STP`` to zero 8 FP32 values at once. This version is called the ``EOR`` approach.

.. literalinclude:: ../../src/kernels/unary/zero_primitive_xzr.cpp
    :language: cpp
    :lines: 56-70
    :lineno-match:
    :caption: general case for the ``XZR zero primitive``
    :dedent:

.. literalinclude:: ../../src/kernels/unary/zero_primitive.cpp
    :language: cpp
    :lines: 61-70
    :lineno-match:
    :caption: general case for the ``EOR zero primitive``
    :dedent:

In this primitive, we handle one column at a time. For all matrices where the number of rows is not divisible by 8, we implemented edge cases that handle the remaining elements. This approach is the same as we used in the matrix multiplication kernels, with the only difference being that we do not need to handle the K dimension.

Both versions also support transposition, by simply swapping the M and N dimensions.

4.2.1.1 Zero Primitive Benchmarks
---------------------------------------

We benchmarked the performance of our zero primitive for the given parameters (M=N=50, M=N=64, M=N=512 and M=N=2048) and obtained the following results:

.. literalinclude:: ../../benchmarks/unary_benchmarks.txt
    :language: text
    :lines: 113-168
    :lineno-match:
    :caption: Benchmarking results for the zero Primitive

In all cases, we can see that the ``EOR`` approach is significantly faster than the ``XZR`` approach. Transposition was not benchmarked, since the swapping of the dimensions happens in the high level code and not in the assembly code.

4.2.2 Identity Primitive
===========================

4.2.2.1 Identity Implementation
---------------------------------
Firstly we implemented the general identity for a matrix A.

This approach was pretty straight forward as we simply copied our ``zero_primitive`` kernel and replaced 
every 'zero store' with:

#. A load from matrix A at the specific address
#. A store, that would store the element from A in matrix B

.. literalinclude:: ../../src/kernels/unary/identity_primitive.cpp
    :language: cpp
    :lines: 57-69
    :lineno-match:
    :caption: ``8x8`` general case for the ``identity_primitive``
    :dedent:

For the base cases, where there was a remainder for the ``m`` dimension, we did the same thing.

.. literalinclude:: ../../src/kernels/unary/identity_primitive.cpp
    :language: cpp
    :lines: 95-101
    :lineno-match:
    :caption: ``m%8=5`` case for the ``identity_primitive``
    :dedent:

4.2.2.2 Identity Transposition Implementation
-----------------------------------------------

After implementing the general identity, we implemented a transposition version.
Our intuition to transpose the identity was to again look at the :ref:`4x4 tranposition kernel <3.7 Transposition>`.

We decided to take the 4x4 matrix as our general case. 
We would then first proceed, always in ``4x4`` blocks, in the ``m`` dimension.

.. literalinclude:: ../../src/kernels/unary/identity_trans_primitive.cpp
    :language: cpp
    :lines: 223-253
    :lineno-match:
    :caption: ``4x4`` general case for the ``identity_trans_primitive``
    :dedent:

To handle the different stores for ``4x4`` blocks that would not be on the matrix diagonal, we 
would do the following:

After processing a ``4x4`` block on the diagonal:

#. Jump by 4 rows in Matrix A
#. Jump by 4 columns in Matrix B

By using this approach, we would guarantee, that after processing a block in the matrix A, we could save it at the correct position in matrix B. For all cases, where the ``m`` dimension would not be divisible by 4, we would need to handle the remaining cases.

.. literalinclude:: ../../src/kernels/unary/identity_trans_primitive.cpp
    :language: cpp
    :lines: 339-356
    :lineno-match:
    :caption: ``2x4`` base case for the ``identity_trans_primitive``
    :dedent:

After implementing the base cases for remainders of ``m``, we would be able to process ``mx4`` blocks of our matrix.

That meant, we needed to consider cases, where there was a remainder of ``n``.
There were two things to consider:

#. The rightmost column (remainder of ``n``), which could be: ``4x3``, ``4x2`` or ``4x1``
#. The last piece in the rightmost corner (remainder of ``m`` and ``n``)

For both of these cases we would consider a similar implementing approach as for the ``m`` remainder implementation.

.. literalinclude:: ../../src/kernels/unary/identity_trans_primitive.cpp
    :language: cpp
    :lines: 529-546
    :lineno-match:
    :caption: ``4x2`` base case for the ``identity_trans_primitive``
    :dedent:

4.2.2.3 Benchmarks the Identity Kernel Performance
----------------------------------------------------

We benchmarked the performance of our identity primitive for the given parameters (M=N=50, M=N=64, M=N=512 and M=N=2048) and obtained the following results:

.. literalinclude:: ../../benchmarks/unary_benchmarks.txt
    :language: text
    :lines: 1-56
    :lineno-match:
    :caption: Benchmarking results for the identity primitives

Most notably, we can see that the performance of the transposition kernel is significantly lower for larger matrices, such as 512x512 and 2048x2048. Here we only achieved a bandwidth of 3.6 to 4 GiB/s, while all other configurations achieved bandwidths greater than 100 GiB/s.

4.2.3 ReLU Primitive
===========================

4.2.3.1 ReLU Primitive Implementation
---------------------------------------

The last unary primitive we implemented was the ReLU primitive. The Rectified Linear Unit activation function is defined as: ``f(x) = max(0, x)``, meaning that all negative values are set to zero and all positive values are kept as they are. To implement this, we first had to add support for the ``FMAX`` instruction, which computes the maximum of two values. Using the ``EOR`` instruction which we implemented for the zero primitive, we can create a zero register and then use the ``FMAX`` instruction to compute the maximum of the input value and zero. Since the primitive should also support transposition, we implemented two versions. 

The first version does not transpose the output and is structurally the same as the zero primitive. However instead of always storing zero, we now store the maximum of the input value and zero.

.. literalinclude:: ../../src/kernels/unary/relu_primitive.cpp
    :language: cpp
    :lines: 56-71
    :lineno-match:
    :caption: Performing the ReLU function on 8 values (``relu_primitive``)
    :dedent:

To support transposition, we started with the identity transposition primitive. The only addition we had to make was to add the ``FMAX`` instruction between the load and store instructions. The rest of the implementation is structurally the same as the identity transposition primitive. The difference can be seen in the following code snippets:

.. literalinclude:: ../../src/kernels/unary/identity_trans_primitive.cpp
    :language: cpp
    :lines: 223-244
    :lineno-match:
    :caption: Original transposition code (``identity_trans_primitive``)
    :dedent:

.. literalinclude:: ../../src/kernels/unary/relu_trans_primitive.cpp
    :language: cpp
    :lines: 226-253
    :lineno-match:
    :caption: Code with the ``FMAX`` instruction (``relu_trans_primitive``)
    :dedent:

4.2.3.2 ReLU Primitive Benchmarks
---------------------------------------

We benchmarked the performance of our ReLU primitive for the given parameters (M=N=50, M=N=64, M=N=512 and M=N=2048) and obtained the following results:

.. literalinclude:: ../../benchmarks/unary_benchmarks.txt
    :language: text
    :lines: 57-112
    :lineno-match:
    :caption: Benchmarking results for the ReLU primitives

The results match the pattern we saw for the zero and identity primitives. The transposition version is significantly slower than the non-transposition version, especially for the larger matrices. Here too, the 2048x2048 benchmark achieved worse results than the smaller matrices, both with and without transposition.