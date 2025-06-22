#####################
4. Code Generation
#####################

After developing Neon kernels for the first few weeks, we gained valuable insights into implementing different operations. 
In this phase of the project, our goal was to leverage that knowledge to build the backbone of a JIT C++ code generator. 
This code generator is designed to produce assembly code for arbitrary matrix dimensions and execute the generated code on our machine.

**********************
4.1 BRGEMM Primitive
**********************

In this section, we explain how we implemented the BRGEMM primitive for our machine learning compiler.

.. _4-1-1-microkernel:

4.1.1 Microkernel
===================

Before we began implementing the Batch-Reduce General Matrix-Matrix Multiplication kernel, we first needed to take a look at how we could multiply matrices using the JIT C++ code generator for assembly. To simplify things, we first put our focus on generating the fixed size microkernels which we had previously implemented in A64 ARM assembly (see: :ref:`3-neon`).

4.1.1.1 Instruction Generation
----------------------------------

The first step to generating a kernel was to create C++ mappings to the A64 ARM assembly instructions we needed. That is, for each instruction we use in our assembly kernel, we require a C++ function that can generate the instruction for us, with support for various input parameters. The output of such a function is a ``uint32_t`` value, representing the 32-bits of the assembly instruction.

.. note::

    Instead of working with binary numbers directly, we decided to use hexadecimal representation.

While we won't show all instructions that implemented, here are some examples:

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

Having implemented all necessary C++ functions for generating the assembly instructions, we then turned our attention to the microkernel generation. Here, the first kernel we approached was the ``matmul_16_6_1`` kernel. The process was to copy the assembly code line by line and replace all instructions with our C++ bindings. A part of the result can be seen in the following code snippet:

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

After setting the foundation for the execution of a specific ``GEMM`` kernel, our plan was now to extend the in :ref:`4-1-1-microkernel` implemented kernel to a more general ``GEMM`` kernel.

4.1.2.1 Implementation of a GEMM kernel
----------------------------------------

The general ``GEMM`` kernel should be able to compute C+=AB for arbitrary A, B and C matrices in the range of 1≤M≤1024, 1≤N≤1024, and 1≤K≤2048.

At first, we had to decide on how to block the matrices. In the M dimension, we decided to use a block size of 16 and in the ``n`` dimension we decided to use a block size of 4. The larger we keep the block size, the more efficiently we can use loads, stores and FMLA instructions. However, the issue with large block sizes is that we need to write a lot of specialized kernels for all M and N dimensions smaller or equal to the block size. If the input parameters are not multiples of the block size, we need to write additional code to handle the remaining elements. 

For a block size of M = 8 we already wrote a kernel in neon assembly, see :ref:`generic-kernel`. Using this generic kernel as a starting point, we have reduced the ``n`` dimension from 6 to 4. Our reasoning was that we wanted to reduce the number of specialized kernels we would need to write. Additionally, we assumed that in practice more numbers would be multiples of 4 instead of 6, thus not depending on such specialized kernels. Nevertheless, we made the decision to increase M from 8 to 16 to increase our overall performance. With this change, we introduced the ``matmul_m_4_k`` kernel, which computes C+=AB for matrices where M and K are freely configurable, and N is fixed at size 4.

The kernel first computes the number of blocks along the M dimension, as well as any remaining elements. 

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

But what does such a specialized kernel look like? For the most part, they are similar to the microkernels we implemented before. The only difference is that we need to adjust the loads, stores and FMLA instructions for a fixed M dimension. For example in the case of M = 3:

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

The FMLA instructions are also adjusted based on M dimension. For example, when M = 3, we need to use two FMLA instructions to compute the result:

**matmul_m_4_k: FMLA instructions with M = 3**

.. code:: cpp

    // B: COLUMN 0
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v29, gpr_t::x16, 0, neon_size_spec_t::s));
    kernel.add_instr(simd_fp::fmlaElem(simd_fp_t::v0, simd_fp_t::v24, simd_fp_t::v29, arr_spec_t::s2));
    kernel.add_instr(simd_fp::fmadd(simd_fp_t::v1, simd_fp_t::v25, simd_fp_t::v29, simd_fp_t::v1, neon_size_spec_t::s));

While one could use an ``fmla`` instruction and zero padding, we decided to use one ``fmla`` instruction for the first two elements and one ``fmadd`` instruction for the last element. We did not observe any performance differences between the two approaches, but chose the second one because to us it seemed more readable and easier to understand. The other specialized kernels for M = 1, 2, 4, 5, 6 and 7 are implemented similarly.

Having implemented the ``matmul_m_4_k`` kernel, we can now turn our attention towards the ``matmul_m_n_k`` kernel. Since we decided to block N by 4, we can use the same approach as before. We first compute the number of blocks along the ``n`` dimension and the remaining elements.

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

.. note::

    As seen in the code snippet above, we extended our kernel object by an ``add_label`` function and a ``getInstrCountFromLabel`` function. Internally, the kernel keeps track of the number of instructions that were added since the label was added. If we want to jump back to a label, we can use ``getInstrCountFromLabel`` to get the number of instructions we have to jump and multiply it by 4, because each instruction is 4 bytes long.

The full code is available in the file `matmul_m_n_k.cpp <https://github.com/Shad00Z/machine-learning-compilers/blob/main/src/kernels/matmul/matmul_m_n_k.cpp>`_.

4.1.2.2 Calling the GEMM kernel
----------------------------------------

Having implemented the code for the ``matmul_m_n_k``, we now had to find a way to call it. For this, we use a ``Brgemm`` class that contains a ``generate`` function. We use the same function to call our ``matmul_br_m_n_k`` BRGEMM kernel, which is explained in the next chapter. For more details on the ``Brgemm`` class, please refer to :ref:`4-1-3-2`.

4.1.2.3 Verification of the GEMM kernel with lda=M, ldb=K, ldc=M
-------------------------------------------------------------------

This task requires us to verify the correctness of our ``matmul_m_n_k`` kernel by comparing it to a reference implementation for 1≤M≤64, 1≤N≤64, K∈[1,16,32,64,128], with lda=M, ldb=K, and ldc=M.
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

The M and N dimensions are generated randomly, while the ``k`` dimension is fixed to multiple given values. We compute the expected result using high level C++ code and compare it to the result of our kernel.

4.1.2.4 Verification of the GEMM kernel with lda>M, ldb>K or ldc>M
-------------------------------------------------------------------

This task is very similar to the previous one, but we need to verify the correctness of our ``matmul_m_n_k`` kernel for 1≤M≤64, 1≤N≤64, K∈[1,16,32,64,128], and lda>M, ldb>K or ldc>M. This means that we need to store the matrices in a way that they are not contiguous in memory. We can do this by first choosing strides that are larger than the M, N and K dimensions. The next step is to use these strides to compute the addresses of the elements in the matrices. We can then use the strides to allocate memory larger larger than the matrices and set the elements used in the computation. The other elements, which will be skipped due to the strides, will be set to 0. Lastly, we call our kernel and compare the result to the expected result:

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

.. _4.1.2.5 GEMM_bench:

4.1.2.5 Benchmarking the GEMM kernel
---------------------------------------

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

The results that we obtained were saved under `benchmarks/gemm_perf.csv <https://github.com/Shad00Z/machine-learning-compilers/blob/main/benchmarks/gemm_perf.csv>`_. 

**Snippet of executed benchmarks for matmul_m_n_k**

.. code:: text

    m,n,k,br_size,trans_a,trans_b,trans_c,ld_a,ld_b,ld_c,br_stride_a,br_stride_b,num_reps,time,gflops
    1,1,1,1,0,0,0,0,0,0,0,0,54127879,1.5,0.0721705
    1,1,16,1,0,0,0,0,0,0,0,0,44228413,1.5,0.943539
    1,1,32,1,0,0,0,0,0,0,0,0,30326543,1.5,1.29393
    1,1,64,1,0,0,0,0,0,0,0,0,19160608,1.5,1.63504
    1,1,128,1,0,0,0,0,0,0,0,0,10973115,1.5,1.87274
    1,2,1,1,0,0,0,0,0,0,0,0,55889405,1.5,0.149038
    1,2,16,1,0,0,0,0,0,0,0,0,43394974,1.5,1.85152
    1,2,32,1,0,0,0,0,0,0,0,0,30144269,1.5,2.57231
    1,2,64,1,0,0,0,0,0,0,0,0,18992617,1.5,3.24141
    1,2,128,1,0,0,0,0,0,0,0,0,10804485,1.5,3.68793
    1,3,1,1,0,0,0,0,0,0,0,0,55753919,1.5,0.223016
    1,3,16,1,0,0,0,0,0,0,0,0,43017743,1.5,2.75314
    1,3,32,1,0,0,0,0,0,0,0,0,30005166,1.5,3.84066
    1,3,64,1,0,0,0,0,0,0,0,0,18859806,1.5,4.82811

4.1.3 Batch-Reduce GEMM
=========================

After generating our GEMM kernel for different values of the M, N, and K dimensions, we implemented a batched version of this kernel. 
This means we now had to implement kernels that support matrix multiplications of the form: C+=∑AᵢBᵢ.

4.1.3.1 Support for Batch-Reduce GEMMs
----------------------------------------

We based our ``matmul_br_m_n_k`` implementation on our assembly version of the :ref:`batch-reduce GEMM <3.6 Batch-Reduce GEMM>`.
As we now had the additional values ``br_stride_a`` and ``br_stride_a`` we needed to slightly adjust the use of our registers.
Apart from that, we were ready to start. 

The first step we took was to initialize the loop counter for the batch dimension.

**matmul_br_m_n_k: br counter initialization**

.. code:: cpp

    // batch counter
    kernel.add_instr(base::mov(gpr_t::x25, br_size));
    kernel.add_label("batch_loop");

The second step was to make sure that after a GEMM has finished, we would increment the pointers, to move to the next respective matrices.

.. code:: cpp

    // handle batching
    // move to next A matrix
    kernel.add_instr(base::add(gpr_t::x0, gpr_t::x0, gpr_t::x6, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x8, gpr_t::x0));
    // move to next B matrix
    kernel.add_instr(base::add(gpr_t::x1, gpr_t::x1, gpr_t::x7, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x20, gpr_t::x1));
    // restore pointer to C matrix
    kernel.add_instr(base::mov(gpr_t::x21, gpr_t::x2));
    kernel.add_instr(base::mov(gpr_t::x10, gpr_t::x21));

    // decrement batch loop counter
    kernel.add_instr(base::sub(gpr_t::x25, gpr_t::x25, 1, 0));
    // check if loop counter is zero
    int l_batchLoopInstrCount = kernel.getInstrCountFromLabel("batch_loop");
    kernel.add_instr(base::cbnz(gpr_t::x25, -l_batchLoopInstrCount * 4));

These were the only changes we had to make. Between initializing the loop and jumping to the next blocks in our matrices, we would loop over our :ref:`matmul_m_n_k kernel <4.1.2 GEMM>`.

.. _4-1-3-2:

4.1.3.2 Calling the Batch-Reduce GEMM kernel
----------------------------------------------

In order to actually call our ``GEMM`` and ``BRGEMM`` kernels, we had to implement a common entry point. The ``Brgemm`` class is responsible for this task.
It first checks all input parameters for their validity and then makes calls to the kernels based on the batch-reduce size.

**Brgemm.cpp**

.. code:: cpp

    mini_jit::error_t mini_jit::Brgemm::generate(uint32_t m,
                                                uint32_t n,
                                                uint32_t k,
                                                uint32_t br_size,
                                                uint32_t trans_a,
                                                uint32_t trans_b,
                                                uint32_t trans_c,
                                                dtype_t dtype)
    {
        /**
        * Currently supported:
        * trans_a, trans_b, trans_c: Column-major
        * dtype: fp32
        */
        if (m <= 0)
        {
            std::cout << ("M must be greater than 0") << std::endl;
            return error_t::wrong_dimension;
        }
        else if (m > 2048)
        {
            std::cout << ("M must not be greater than 2048") << std::endl;
            return error_t::wrong_dimension;
        }
        else if (n <= 0)
        {
            std::cout << ("N must be greater than 0") << std::endl;
            return error_t::wrong_dimension;
        }
        else if (n > 2048)
        {
            std::cout << ("N must not be greater than 2048") << std::endl;
            return error_t::wrong_dimension;
        }
        else if (k <= 0)
        {
            std::cout << ("K must be greater than 0") << std::endl;
            return error_t::wrong_dimension;
        }
        else if (k > 2048)
        {
            std::cout << ("K must not be greater than 2048") << std::endl;
            return error_t::wrong_dimension;
        }
        else if (br_size <= 0)
        {
            std::cout << ("BR_SIZE must greater than 0") << std::endl;
            return error_t::wrong_dimension;
        }
        else if (br_size > 2048)
        {
            std::cout << ("BR_SIZE must not be greater than 2048") << std::endl;
            return error_t::wrong_dimension;
        }
        else if (trans_a != 0 || trans_b != 0 || trans_c != 0)
        {
            std::cout << ("Matrix ordering must be column-major") << std::endl;
            return error_t::wrong_matrix_ordering_format;
        }
        else if (dtype != dtype_t::fp32)
        {
            std::cout << ("Matrix data type must be fp32") << std::endl;
            return error_t::wrong_dtype;
        }
        else
        {
            reset_kernel();

            if (br_size == 1)
            {
                mini_jit::kernels::matmul::matmul_m_n_k(*m_kernel, m, n, k);
            }
            else
            {
                mini_jit::kernels::matmul::matmul_br_m_n_k(*m_kernel, m, n, k, br_size);
            }

            // Valid matrix kernel
            return error_t::success;
        }
    }

    mini_jit::Brgemm::kernel_t mini_jit::Brgemm::get_kernel() const
    {
        return reinterpret_cast<kernel_t>(const_cast<void *>(m_kernel->get_kernel()));
    }

    void mini_jit::Brgemm::reset_kernel()
    {
        if (m_kernel)
        {
            delete m_kernel;
            m_kernel = nullptr;
        }
        m_kernel = new mini_jit::Kernel();
    }

The example below demonstrates how this function can be called:

**Example code for generating and executing a kernel**

.. code:: cpp

    mini_jit::Kernel l_kernel;
    mini_jit::kernels::matmul::matmul_m_n_k(l_kernel, M, N, K);
    mini_jit::Brgemm::kernel_t l_kernel_t = reinterpret_cast<mini_jit::Brgemm::kernel_t>(const_cast<void *>(l_kernel.get_kernel()));
    l_kernel_t(A, B, C, M, K, M, 0, 0);

4.1.3.3 Verification of the Batch-Reduce GEMM kernel
------------------------------------------------------

Similar to the ``GEMM`` kernel, we also tested our implementation of the batch-reduce GEMM.
We executed several initializations of our kernel, using a similar approach to the testing of the ``GEMM`` kernel:

.. code:: cpp

    TEST_CASE("Reference test for batch reduce matmul kernel with variable M, N, K", "[br_matmul][parameterized]")
    {
        const int M = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        const int N = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        const int K = GENERATE(1, 16, 32, 64, 128);
        const int br_size = 16;

        float *A = new float[M * K * br_size];
        float *B = new float[K * N * br_size];
        float *C = new float[M * N];
        float *C_expected = new float[M * N];

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-0.5f, 100.0f);

        for (int i = 0; i < M * K * br_size; ++i)
        {
            A[i] = dist(gen);
        }

        for (int i = 0; i < K * N * br_size; ++i)
        {
            B[i] = dist(gen);
        }

        for (int i = 0; i < M * N; ++i)
        {
            C[i] = C_expected[i] = dist(gen);
        }

        // Reference batched GEMM calculation
        for (int col = 0; col < N; ++col)
        {
            for (int row = 0; row < M; ++row)
            {
                float sum = 0.0f;
                for (int br = 0; br < br_size; ++br)
                {
                    for (int k = 0; k < K; ++k)
                    {
                        sum += A[br * M * K + row + k * M] * B[br * K * N + k + col * K];
                    }
                }
                C_expected[row + col * M] += sum;
            }
        }

        mini_jit::Kernel l_kernel;
        mini_jit::kernels::matmul::matmul_br_m_n_k(l_kernel, M, N, K, br_size);
        mini_jit::Brgemm::kernel_t l_kernel_t = reinterpret_cast<mini_jit::Brgemm::kernel_t>(const_cast<void *>(l_kernel.get_kernel()));
        l_kernel_t(A, B, C, M, K, M, M * K, K * N);

        for (int i = 0; i < M * N; ++i)
        {
            REQUIRE(C[i] == Approx(C_expected[i]).margin(FLOAT_ERROR_MARGIN));
        }

        delete[] A;
        delete[] B;
        delete[] C;
        delete[] C_expected;
    }

.. _4.1.3.4 BRGEMM_bench:

4.1.3.4 Benchmarking the Batch-Reduce GEMM kernel
---------------------------------------------------

For the benchmarks, we enhanced our ``benchmarking.cpp`` file again.
We introduced a new function that should handle 1≤M≤64, 1≤N≤64, K∈[1,16,32,64,128], lda=M, ldb=K and ldc=M and reduced the time for our benchmarks to ``1.0s``. The calculation for the GFLOPs is almost the same as for the ``GEMM`` kernel, however now we also need to multiply the number of operations by the Batch-Reduce dimension size ``br_size``.

.. code:: cpp

    // Generate and get the kernel function
    mini_jit::Kernel l_kernel;
    mini_jit::kernels::matmul::matmul_br_m_n_k(l_kernel, m_M, m_N, m_K, m_br_size);
    mini_jit::Brgemm::kernel_t l_kernel_t = reinterpret_cast<mini_jit::Brgemm::kernel_t>(const_cast<void *>(l_kernel.get_kernel()));

    // RUN
    long l_num_reps = 0;
    auto l_start_time = std::chrono::high_resolution_clock::now();
    double l_elapsed = 0.0;
    double l_runTimeMs = m_run_time * 1e6;
    do
    {
        l_kernel_t(m_A, m_B, m_C, m_M, m_K, m_M, m_M * m_K, m_K * m_N);
        ++l_num_reps;
        auto l_now = std::chrono::high_resolution_clock::now();
        l_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(l_now - l_start_time).count();
    } while (l_elapsed < l_runTimeMs);
    l_elapsed /= 1e6; // Convert to seconds
    // END RUN

    // Calculate metrics
    long l_totalOperations = 2.0 * m_M * m_N * m_K * l_num_reps * m_br_size;
    double l_gflops = ((double)l_totalOperations) / (l_elapsed * 1e9);

The results that we obtained were saved in `br_gemm_perf.csv <https://github.com/Shad00Z/machine-learning-compilers/blob/main/benchmarks/brgemm_perf.csv>`_. 

**Snippet of executed benchmarks for matmul_br_m_n_k**

.. code:: text

    m,n,k,br_size,trans_a,trans_b,trans_c,ld_a,ld_b,ld_c,br_stride_a,br_stride_b,num_reps,time,gflops
    1,1,1,16,0,0,0,1,1,1,1,1,14713094,1,0.470819
    1,1,16,16,0,0,0,1,16,1,16,16,3412968,1,1.74744
    1,1,32,16,0,0,0,1,32,1,32,32,1845891,1,1.89019
    1,1,64,16,0,0,0,1,64,1,64,64,1007179,1,2.0627
    1,1,128,16,0,0,0,1,128,1,128,128,516692,1,2.11637
    1,2,1,16,0,0,0,1,1,1,1,2,15004415,1,0.960283
    1,2,16,16,0,0,0,1,16,1,16,32,3483409,1,3.56701
    1,2,32,16,0,0,0,1,32,1,32,64,1914029,1,3.91993
    1,2,64,16,0,0,0,1,64,1,64,128,1005414,1,4.11817
    1,2,128,16,0,0,0,1,128,1,128,256,515745,1,4.22498
    1,3,1,16,0,0,0,1,1,1,1,3,14941217,1,1.43436
    1,3,16,16,0,0,0,1,16,1,16,48,3458013,1,5.31151
    1,3,32,16,0,0,0,1,32,1,32,96,1911851,1,5.87321
    1,3,64,16,0,0,0,1,64,1,64,192,1004800,1,6.17349

Evaluating our GFLOP performance, we can see that we achieve a similar performance as in our ``matmul_m_n_k`` benchmark.

.. note::

    Both the :ref:`gemm <4.1.2.5 GEMM_bench>` and :ref:`brgemm <4.1.3.4 BRGEMM_bench>` benchmarks were executed using our initial kernel configurations of M=8 and N=4.
    Therefore, the results should be viewed carefully, as the new configuration M=16 and N=4 should drastically enhance the throughput, especially for large matrices.

**********************
4.2 Unary Primitives
**********************

After implementing our main primitives using the ``GEMM`` and ``BRGEMM`` kernels, the next step was to implement unary primitives. 
These can be called before an operation is executed (first touch) or after the final block of a matrix has been processed (last touch). 
Specifically we are implementing three of those primitives:

1. Zero Primitive
2. Identity Primitive
3. ReLU Primitive

.. note::

    For this submission, we overhauled our benchmarking framework once again. 
    After compilation, the main entry point can be called using ``./build/<OS_NAME>/benchmarks``, but this alone will not execute any benchmarks. 
    The benchmark types to run are specified using command-line arguments, such as ``matmul`` or ``unary``. 
    Multiple benchmarks can be run at once, for example by running: ``./build/OS_NAME/benchmarks matmul unary``. 
    The results are saved as text files in the ``benchmarks`` folder.

4.2.1 Zero Primitive
===========================

The first unary primitive we implemented was the zero primitive. 
This kernel is supposed to set all elements of the output matrix to zero, while ignoring the input matrix.
For this reason, this primitive is exclusively executed as a first touch primitive.

4.2.1.1 Zero Primitive Implementation
---------------------------------------

The functionality of the zero primitive can be implemented in many different ways, but we started with using an ARM instruction which we had already implemented: ``STR``. We refer to this version as the ``XZR`` approach, because it uses the ``XZR`` (and sometimes ``WZR``) register to store zeroes in the output matrix. The limitation here is that the ``XZR`` is only 64 bits wide, which means we can only set 2 FP32 values to zero at once. To improve this, we implemented a second version that uses ``Neon`` instructions. We first created a zero register using the ``EOR`` instruction (eg. ``eor v31.16b, v31.16b, v31.16b`` sets ``v31`` to zero) and then use ``STP`` to zero 8 FP32 values at once. This version is called the ``EOR`` approach.

**XZR Zero Primitive: main loop**

.. code:: cpp

    kernel.add_label("m_8_loop");
    // store 8 zeros
    kernel.add_instr(base::mov(gpr_t::x8, gpr_t::x7));
    kernel.add_instr(base::strPost(gpr_t::xzr, gpr_t::x8, 8));
    kernel.add_instr(base::strPost(gpr_t::xzr, gpr_t::x8, 8));
    kernel.add_instr(base::strPost(gpr_t::xzr, gpr_t::x8, 8));
    kernel.add_instr(base::str(gpr_t::xzr, gpr_t::x8, 0));

    // jump by 8 rows
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, 8*4, 0));
    // decrement m loop counter
    kernel.add_instr(base::sub(gpr_t::x6, gpr_t::x6, 1, 0));
    // check if loop counter is zero
    int l_mLoopInstrCount = kernel.getInstrCountFromLabel("m_8_loop");
    kernel.add_instr(base::cbnz(gpr_t::x6, -l_mLoopInstrCount * 4));

**EOR Zero Primitive: main loop**

.. code:: cpp

    kernel.add_label("m_8_loop");
    // store 8 zeros
    kernel.add_instr(simd_fp::stp(simd_fp_t::v31, simd_fp_t::v31, gpr_t::x7, 0, neon_size_spec_t::q));
    // jump by 8 rows
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, 8*4, 0));
    // decrement m loop counter
    kernel.add_instr(base::sub(gpr_t::x6, gpr_t::x6, 1, 0));
    // check if loop counter is zero
    int l_mLoopInstrCount = kernel.getInstrCountFromLabel("m_8_loop");
    kernel.add_instr(base::cbnz(gpr_t::x6, -l_mLoopInstrCount * 4));

In this primitive, we handle one column at a time. For all matrices where the number of rows is not divisible by 8, we implemented edge cases that handle the remaining elements. This approach is the same as the one we used in the matrix multiplication kernels, with the only difference being that we do not need to handle the K dimension.

4.2.1.1 Zero Primitive Benchmarks
---------------------------------------

We benchmarked the performance of our zero primitive for the given parameters (M=N=50, M=N=64, M=N=512 and M=N=2048) and obtained the following results:

**Benchmarking results for the zero primitives**

.. code:: text

    Running zero_eor_primitive 50x50 benchmark
    Total time (s):                       3
    Total reps:                           24095571
    Total number of elements:             60238927500
    Total amount of processed data (GiB): 448.815
    Bandwidth (GiB/s)                     149.605
    --------------------------------------------------
    Running zero_eor_primitive 64x64 benchmark
    Total time (s):                       3
    Total reps:                           14348177
    Total number of elements:             58770132992
    Total amount of processed data (GiB): 437.872
    Bandwidth (GiB/s)                     145.957
    --------------------------------------------------
    Running zero_eor_primitive 512x512 benchmark
    Total time (s):                       3
    Total reps:                           333722
    Total number of elements:             87483219968
    Total amount of processed data (GiB): 651.801
    Bandwidth (GiB/s)                     217.267
    --------------------------------------------------
    Running zero_eor_primitive 2048x2048 benchmark
    Total time (s):                       3.00013
    Total reps:                           8570
    Total number of elements:             35945185280
    Total amount of processed data (GiB): 267.812
    Bandwidth (GiB/s)                     89.2671
    --------------------------------------------------
    Running zero_xzr_primitive 50x50 benchmark
    Total time (s):                       3
    Total reps:                           18821607
    Total number of elements:             47054017500
    Total amount of processed data (GiB): 350.58
    Bandwidth (GiB/s)                     116.86
    --------------------------------------------------
    Running zero_xzr_primitive 64x64 benchmark
    Total time (s):                       3
    Total reps:                           8987787
    Total number of elements:             36813975552
    Total amount of processed data (GiB): 274.285
    Bandwidth (GiB/s)                     91.4285
    --------------------------------------------------
    Running zero_xzr_primitive 512x512 benchmark
    Total time (s):                       3
    Total reps:                           184240
    Total number of elements:             48297410560
    Total amount of processed data (GiB): 359.844
    Bandwidth (GiB/s)                     119.948
    --------------------------------------------------
    Running zero_xzr_primitive 2048x2048 benchmark
    Total time (s):                       3.0004
    Total reps:                           8216
    Total number of elements:             34460401664
    Total amount of processed data (GiB): 256.75
    Bandwidth (GiB/s)                     85.5719
    --------------------------------------------------

In all cases, we can see that the ``EOR`` approach is significantly faster than the ``XZR`` approach. Transposition was not benchmarked, since the dimension swapping happens in the high-level code and not in the assembly code.

4.2.2 Identity Primitive
===========================

This primitive differs slightly from the zero and ReLU primitives. 
The identity (or copy) primitive is intended to copy values from the input matrix to the output matrix, while considering for potential transpositions. 
Since this does not represent a true first or last touch, we implemented this primitive as an additional main primitive.

4.2.2.1 Identity Implementation
---------------------------------

Firstly we implemented the general identity for a matrix A.

This approach was mostly straight forward, as we copied our ``zero_primitive`` kernel and replaced 
every 'zero store' with:

#. a load from matrix ``A`` at the specific address, and
#. a store, that would store the element from ``A`` in matrix ``B``.

**Identity Primitive: main loop**

.. code:: cpp

    kernel.add_label("m_8_loop");
    // load and store 8 rows of A and B
    kernel.add_instr(simd_fp::ldp(simd_fp_t::v0, simd_fp_t::v1, gpr_t::x8, 0, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::stp(simd_fp_t::v0, simd_fp_t::v1, gpr_t::x7, 0, neon_size_spec_t::q));
    // jump by 8 rows
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, 8*4, 0));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, 8*4, 0));
    // decrement m loop counter
    kernel.add_instr(base::sub(gpr_t::x6, gpr_t::x6, 1, 0));
    // check if loop counter is zero
    int l_mLoopInstrCount = kernel.getInstrCountFromLabel("m_8_loop");
    kernel.add_instr(base::cbnz(gpr_t::x6, -l_mLoopInstrCount * 4));

For the edge cases where there was a remainder for the ``m`` dimension, we used the same procedure as before:

**Identity Primitive: M = 5 edge case**

.. code:: cpp

    case 5:
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v0, gpr_t::x8, 16, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v1, gpr_t::x8, 0, neon_size_spec_t::s));

    kernel.add_instr(simd_fp::strPost(simd_fp_t::v0, gpr_t::x7, 16, neon_size_spec_t::q));
    kernel.add_instr(simd_fp::str(simd_fp_t::v1, gpr_t::x7, 0, neon_size_spec_t::s));
    break;

4.2.2.2 Identity Transposition Implementation
-----------------------------------------------

After implementing the general identity, we implemented a transposition version.
Our intuition to transpose the identity was to look at the :ref:`4x4 tranposition kernel <3.7 Transposition>`.

We decided to take the 4x4 matrix as our general case. 

**Identity Transposition Primitive: main loop**

.. code:: cpp

    // Load 4x4 block of A (input matrix)
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v0, gpr_t::x7, 0, neon_size_spec_t::q));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v1, gpr_t::x7, 0, neon_size_spec_t::q));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v2, gpr_t::x7, 0, neon_size_spec_t::q));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v3, gpr_t::x7, 0, neon_size_spec_t::q));

    // Transpose 4x4 block
    // TRN
    kernel.add_instr(simd_fp::trn1(simd_fp_t::v4, simd_fp_t::v0, simd_fp_t::v2, arr_spec_t::s4));
    kernel.add_instr(simd_fp::trn1(simd_fp_t::v5, simd_fp_t::v1, simd_fp_t::v3, arr_spec_t::s4));
    kernel.add_instr(simd_fp::trn2(simd_fp_t::v6, simd_fp_t::v0, simd_fp_t::v2, arr_spec_t::s4));
    kernel.add_instr(simd_fp::trn2(simd_fp_t::v7, simd_fp_t::v1, simd_fp_t::v3, arr_spec_t::s4));

    // ZIP
    kernel.add_instr(simd_fp::zip1(simd_fp_t::v8, simd_fp_t::v4, simd_fp_t::v5, arr_spec_t::s4));
    kernel.add_instr(simd_fp::zip1(simd_fp_t::v9, simd_fp_t::v6, simd_fp_t::v7, arr_spec_t::s4));

    kernel.add_instr(simd_fp::zip2(simd_fp_t::v10, simd_fp_t::v4, simd_fp_t::v5, arr_spec_t::s4));
    kernel.add_instr(simd_fp::zip2(simd_fp_t::v11, simd_fp_t::v6, simd_fp_t::v7, arr_spec_t::s4));

    // Store 4x4 Block of B
    kernel.add_instr(simd_fp::str(simd_fp_t::v8, gpr_t::x8, 0, neon_size_spec_t::q));
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, gpr_t::x3, 0, 0));
    kernel.add_instr(simd_fp::str(simd_fp_t::v9, gpr_t::x8, 0, neon_size_spec_t::q));
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, gpr_t::x3, 0, 0));
    kernel.add_instr(simd_fp::str(simd_fp_t::v10, gpr_t::x8, 0, neon_size_spec_t::q));
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, gpr_t::x3, 0, 0));
    kernel.add_instr(simd_fp::str(simd_fp_t::v11, gpr_t::x8, 0, neon_size_spec_t::q));

To handle the different stores for ``4x4`` blocks that are not on the matrix diagonal, we 
would do the following:

After processing a ``4x4`` block on the diagonal:

#. Jump by 4 rows in Matrix A
#. Jump by 4 columns in Matrix B

By using this approach, we would guarantee that after processing a block in the matrix A, we could save it at the correct position in matrix B. For all cases where the ``m`` dimension is not be divisible by 4, we would need to implement specific kernels.

**Identity Transposition Primitive: 2x4 edge case**

.. code:: cpp

    kernel.add_instr(simd_fp::ldr(simd_fp_t::v3, gpr_t::x7, 0, neon_size_spec_t::d));

    // Transpose 2x4 block
    // TRN
    kernel.add_instr(simd_fp::trn1(simd_fp_t::v4, simd_fp_t::v0, simd_fp_t::v2, arr_spec_t::s4));
    kernel.add_instr(simd_fp::trn1(simd_fp_t::v5, simd_fp_t::v1, simd_fp_t::v3, arr_spec_t::s4));

    kernel.add_instr(simd_fp::trn2(simd_fp_t::v6, simd_fp_t::v0, simd_fp_t::v2, arr_spec_t::s4));
    kernel.add_instr(simd_fp::trn2(simd_fp_t::v7, simd_fp_t::v1, simd_fp_t::v3, arr_spec_t::s4));

    // ZIP
    kernel.add_instr(simd_fp::zip1(simd_fp_t::v8, simd_fp_t::v4, simd_fp_t::v5, arr_spec_t::s4));
    kernel.add_instr(simd_fp::zip1(simd_fp_t::v9, simd_fp_t::v6, simd_fp_t::v7, arr_spec_t::s4));

    // Store 2x4 Block of B
    kernel.add_instr(simd_fp::str(simd_fp_t::v8, gpr_t::x8, 0, neon_size_spec_t::q));
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, gpr_t::x3, 0, 0));

After implementing the edge cases for remainders of ``m``, we would be able to process ``mx4`` blocks of our matrix.

That meant we needed to consider cases where there was a remainder of ``n``.
There were two things to consider:

#. The rightmost column (remainder of ``n``), which could be: ``4x3``, ``4x2`` or ``4x1``
#. The last piece in the rightmost corner (remainder of ``m`` and ``n``)

For both of these cases we would consider a similar implementing approach as for the ``m`` remainder implementation.

**Identity Transposition Primitive: 4x2 edge case**

.. code:: cpp

    // Load 4x2 block of A (input matrix)
    kernel.add_instr(base::mov(gpr_t::x17, gpr_t::x7));
    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v0, gpr_t::x17, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v1, gpr_t::x17, 0, neon_size_spec_t::d));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));
    kernel.add_instr(base::mov(gpr_t::x17, gpr_t::x7));

    kernel.add_instr(simd_fp::ldrPost(simd_fp_t::v2, gpr_t::x17, 8, neon_size_spec_t::d));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v3, gpr_t::x17, 0, neon_size_spec_t::d));

    // Transpose 4x2 matrix
    // TRN
    kernel.add_instr(simd_fp::trn1(simd_fp_t::v4, simd_fp_t::v0, simd_fp_t::v2, arr_spec_t::s4));
    kernel.add_instr(simd_fp::trn2(simd_fp_t::v5, simd_fp_t::v0, simd_fp_t::v2, arr_spec_t::s4));

    kernel.add_instr(simd_fp::trn1(simd_fp_t::v6, simd_fp_t::v1, simd_fp_t::v3, arr_spec_t::s4));
    kernel.add_instr(simd_fp::trn2(simd_fp_t::v7, simd_fp_t::v1, simd_fp_t::v3, arr_spec_t::s4));

    // Store 4x2 Block of B
    kernel.add_instr(simd_fp::str(simd_fp_t::v4, gpr_t::x8, 0, neon_size_spec_t::d));
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, gpr_t::x3, 0, 0));

    kernel.add_instr(simd_fp::str(simd_fp_t::v5, gpr_t::x8, 0, neon_size_spec_t::d));
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, gpr_t::x3, 0, 0));

    kernel.add_instr(simd_fp::str(simd_fp_t::v6, gpr_t::x8, 0, neon_size_spec_t::d));
    kernel.add_instr(base::add(gpr_t::x8, gpr_t::x8, gpr_t::x3, 0, 0));

    kernel.add_instr(simd_fp::str(simd_fp_t::v7, gpr_t::x8, 0, neon_size_spec_t::d));

4.2.2.3 Benchmarks the Identity Kernel Performance
----------------------------------------------------

We benchmarked the performance of our identity primitive for the given parameters (M=N=50, M=N=64, M=N=512 and M=N=2048) and obtained the following results:

**Benchmarking results for the identity primitives**

.. code:: text

    Running identity_primitive 50x50 benchmark
    Total time (s):                       3
    Total reps:                           20635000
    Total number of elements:             51587500000
    Total amount of processed data (GiB): 384.357
    Bandwidth (GiB/s)                     128.119
    --------------------------------------------------
    Running identity_primitive 64x64 benchmark
    Total time (s):                       3
    Total reps:                           14687433
    Total number of elements:             60159725568
    Total amount of processed data (GiB): 448.225
    Bandwidth (GiB/s)                     149.408
    --------------------------------------------------
    Running identity_primitive 512x512 benchmark
    Total time (s):                       3
    Total reps:                           186337
    Total number of elements:             48847126528
    Total amount of processed data (GiB): 363.939
    Bandwidth (GiB/s)                     121.313
    --------------------------------------------------
    Running identity_primitive 2048x2048 benchmark
    Total time (s):                       3.00001
    Total reps:                           9976
    Total number of elements:             41842376704
    Total amount of processed data (GiB): 311.75
    Bandwidth (GiB/s)                     103.916
    --------------------------------------------------
    Running identity_trans_primitive 50x50 benchmark
    Total time (s):                       3
    Total reps:                           17759330
    Total number of elements:             44398325000
    Total amount of processed data (GiB): 330.793
    Bandwidth (GiB/s)                     110.264
    --------------------------------------------------
    Running identity_trans_primitive 64x64 benchmark
    Total time (s):                       3
    Total reps:                           11603499
    Total number of elements:             47527931904
    Total amount of processed data (GiB): 354.111
    Bandwidth (GiB/s)                     118.037
    --------------------------------------------------
    Running identity_trans_primitive 512x512 benchmark
    Total time (s):                       3.00044
    Total reps:                           6236
    Total number of elements:             1634729984
    Total amount of processed data (GiB): 12.1797
    Bandwidth (GiB/s)                     4.0593
    --------------------------------------------------
    Running identity_trans_primitive 2048x2048 benchmark
    Total time (s):                       3.00888
    Total reps:                           347
    Total number of elements:             1455423488
    Total amount of processed data (GiB): 10.8438
    Bandwidth (GiB/s)                     3.60391
    --------------------------------------------------

Most notably, we can see that the performance of the transposition kernel is significantly lower for larger matrices, such as 512x512 and 2048x2048. Here, we achieved a bandwidth of only 3.6 to 4 GiB/s, while all other configurations achieved bandwidths greater than 100 GiB/s.

4.2.3 ReLU Primitive
===========================

The last unary primitive we implemented was the ReLU primitive, a commonly employed function machine learning models. 
The Rectified Linear Unit activation function is defined as: ``f(x) = max(0, x)``, meaning that all negative values are set to zero and all positive values are kept as they are.

4.2.3.1 ReLU Primitive Implementation
---------------------------------------

To implement this, we first had to add support for the ``FMAX`` instruction, which computes the maximum of two values. Using the ``EOR`` instruction which we implemented for the zero primitive, we can create a zero register and then use the ``FMAX`` instruction to compute the maximum of the input value and zero. Since the primitive should also support transposition, we implemented two versions. 

The first version does not transpose the output and is structurally the same as the zero primitive. However instead of always storing zero, we now store the maximum of the input value and zero.

**ReLU Primitive: main loop**

.. code:: cpp

    kernel.add_label("m_8_loop");
    kernel.add_instr({
    // load 8 elements from A
    simd_fp::ldp(simd_fp_t::v0, simd_fp_t::v1, gpr_t::x8, 0, neon_size_spec_t::q),
    // compute f(x)=max(x,0)
    simd_fp::fmax(simd_fp_t::v0, simd_fp_t::v0, simd_fp_t::v31, arr_spec_t::s4),
    simd_fp::fmax(simd_fp_t::v1, simd_fp_t::v1, simd_fp_t::v31, arr_spec_t::s4),
    // store 8 elements to B
    simd_fp::stp(simd_fp_t::v0, simd_fp_t::v1, gpr_t::x9, 0, neon_size_spec_t::q),
    // jump by 8 rows
    base::add(gpr_t::x8, gpr_t::x8, 8*4, 0),
    base::add(gpr_t::x9, gpr_t::x9, 8*4, 0),
    // decrement m loop counter
    base::sub(gpr_t::x7, gpr_t::x7, 1, 0),
    });
    // check if loop counter is zero
    kernel.add_instr(base::cbnz(gpr_t::x7, -kernel.getInstrCountFromLabel("m_8_loop") * 4));

To support transposition, we started with the identity transposition primitive. The only addition we had to make was to add the ``FMAX`` instruction between the load and store instructions. The rest of the implementation is structurally identical to the identity transposition primitive. The difference can be seen in the following code snippets:

**Original transposition code (identity_trans_primitive)**

.. code:: cpp

    // Load 4x4 block of A (input matrix)
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v0, gpr_t::x7, 0, neon_size_spec_t::q));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v1, gpr_t::x7, 0, neon_size_spec_t::q));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v2, gpr_t::x7, 0, neon_size_spec_t::q));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v3, gpr_t::x7, 0, neon_size_spec_t::q));

    // Transpose 4x4 block
    // TRN
    kernel.add_instr(simd_fp::trn1(simd_fp_t::v4, simd_fp_t::v0, simd_fp_t::v2, arr_spec_t::s4));
    kernel.add_instr(simd_fp::trn1(simd_fp_t::v5, simd_fp_t::v1, simd_fp_t::v3, arr_spec_t::s4));
    kernel.add_instr(simd_fp::trn2(simd_fp_t::v6, simd_fp_t::v0, simd_fp_t::v2, arr_spec_t::s4));
    kernel.add_instr(simd_fp::trn2(simd_fp_t::v7, simd_fp_t::v1, simd_fp_t::v3, arr_spec_t::s4));

    // ZIP
    kernel.add_instr(simd_fp::zip1(simd_fp_t::v8, simd_fp_t::v4, simd_fp_t::v5, arr_spec_t::s4));
    kernel.add_instr(simd_fp::zip1(simd_fp_t::v9, simd_fp_t::v6, simd_fp_t::v7, arr_spec_t::s4));

    kernel.add_instr(simd_fp::zip2(simd_fp_t::v10, simd_fp_t::v4, simd_fp_t::v5, arr_spec_t::s4));
    kernel.add_instr(simd_fp::zip2(simd_fp_t::v11, simd_fp_t::v6, simd_fp_t::v7, arr_spec_t::s4));

**Code with the FMAX instruction (relu_trans_primitive)**

.. code:: cpp

    // Load 4x4 block of A (input matrix)
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v0, gpr_t::x7, 0, neon_size_spec_t::q));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v1, gpr_t::x7, 0, neon_size_spec_t::q));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v2, gpr_t::x7, 0, neon_size_spec_t::q));
    kernel.add_instr(base::add(gpr_t::x7, gpr_t::x7, gpr_t::x2, 0, 0));
    kernel.add_instr(simd_fp::ldr(simd_fp_t::v3, gpr_t::x7, 0, neon_size_spec_t::q));

    // Compute ReLU
    kernel.add_instr(simd_fp::fmax(simd_fp_t::v0, simd_fp_t::v0, simd_fp_t::v31, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmax(simd_fp_t::v1, simd_fp_t::v1, simd_fp_t::v31, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmax(simd_fp_t::v2, simd_fp_t::v2, simd_fp_t::v31, arr_spec_t::s4));
    kernel.add_instr(simd_fp::fmax(simd_fp_t::v3, simd_fp_t::v3, simd_fp_t::v31, arr_spec_t::s4));

    // Transpose 4x4 block
    // TRN
    kernel.add_instr(simd_fp::trn1(simd_fp_t::v4, simd_fp_t::v0, simd_fp_t::v2, arr_spec_t::s4));
    kernel.add_instr(simd_fp::trn1(simd_fp_t::v5, simd_fp_t::v1, simd_fp_t::v3, arr_spec_t::s4));
    kernel.add_instr(simd_fp::trn2(simd_fp_t::v6, simd_fp_t::v0, simd_fp_t::v2, arr_spec_t::s4));
    kernel.add_instr(simd_fp::trn2(simd_fp_t::v7, simd_fp_t::v1, simd_fp_t::v3, arr_spec_t::s4));

    // ZIP
    kernel.add_instr(simd_fp::zip1(simd_fp_t::v8, simd_fp_t::v4, simd_fp_t::v5, arr_spec_t::s4));
    kernel.add_instr(simd_fp::zip1(simd_fp_t::v9, simd_fp_t::v6, simd_fp_t::v7, arr_spec_t::s4));

    kernel.add_instr(simd_fp::zip2(simd_fp_t::v10, simd_fp_t::v4, simd_fp_t::v5, arr_spec_t::s4));
    kernel.add_instr(simd_fp::zip2(simd_fp_t::v11, simd_fp_t::v6, simd_fp_t::v7, arr_spec_t::s4));

4.2.3.2 ReLU Primitive Benchmarks
---------------------------------------

We benchmarked the performance of our ReLU primitive for the given parameters (M=N=50, M=N=64, M=N=512 and M=N=2048), and obtained the following results:

**Benchmarking results for the relu primitives**

.. code:: text

    Running relu_primitive 50x50 benchmark
    Total time (s):                       3
    Total reps:                           19774014
    Total number of elements:             49435035000
    Total amount of processed data (GiB): 368.32
    Bandwidth (GiB/s)                     122.773
    --------------------------------------------------
    Running relu_primitive 64x64 benchmark
    Total time (s):                       3
    Total reps:                           12192431
    Total number of elements:             49940197376
    Total amount of processed data (GiB): 372.083
    Bandwidth (GiB/s)                     124.028
    --------------------------------------------------
    Running relu_primitive 512x512 benchmark
    Total time (s):                       3.00001
    Total reps:                           179693
    Total number of elements:             47105441792
    Total amount of processed data (GiB): 350.963
    Bandwidth (GiB/s)                     116.987
    --------------------------------------------------
    Running relu_primitive 2048x2048 benchmark
    Total time (s):                       3.00018
    Total reps:                           8874
    Total number of elements:             37220253696
    Total amount of processed data (GiB): 277.312
    Bandwidth (GiB/s)                     92.4321
    --------------------------------------------------
    Running relu_trans_primitive 50x50 benchmark
    Total time (s):                       3
    Total reps:                           16995447
    Total number of elements:             42488617500
    Total amount of processed data (GiB): 316.565
    Bandwidth (GiB/s)                     105.522
    --------------------------------------------------
    Running relu_trans_primitive 64x64 benchmark
    Total time (s):                       3
    Total reps:                           11039409
    Total number of elements:             45217419264
    Total amount of processed data (GiB): 336.896
    Bandwidth (GiB/s)                     112.299
    --------------------------------------------------
    Running relu_trans_primitive 512x512 benchmark
    Total time (s):                       3.00018
    Total reps:                           6131
    Total number of elements:             1607204864
    Total amount of processed data (GiB): 11.9746
    Bandwidth (GiB/s)                     3.9913
    --------------------------------------------------
    Running relu_trans_primitive 2048x2048 benchmark
    Total time (s):                       3.00082
    Total reps:                           347
    Total number of elements:             1455423488
    Total amount of processed data (GiB): 10.8438
    Bandwidth (GiB/s)                     3.6136
    --------------------------------------------------

The results match the pattern we saw for the zero and identity primitives. The transposition version is significantly slower than the non-transposition version, especially for larger matrices. Here as well, the 2048x2048 benchmark achieved worse results than the smaller matrices, both with and without transposition.