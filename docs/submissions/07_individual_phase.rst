##############################
7. Individual Phase
##############################

After following the given steps for the first couple of weeks, we were given the opportunity to explore the customizations of machine learning compilers.

**********************************
7.1 Our Pitch
**********************************

7.1.1 Roadmap
====================================

* `Unary Primitives (Tensor Processing Primitives, Table 1) <https://arxiv.org/pdf/2104.05755>`_

    * Square

    * Reciprocal

    * Increment & Decrement

* `Binary Primitives (Tensor Processing Primitives, Table 2) <https://arxiv.org/pdf/2104.05755>`_

    * Add & Sub

    * Mul & Div

    * Max & Min

* Optimizations

    * Optimize new primitives

    * Extend current optimizer (Optional): Dimension Fusion + Dimension Reordering

7.2.1 Risk Evaluation
====================================

* Risks

    * Incorporation of new primitives

    * Considering edge cases (division by zero)

    * Compatibility with our current code - adjustments resulting from this

    * Time management (considering we only have two weeks)

* Rewards / Outcomes

    * Regarding new primitives: enhanced / diversified compiler

    * Regarding optimization: high throughput over the board

**********************************
7.2 Sketch
**********************************

In this phase, we aim to extend our tensor compiler by implementing new primitives and subsequently optimizing their performance. 
Currently, the compiler is rather limited, as supports only a handful of processing primitives. 
To handle more diverse machine learning workloads, we plan to add selected unary and binary primitives, as presented in `Tensor Processing Primitives <https://arxiv.org/pdf/2104.05755>`_.

We will begin by adding a few unary primitives, specifically **Square**, **Reciprocal**, and **Increment & Decrement**.
Since our compiler does not yet support any binary primitives, our next step will be to implement **Add & Sub**, **Mul & Div** and **Max & Min**.

Regarding the implementation, we anticipate that some primitives may be more challenging than others.
For example, we need to account for edge cases, such as division by zero in the **Reciprocal** and **Div** operations.
However, as we have already developed some unary primitives, integrating the new ones into our current framework should be relatively straightfoward.

The situation is slightly different for the binary primitives. 
We do not have a direct starting point for these, but our plan is to integrate them similar to the existing main primitives. 
That said, as this approach is still untested, we may encounter compatibility issues that will need to be addressed along the way.

Importantly, we aim not only to integrate these implementations into our framework but also to achieve a high performance.
Therefore, we plan to optimize these primitives as much as possible. 

Given, that this is a relatively short-term project, we will need to assess our progress as we go.
If time allows, we would also like to further optimize our tensor operations by implementing **dimension fusion** and **dimension reordering**. 
Since we already have some other optimizations in place, integrating these should not result in major issues, although we do expect some challenges along the way.

**********************************
7.3 Implementation
**********************************

As suggested in our sketch, our plan was to implement the new functionalities in the following order:

1. Unary Primitives
2. Binary Primitives

7.3.1 Unary Primitives
====================================

For the unary primitives we were looking at **Square**, **Reciprocal** and **Div** operations.

7.3.1.1 Square Primitive
^^^^^^^^^^^^^^^^^^^^^^^^^

Our initial approach was to use instructions that we already had implemented.
Therefore, we started by using the ``FMLA`` instruction.
However, we quickly realized that the performance from multiplying two values and adding a zero value to it was not great. We decided to implement new instructions which would make our code more performant:

.. code:: cpp

    constexpr uint32_t fmulVec(simd_fp_t reg_dest,
                                       simd_fp_t reg_src1,
                                       simd_fp_t reg_src2,
                                       arr_spec_t arr_spec)
            {
                if (arr_spec != arr_spec_t::s2 && 
                    arr_spec != arr_spec_t::s4 &&
                    arr_spec != arr_spec_t::d2)
                {
                    throw std::invalid_argument("Invalid arrangement specifier");
                }

                uint32_t l_ins = 0x2E20DC00;

                // set destination register id
                l_ins |= (reg_dest & 0x1f);

                // set first source register id
                l_ins |= (reg_src1 & 0x1f) << 5;

                // set second source register id
                l_ins |= (reg_src2 & 0x1f) << 16;

                // set arrangement specifier
                l_ins |= (arr_spec & 0x40400000);

                return l_ins;
            }

This ``FMUL`` (vector) allowed us to multiply several elements simultaneously. 
For the cases where we needed to multiply single elements (``arr_spec_t::``) together, we implemented the following instruction:

.. code:: cpp

    constexpr uint32_t fmulScalar(simd_fp_t reg_dest,
                                          simd_fp_t reg_src1,
                                          simd_fp_t reg_src2,
                                          neon_size_spec_t size_spec)
            {
                if (size_spec != neon_size_spec_t::s && 
                    size_spec != neon_size_spec_t::d)
                {
                    throw std::invalid_argument("Invalid size specifier");
                }

                uint32_t l_ins = 0x1E200800;

                // set destination register id
                l_ins |= (reg_dest & 0x1f);

                // set first source register id
                l_ins |= (reg_src1 & 0x1f) << 5;

                // set second source register id
                l_ins |= (reg_src2 & 0x1f) << 16;

                // set size specifier
                l_ins |= (size_spec & 0x3) << 22;

                return l_ins;
            }
        }

These instructions allowed us to develop a kernel for the squared primitive. 
The approach for constructing this kernel was similar to the zero, ReLU or identity kernel. 

.. code:: cpp

    int mLoopIterations = m / 16;
    int mLoopRemainder = m % 16;

As a first step, we would calculate how many iterations we had to perform. 
With this number, we were then able to execute our main kernel accordingly:

.. code:: cpp

    ldp(v0, v1, x8, 0, q)
    ldp(v2, v3, x8, 32, q)

    fmulVec(v4, v0, v0, s4)
    fmulVec(v5, v1, v1, s4)
    fmulVec(v6, v2, v2, s4)
    fmulVec(v7, v3, v3, s4)

    stp(v4, v5, x9, 0, q)
    stp(v6, v7, x9, 32, q)

That means, in our main loop we would calculate 16 squared elements in one iteration. 
If there were no iterations left, we had to check if there would be a remainder: 

.. code:: cpp

    case 8:
        kernel.add_instr({
            ldp(v0, v1, x8, 0, q),
            fmulVec(v2, v0, v0, s4),
            fmulVec(v3, v1, v1, s4),
            stp(v2, v3, x9, 0, q)
        });
        break;
    case 9:
        kernel.add_instr({
            ldp(v0, v1, x8, 0, q),
            fmulVec(v2, v0, v0, s4),
            fmulVec(v3, v1, v1, s4),
            stp(v2, v3, x9, 0, q),

            ldr(v4, x8, 32, s),
            fmulScalar(v5, v4, v4, s),
            str(v5, x9, 32, s)
        });
        break;

We had to calculate the remainder for all of our 15 cases, in order to guarantee a correctly functioning kernel. 
After implementing the kernel, we also verified its correctness for different configurations:

.. code:: cpp

    uint32_t M = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    uint32_t N = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    test_square_primitive(M, N);

In order to be universally usable, we have also implemented a transposition square kernel. 
The implementation for this kernel was simple, as we could reuse the ``ReLU`` kernel and replace the ReLU operation with the square operation: 

.. code:: cpp

    // Load 4x4 block of A (input matrix)
    ldr(v0, x7, 0, q)
    add(x7, x7, x2, 0, 0)
    ldr(v1, x7, 0, q)
    add(x7, x7, x2, 0, 0)
    ldr(v2, x7, 0, q)
    add(x7, x7, x2, 0, 0)
    ldr(v3, x7, 0, q)

    // Square values
    fmulVec(v0, v0, v0, s4)
    fmulVec(v1, v1, v1, s4)
    fmulVec(v2, v2, v2, s4)
    fmulVec(v3, v3, v3, s4)

    // Transpose 4x4 block
    // TRN
    trn1(v4, v0, v2, s4)
    trn1(v5, v1, v3, s4)
    trn2(v6, v0, v2, s4)
    trn2(v7, v1, v3, s4)

    // ZIP
    zip1(v8, v4, v5, s4)
    zip1(v9, v6, v7, s4)

    zip2(v10, v4, v5, s4)
    zip2(v11, v6, v7, s4)

    // Store 4x4 Block of B
    str(v8, x8, 0, q)
    add(x8, x8, x3, 0, 0)
    str(v9, x8, 0, q)
    add(x8, x8, x3, 0, 0)
    str(v10, x8, 0, q)
    add(x8, x8, x3, 0, 0)
    str(v11, x8, 0, q)

However, that also meant we were limited to a ``4x4`` kernel, which would reduce our overall performance. 
For the transposition kernel, we did not implement any further optimizations. 

On the other hand, for the normal squared kernel we enhanced our initial dimension size from ``M=8`` to ``M=16``.

Lastly, we performed benchmarks similar to those of the other unary kernels: 

.. code:: text

    --------------------------------------------------
    Running square_primitive 50x50 benchmark
    Total time (s):                       3
    Total reps:                           19109506
    Total floating point operations:      47773765000
    Estimated GFLOPS/sec:                 15.9246
    --------------------------------------------------
    Running square_primitive 64x64 benchmark
    Total time (s):                       3
    Total reps:                           13569270
    Total floating point operations:      55579729920
    Estimated GFLOPS/sec:                 18.5266
    --------------------------------------------------
    Running square_primitive 512x512 benchmark
    Total time (s):                       3.00001
    Total reps:                           175397
    Total floating point operations:      45979271168
    Estimated GFLOPS/sec:                 15.3264
    --------------------------------------------------
    Running square_primitive 2048x2048 benchmark
    Total time (s):                       3.00007
    Total reps:                           9832
    Total floating point operations:      41238396928
    Estimated GFLOPS/sec:                 13.7458
    --------------------------------------------------

.. code:: text 

    Running square_trans_primitive 50x50 benchmark
    Total time (s):                       3
    Total reps:                           17201142
    Total floating point operations:      43002855000
    Estimated GFLOPS/sec:                 14.3343
    --------------------------------------------------
    Running square_trans_primitive 64x64 benchmark
    Total time (s):                       3
    Total reps:                           10953385
    Total floating point operations:      44865064960
    Estimated GFLOPS/sec:                 14.955
    --------------------------------------------------
    Running square_trans_primitive 512x512 benchmark
    Total time (s):                       3.00041
    Total reps:                           6112
    Total floating point operations:      1602224128
    Estimated GFLOPS/sec:                 0.534002
    --------------------------------------------------
    Running square_trans_primitive 2048x2048 benchmark
    Total time (s):                       3.00258
    Total reps:                           342
    Total floating point operations:      1434451968
    Estimated GFLOPS/sec:                 0.47774
    --------------------------------------------------

This time we were measuring the throughput of our kernel, differently to the ``zero``, ``identity``, and ``ReLU`` kernel, where we were measuring the data transfer rate.

7.3.1.2 Reciprocal Primitive
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The next primitive we implemented is the ``reciprocal`` operation, which computes ``1.0 / x`` for all input values ``x``.
For this, the AArch64 ISA already provides two instructions ``FRECPE`` and ``FRECPS``. ``FRECPE`` is the ``floating point reciprocal compute estimate`` instruction, which computes a first estimate of ``1.0 / x``. However, this estimate is generally not good enough for 32-bit floating point precision. To solve this, we can utilize ``FRECPS`` (``floating point reciprocal compute step``) iteratively, which improves the accuracy of the previously calculated estimate. We decided to perform only one step, as this already satisfied our used 32-bit floating point precision.

**FRECPE instruction generation**

.. code:: cpp

    constexpr uint32_t frecpeVec(simd_fp_t reg_dest,
                                 simd_fp_t reg_src,
                                 arr_spec_t arr_spec)
    {
        u_int32_t l_ins = 0xEA1D800;
        // set destination register id - Rd
        l_ins |= (reg_dest & 0x1f);
        // set source register id - Rn
        l_ins |= (reg_src & 0x1f) << 5;
        // set arrangement specifier
        l_ins |= (arr_spec & 0x40400000);
        return l_ins;
    }

    constexpr uint32_t frecpeScalar(simd_fp_t reg_dest,
                                    simd_fp_t reg_src,
                                    size_spec_t size_spec)
    {
        if (size_spec != neon_size_spec_t::s && 
            size_spec != neon_size_spec_t::d)
        {
                throw std::invalid_argument("Invalid size specifier");
        }
        u_int32_t l_ins = 0x5EA1D800;
        // set destination register id - Rd
        l_ins |= (reg_dest & 0x1f);
        // set source register id - Rn
        l_ins |= (reg_src & 0x1f) << 5;
        // set size specifier
        l_ins |= (size_spec & 0x1) << 22;
        return l_ins;
    }

**FRECPS instruction generation**

.. code:: cpp

    constexpr uint32_t frecpsVec(simd_fp_t reg_dest,
                                 simd_fp_t reg_src1,
                                 simd_fp_t reg_src2,
                                 arr_spec_t arr_spec)
    {
        u_int32_t l_ins = 0xE20FC00;
        // set destination register id - Rd
        l_ins |= (reg_dest & 0x1f);
        // set first source register id
        l_ins |= (reg_src1 & 0x1f) << 5;
        // set second source register id
        l_ins |= (reg_src2 & 0x1f) << 16;
        // set size specifier
        l_ins |= (arr_spec & 0x40400000);
        return l_ins;
    }

    constexpr uint32_t frecpsScalar(simd_fp_t reg_dest,
                                    simd_fp_t reg_src1,
                                    simd_fp_t reg_src2,
                                    size_spec_t size_spec)
    {

        if (size_spec != neon_size_spec_t::s && 
            size_spec != neon_size_spec_t::d)
        {
                throw std::invalid_argument("Invalid size specifier");
        }
        u_int32_t l_ins = 0x5E20FC00;
        // set destination register id - Rd
        l_ins |= (reg_dest & 0x1f);
        // set first source register id
        l_ins |= (reg_src1 & 0x1f) << 5;
        // set second source register id
        l_ins |= (reg_src2 & 0x1f) << 16;
        // set size specifier
        l_ins |= (size_spec & 0x1) << 22;
        return l_ins;
    }

To compute the reciprocal, we also needed the ``FMUL`` instruction which we implemented in the previous section. A full reciprocal computation looks like this:


.. code:: asm

    frecpe  v0.4s, v1.4s        // Estimate reciprocal of v1 and save to v0
    frecps  v2.4s, v1.4s, v0.4s // Refine reciprocal
    fmul    v0.4s, v0.4s, v2.4s // Apply refinement -> v0 now has better estimate

With these instructions, we began implementing the new kernel. Structurally it is identical to the square primitive. We simply replaced the calculations with the new instructions:

**Reciprocal primitive: main loop**

.. code:: cpp

    kernel.add_instr({
        // load 16 elements from A
        ldp(v0, v1, x8, 0, q),
        ldp(v2, v3, x8, 32, q),

        frecpeVec(v4, v0, s4),
        frecpsVec(v10, v0, v4, s4),
        fmulVec(v4, v4, v10, s4),

        frecpeVec(v5, v1, s4),
        frecpsVec(v10, v1, v5, s4),
        fmulVec(v5, v5, v10, s4),

        frecpeVec(v6, v2, s4),
        frecpsVec(v10, v2, v6, s4),
        fmulVec(v6, v6, v10, s4),

        frecpeVec(v7, v3, s4),
        frecpsVec(v10, v3, v7, s4),
        fmulVec(v7, v7, v10, s4),

        // store 16 elements to B
        stp(v4, v5, x9, 0, q),
        stp(v6, v7, x9, 32, q),

        // jump by 16 rows
        add(x8, x8, 16*4, 0),
        add(x9, x9, 16*4, 0),

        // decrement m loop counter
        sub(x7, x7, 1, 0),
    });

**Reciprocal transposition primitive: main loop**

.. code:: cpp

    kernel.add_instr({
        // working pointer for A and B
        mov(x7, x4),
        mov(x8, x5),
        
        // Load 4x4 block of A (input matrix)
        ldr(v0, x7, 0, q),
        add(x7, x7, x2, 0, 0),
        ldr(v1, x7, 0, q),
        add(x7, x7, x2, 0, 0),
        ldr(v2, x7, 0, q),
        add(x7, x7, x2, 0, 0),
        ldr(v3, x7, 0, q),

        frecpeVec(v16, v0, s4),
        frecpsVec(v17, v0, v16, s4),
        fmulVec(v0, v16, v17, s4),

        frecpeVec(v16, v1, s4),
        frecpsVec(v17, v1, v16, s4),
        fmulVec(v1, v16, v17, s4),

        frecpeVec(v16, v2, s4),
        frecpsVec(v17, v2, v16, s4),
        fmulVec(v2, v16, v17, s4),

        frecpeVec(v16, v3, s4),
        frecpsVec(v17, v3, v16, s4),
        fmulVec(v3, v16, v17, s4),

        // Transpose 4x4 block
        // TRN
        trn1(v4, v0, v2, s4),
        trn1(v5, v1, v3, s4),
        trn2(v6, v0, v2, s4),
        trn2(v7, v1, v3, s4),

        // ZIP
        zip1(v8, v4, v5, s4),
        zip1(v9, v6, v7, s4),

        zip2(v10, v4, v5, s4),
        zip2(v11, v6, v7, s4),

        // Store 4x4 Block of B
        str(v8, x8, 0, q),
        add(x8, x8, x3, 0, 0),
        str(v9, x8, 0, q),
        add(x8, x8, x3, 0, 0),
        str(v10, x8, 0, q),
        add(x8, x8, x3, 0, 0),
        str(v11, x8, 0, q),

        // Matrix A next 4 rows
        add(x4, x4, x25, 0, 0),

        // Matrix B next 4 columns
        add(x5, x5, x27, 0, 0),
        
        // decrement m loop counter
        sub(x6, x6, 1, 0)
    });

7.3.1.3 Increment and Decrement Primitive
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The last unary primitives that we wanted to implement were the increment and decrement operations.

Similar to the other primitives, we had to first implement a few new instructions. 
Instructions that were directly needed for these primitives are ``FADD`` and ``FSUB``. 
To fully utilize these instructions, we were implementing both a scalar and a vector version for these instructions:

.. code:: cpp

    constexpr uint32_t faddVec(simd_fp_t reg_dest,
                               simd_fp_t reg_src1,
                               simd_fp_t reg_src2,
                               arr_spec_t arr_spec)
    {
        if (arr_spec != arr_spec_t::s2 && 
            arr_spec != arr_spec_t::s4 &&
            arr_spec != arr_spec_t::d2)
        {
            throw std::invalid_argument("Invalid arrangement specifier");
        }
        uint32_t l_ins = 0xE20D400;
        // set destination register id
        l_ins |= (reg_dest & 0x1f);
        // set first source register id
        l_ins |= (reg_src1 & 0x1f) << 5;
        // set second source register id
        l_ins |= (reg_src2 & 0x1f) << 16;
        // set arrangement specifier
        l_ins |= (arr_spec & 0x40400000);
        return l_ins;
    }

    constexpr uint32_t faddScalar(simd_fp_t reg_dest,
                                simd_fp_t reg_src1,
                                simd_fp_t reg_src2,
                                neon_size_spec_t size_spec)
    {
        if (size_spec != neon_size_spec_t::s && 
            size_spec != neon_size_spec_t::d)
        {
            throw std::invalid_argument("Invalid size specifier");
        }
        uint32_t l_ins = 0x1E202800;
        // set destination register id
        l_ins |= (reg_dest & 0x1f);
        // set first source register id
        l_ins |= (reg_src1 & 0x1f) << 5;
        // set second source register id
        l_ins |= (reg_src2 & 0x1f) << 16;
        // set size specifier
        l_ins |= (size_spec & 0x3) << 22;
        return l_ins;
    }

Beside these instructions, we needed to move the value ``1`` into a Neon register. 
That meant, we had to also implement the ``FMOV`` instruction. 
Implementing the ``FMOV`` instruction has been slightly different to those of other implemenations. 
The main reason for this special behavior is the split of the 8-bit immediate into 1 signed bit, a 3-bit exponent and a 4-bit precision part. 
This unique characteristic changes the use of these 8-bits slightly. 

For example, we have looked at different scenarios for moving a floating point value:

.. list-table:: Different Floating-Point Movements
   :widths: 20 10 10 10 10 10 10 10 10
   :header-rows: 1

   * - FP-Number
     - Bit-18
     - Bit-17
     - Bit-16
     - Bit-9
     - Bit-8
     - Bit-7
     - Bit-6
     - Bit-5
   * - **1.0f**
     - 0
     - 1
     - 1
     - 1
     - 0
     - 0
     - 0
     - 0
   * - **2.0f**
     - 0
     - 0
     - 0
     - 0
     - 0
     - 0
     - 0
     - 0
   * - **3.0f**
     - 0
     - 0
     - 0
     - 0
     - 1
     - 0
     - 0
     - 0
   * - **7.0f**
     - 0
     - 0
     - 0
     - 1
     - 1
     - 1
     - 0
     - 0
   * - **18.0f**
     - 0
     - 0
     - 1
     - 1
     - 0
     - 0
     - 1
     - 0
   * - **31.0f**
     - 0
     - 0
     - 1
     - 1
     - 1
     - 1
     - 1
     - 1
   * - **\-31.0f**
     - 1
     - 0
     - 1
     - 1
     - 1
     - 1
     - 1
     - 1

Looking at these examples we were able to find some special cases (e.g. ``1``), but also patterns, that we were trying apply to our implementation:

.. code:: cpp

    constexpr uint32_t fmovVec(simd_fp_t reg_dest,
                               int32_t imm8,
                               arr_spec_t arr_spec)
    {
        if (arr_spec != arr_spec_t::s2 && 
            arr_spec != arr_spec_t::s4 && 
            arr_spec != arr_spec_t::d2)
        {
            throw std::invalid_argument("Invalid arrangement specifier");
        }
        int32_t l_ins = 0xF00F400;
        // set destination register id
        l_ins |= (reg_dest & 0x1f);

        if (imm8 > 31 || imm8 < -31)
        {
            throw std::invalid_argument("Invalid immediate (allowed range: -31, 31)");
        }
        if (imm8 < 0)
        {
            l_ins |= (0x1) << 18;
            imm8 *= -1;
        }

        // immediate bits
        if (imm8 == 1)
        {
            l_ins |= (0x3) << 16;
            l_ins |= (0x1) << 9;
        }
        else if (imm8 == 2)
        {
        }
        else if (imm8 == 3)
        {
            l_ins |= (0x1) << 8;
        }
        else if (imm8 < 8)
        {
            l_ins |= (imm8 & 0x7) << 7;
        }
        else
        {
            l_ins |= (0x1) << 16;

            if (imm8 > 8 && imm8 < 16)
            {
                l_ins |= (imm8 & 0x7) << 6;
            }
            else if (imm8 > 16)
            {
                l_ins |= (imm8 & 0x1f) << 5;
            }
        }

        // set arrangement specifier
        if (arr_spec == arr_spec_t::s4)
        {
            l_ins |= (0x1) << 30;
        }
        else if (arr_spec == arr_spec_t::d2)
        {
            l_ins |= (0x1) << 29;
            l_ins |= (0x1) << 30;
        }

        return l_ins;
    }

In practice we would need the ``FMOV`` instruction to transfer the ``1.0f`` into a vector, in order to be able to execute vector addition and subtraction operations. 

After implementing the instructions we simply took our ``square`` kernel and replaced all multiplication operations with a ``FADD`` or a ``FSUB`` operation:

.. code:: cpp

    // Set register with value 1
    fmovVec(v19, 1, s4)

    ...

    // load 16 elements from A
    ldp(v0, v1, x8, 0, q)
    ldp(v2, v3, x8, 32, q)

    faddVec(v4, v0, v19, s4)
    faddVec(v5, v1, v19, s4)
    faddVec(v6, v2, v19, s4)
    faddVec(v7, v3, v19, s4)

    // store 16 elements to B
    stp(v4, v5, x9, 0, q)
    stp(v6, v7, x9, 32, q)

    // jump by 16 rows
    add(x8, x8, 16*4, 0)
    add(x9, x9, 16*4, 0)

    // decrement m loop counter
    sub(x7, x7, 1, 0)

After implementing both the ``increment`` and ``decrement`` kernel, we also implemented their transposed versions.

7.3.1.4 Integration in Framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
