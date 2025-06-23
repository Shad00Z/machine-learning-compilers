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

Our initial approach was to use instructions, that we already had implemented.
Thea meant, we first started by using the ``FMLA`` instruction.
However, we quickly realized that the performance from multiplying two values and adding a zero value to it, was not the best.
Therefore, we needed to implement new instructions.

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

This ``FMUL`` (vector) allowed us to multiply several elements simultaneous. 
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

These instructions allowed us, to develop a kernel for the squared primitive. 
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
If there were no iterations left, we had to look, if there would be a remainder: 

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
After implementing the kernel, we also verified its correctness, for different configurations:

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
