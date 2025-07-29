.. _project-report:

Project Report
===============

In the first two weeks of our project, we focused on ARM AArch64 assembly to build a solid foundation for understanding machine learning compilers. 
During the first week, we examined the assembly output of a simple C program compiled with both **GCC** and **Clang** to explore the differences in code generation and function call conventions. 
This initial exploration helped us become familiar with compiler behavior, instruction-level representations, and low-level debugging tools such as **GDB**. 
Further details about the specific steps and tasks can be found in the :ref:`assembly section<Assembly>` of our project documentation.

In the second week, we began writing assembly programs from scratch using only base instructions. 
Specifically, we reimplemented two simple C functions in AArch64 assembly to better understand data movements and control flow at the instruction level. 
After successfully replicating the functionality of the C programs, we developed microbenchmarks to evaluate the **throughput** and **latency** of key instructions such as ``ADD`` and ``MUL``. 
These benchmarks helped us gain insight into the performance characteristics of modern ARM processors and how instruction-level behavior can impact overall computation speed.
Further details about the specific steps and tasks can be found in the :ref:`base section<Base>` of our project documentation.

After spending the first two weeks experimenting with base instructions and writing simple assembly programs, we advanced to working with **Neon** (**Advanced SIMD**) instructions. 
In the following weeks, we explored the performance characteristics of Neon operations, an essential step toward mastering the fundamentals required for building our own tensor compiler.

We began by benchmarking the :ref:`throughput and latency<3.1-throughput-latency>` of the ``FMLA`` and ``FMADD`` instructions. 
This helped us understand the significance of instruction-level parallelism and how much instruction ordering and data dependencies can impact performance. 
After these initial experiments, we shifted our focus toward understanding the role of **microkernels**. 
We explored key design considerations, like data reuse, register allocation, and memory access patterns, and conducted our first experiments with optimizing :ref:`microkernels<3.2-microkernel>` for performance. 

In the following week, we extended our microkernel by wrapping it in **loops** to support larger matrix dimensions and improve the performance.
Starting from our base kernel of ``16x6x1``, we progressively scaled it to handle matrices of size ``64x48x64``. 
This allowed us to reach the architectural performance limits of a M4 Chip. 
Further implementation details can be found in the :ref:`loops section <3.3 Loops>` of our project documentation. 

After exploring ideal kernel sizes aligned with our vector register widths, we also experimented with cases where the ``M`` dimension is not a multiple of 4 or 16. 
In these scenarios, special handling was required to maintain correctness and efficiency. 
We implemented and optimized dedicated kernels for two such cases, which are documented in detail in the :ref:`SIMD Lanes <3.4 SIMD>` section of our documentation. 
In the same week, we also explored the impact of **accumulator block** shapes for performance reasons. 
Specifically, we implemented a microkernel for computing ``C+=AB`` with dimensions ``M=64``, ``N=64``, and ``K=64``. 
This required adapting our existing ``matmul_64_48_64`` kernel to support the extended ``N`` dimension. 
The details and benchmarking results of this extension are documented in the :ref:`Accumulator Block Shapes<3.5 Accumulator Block Shapes>` section. 

After implementing and optimizing standard matrix multiplication kernels, we extended our work to support :ref:`batch-reduce GEMM<3.6 Batch-Reduce GEMM>` operations. 
We reused and adapted our existing microkernels to handle **batches of matrices** efficiently. 

The last part of the Neon section was to explore how to **transpose** matrices using Neon assembly. 
Our goal was to handle a ``8x8`` matrix, which we handled by dividing it into four ``4x4`` submatrices. 
More details and code can be found in the :ref:`transposition section<3.7 Transposition>` of our documentation. 

In week 4, we turned our attention to code generation.
The idea was to generate the previously implemented Neon assembly kernels using C++ during runtime.
Starting with a rather simple example, we began by implementing a ``matmul_16_6_k`` :ref:`microkernel <brgemm-microkernel>`.
For this, we learned how to generate assembly instructions by setting each bit manually, writing the 32-bit instructions to previously allocated memory and lastly making that memory executable.

The next task was to extend the ``matmul_16_6_k`` by generating loops over the ``M`` and ``N`` dimensions, resulting in a **GEMM** kernel.
This also involved writing a backend entry point for kernel generation and unit tests for verification.
After verifying that the GEMM kernel generation worked as intended, we proceeded with implementing a **Batch-Reduce GEMM (BRGEMM)** kernel.
Here too, we verified our implementation using unit tests.
Lastly, we performed an extensive benchmark of the GEMM and BRGEMM kernels for different matrix dimensions.
In total, this benchmark took over 8 hours on the provided Apple M4 machine.
For more information, refer to :ref:`brgemm-primitive`.

In week 5, we extended our code generator by **Unary Primitives** of the form B:=op(A).
Similarly to the BRGEMM backend, we first implemented a new entry point which can be used to generate various unary primitive kernels.
The first unary primitive we implemented was the **Zero Primitive**, which sets all elements of the output tensor to zero.
Secondly, we implemented the **Identity Primitive** which copies all elements of the input tensor to the output tensor.
The complicated part here was to support transposition for arbitrary tensor sizes.
Lastly, we implemented an activation function commonly found in machine learning frameworks: the **ReLU Primitive**.
This operation sets all negative elements to zero and keeps positive elements as they are.
For all implemented unary operations we implemented unit tests and benchmarked the performance.
Further information can be found in :ref:`unary-primitives`.

In week 6 we received the task of developing a **Tensor Operation Backend**, see :ref:`tensor-op-backend`.
This backend not only **sets up and holds the kernel objects**,
but also **blocks the input and output tensors** and executes the kernels accordingly.
We started with a verification of all input parameters and then implemented an ``execute`` function,
which calls itself recursively to work through the nested sequential loops of an input tensor.
To end this week's task, we performed extensive benchmarks with various configuration parameters.

In the following week, we added support for **Shared Memory Parallelization**.
This meant that we had to check whether the input tensor contained any dimensions that should be executed in parallel.
If that was the case, we flattened all shared dimensions into one big iteration space and then parallelized it using **OpenMP**.
You can find more information :ref:`here <shared-memory-parallelization>`.

The second task of week 7 was to implement optimization passes, see :ref:`optimization-passes`.
First, we developed an intermediate representation of the tensors consisting of Dimension ``struct``'s.
Then, we applied **Primitive Identification**, **Dimension Splitting** and the **Parallelization of Sequential Dimensions** to the vectors of the Dimension ``struct``'s that represent each input tensor.

In week 8, we enhanced our tensor operation backend and our optimizer by supporting **Unary Operations**, such as permuting a tensor's dimensions.
Here, we first had to verify that all dimensions of such an operation are of type ``C`` and appear both in the input and the output tensor. Additionally, the second input tensor had to be automatically set to ``nullptr``, as unary operations only support one input.
Next, we had to implement new primitive identification and shared memory parallelization optimization passes for the unary operations and finally verify the correctness of our code against a reference implementation.
To view the results and implementations, visit our :ref:`detailed report <unary-operations>`.

In the final weekly submission of our project, we made important enhancements to the flexibility of our tensor compiler.
The first major task was the support for **Einsum Tree Expressions**.
Starting with a :ref:`string parser <einsum-parsing>` that creates an internal tree representation from string expressions of the form ``[...],[...]->[...]``, we also had to find a way to :ref:`lower the einsum tree to our tensor operation backend <einsum-lowering-subchapter>`.
Next, we had to ensure that our previously implemented :ref:`optimization passes <einsum-node-optimizations>` were applicable here too and lastly ensure that the tree could be :ref:`executed <einsum-execution>`. To validate the correctness and effectiveness of our implementation, we performed various benchmarks.
The detailed report on this task can be found in :ref:`einsum-lowering`

The second major task was to implement **Optimization Passes for Einsum Trees**.
These differ from the previous optimization which focused on the internals of a single tensor operation.
In this task, we looked at **Node Swapping**, **Dimension Reordering** and the **Insertion of Permutation Nodes**.
These tree optimizations come before the initialization of each tree node, and reshape the einsum tree to improve performance.
Refer to :ref:`einsum-tree-optimizations` to learn more about our implementations.

The last task was to propose a pitch and write a short sketch on the individual phase that would conclude this project.
To read our pitch and sketch, please take a look at :ref:`individual-phase`.

During the last two weeks, we had to fulfill the pitch that we had written in the prior week.
We used the first week of the individual phase to extend the flexibility of our tensor compiler by implementing various unary and binary primitives.
In the second week, we put our attention on the rather complex sigmoid activation function and added multiple implementations with different speeds and accuracies.
Lastly, we improved our optimizer by adding support for **Dimension Fusion**.