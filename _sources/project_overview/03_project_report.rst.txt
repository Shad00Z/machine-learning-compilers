Project Report
===============

In the first two weeks of our project, we focused on ARM AArch64 assembly to build a solid foundation for understanding machine learning compilers. During the first week, we examined the assembly output of a simple C program compiled with both **GCC** and **Clang** to explore the differences in code generation and function call conventions. 

This initial exploration helped us become familiar with compiler behavior, instruction-level representations, and low-level debugging tools such as **GDB**. 

Further details about the specific steps and tasks can be found in the :ref:`assembly section<Assembly>` of our project documentation.

In the second week, we began writing assembly programs from scratch using only base instructions. Specifically, we reimplemented two simple C functions in AArch64 assembly to better understand data movements and control flow at the instruction level. 

After successfully replicating the functionality of the C programs, we developed microbenchmarks to evaluate the **throughput** and **latency** of key instructions such as ``ADD`` and ``MUL``. These benchmarks helped us gain insight into the performance characteristics of modern ARM processors and how instruction-level behavior can impact overall computation speed.

Further details about the specific steps and tasks can be found in the :ref:`base section<Base>` of our project documentation.
