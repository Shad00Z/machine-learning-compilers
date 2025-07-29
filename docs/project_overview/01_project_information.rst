#############################
Introduction
#############################

This project discusses the development of a **Machine Learning Compiler**. 
Specifically, we explain how we developed a domain-specific, primitive-based tensor compiler.
If that sounds like a random string of technical buzzwords to you, don't worry. 
In the following sections, we will break down what a tensor compiler is, why it's useful, and how you can use our implementation.

**************************************
What are Machine Learning Compilers?
**************************************

Machine Learning Compilers are tools that translate high-level machine learning models or tensor operations into low-level machine code that can be executed efficiently on hardware.
Compilers which translate tensor operations are referred to as **tensor compilers**.
A practical application of a tensor compiler is to close the gap between machine learning frameworks (like TensorFlow or PyTorch) and the underlying hardware.
In this project, we focus on developing a tensor compiler that operates on a set of primitive operations, which are the building blocks for more complex tensor computations.
If you wish to learn more about tensor processing primitives, we recommend reading `this paper <https://arxiv.org/pdf/2104.05755>`_.

While the main task of a tensor compiler is to enable tensor operations to be run on hardware, there are several goals that a good tensor compiler should achieve:

- **High throughput**: Minimize the average execution time for many independent evaluations of the same tensor expression.
- **Low latency**: Minimize the time it takes to execute a single tensor operation.
- **Short compilation time**: Minimize the time it takes for the compiler to generate executable code.
- **Flexibility**: Support a wide range of tensor expressions.
- **Portability**: Support for high performance execution on different hardware platforms.

In our implementation, we decided to focus on the first four goals. The tensor compiler we developed is specifically designed for the **A64 Instruction Set Architecture**, which is used in **ARM-based CPUs**.
While this might be a large constraint in portability, it allowed us to focus on the specific features and optimizations that are relevant for the chosen architecture.

*************************************
Project Background
*************************************

Having explained the motivation behind the project, we would now like to provide some further background information.
Everything you'll encounter in this guide and the accompanying `GitHub repository <https://github.com/Shad00Z/machine-learning-compilers>`_ was developed as part of the **Machine Learning Compilers** class at the `University of Jena <https://www.uni-jena.de/en>`_ during the summer semester of 2025.
The codebase was created through a series of weekly assignments which we completed throughout the semester.  
Alongside the class sessions, we were also provided with `an online book <https://scalable.uni-jena.de/opt/pbtc/index.html>`_ that outlines the core concepts behind this project. 
If you are more interested in the theoretical background of the project, we recommend checking out the book first.

*************************************
Project Structure
*************************************

The documentation you are currently reading is part of the `docs` directory in the `GitHub repository <https://github.com/Shad00Z/machine-learning-compilers>`_. 
It is organized into three main sections:

#. **Project Overview**: Provides background on the project, its structure, and guidance on how to use the codebase.
#. **Submissions**: Contains the individual assignments completed throughout the semester, along with explanations of the tasks and the code we wrote to solve them.
#. **API**: Automatically generated from the codebase using `Doxygen <https://www.doxygen.nl/index.html>`_, this section offers detailed information about the classes, functions, and modules we implemented.

The structure of the codebase itself is more complex, so we'll only provide a brief overview here.

The main code can be found in the `src` directory, which contains the implementation of the tensor compiler.  
Within `src`, you'll also find a `submissions` directory containing the code from the initial assignments we completed before beginning to work on the main compiler implementation.  
We recommend starting with the **API** section if you're looking to understand the code structure and class design in detail.

Next to the `src` directory is a `tests` directory, which includes unit tests used to verify the correctness of our implementation.  
You'll also find the `docs` directory, which contains the documentation you're currently reading.
Finally, the `benchmarks` directory includes various results from performance evaluations of our compiler.

**************************************
Where to Go Next?
**************************************

If you wish to learn more about the project, we recommend starting with the :ref:`project-report` section.
This section provides an overview of the project, including the tasks we completed, the code we wrote, and the results we achieved.
It also includes links to the individual assignments we completed throughout the semester, which provide more detailed information about the specific tasks and our solutions.

If you wish to use the tensor compiler yourself, you can find instructions in our :ref:`user-guide`.

Lastly, if you are interested in building the documentation yourself, you should check out the :ref:`documentation-setup` section.