#############################
Introduction
#############################

This project discusses the development of a **Machine Learning Compiler**. 
In other words, we explain how we developed a domain-specific, primitive-based tensor compiler.
If that sounds like a random string of technical buzzwords to you, don't worry. 
In the following sections, we will break down what a tensor compiler is, why it's useful, and how you can use our implementation.

*************************************
Project Background
*************************************

Before we begin explaining the more technical topics, we'd like to provide some background on this project.
Everything you'll encounter in this guide and the accompanying GitHub repository was developed as part of the **Machine Learning Compilers** class at the `University of Jena <https://www.uni-jena.de/en>`_ during the summer semester of 2025.
The codebase was created through a series of weekly assignments we completed throughout the semester.  
Alongside the class sessions, we were also provided with `an online book <https://scalable.uni-jena.de/opt/pbtc/index.html>`_ that outlines the core concepts behind this project.

*************************************
Project Structure
*************************************


**************************************
What are Machine Learning Compilers?
**************************************

Machine learning compilers are tools that translate high-level machine learning models or tensor operations into low-level machine code that can be executed efficiently on hardware.
Compilers which translate tensor operations are referred to as **tensor compilers**.
A practical application of a tensor compiler is to close the gap between machine learning frameworks (like TensorFlow or PyTorch) and the underlying hardware.
In this project, we focus on developing a tensor compiler that operates on a set of primitive operations, which are the building blocks for more complex tensor computations.
If you wish to learn more about tensor processing primitives, we recommend reading `this paper <https://arxiv.org/pdf/2104.05755>`_.

While the main task of a tensor compiler is to enable tensor operations to be run on hardware, there are several goals that a good tensor compiler should achieve:

- **High throughput**: The compiler should generate code that can process large amounts of data efficiently.
- **Low latency**: The compiler should minimize the time it takes to execute tensor operations.
- **Short compilation time**: The compiler should be able to generate code quickly.
- **Flexibility**: The compiler should be able to handle a wide range of tensor operations and hardware architectures.
- **Portability**: The compiler should be able to generate code that can run on different hardware platforms without requiring significant modifications.

In our implementation, we decided to only focus on the first four goals. The tensor compiler we developed is specifically designed for the **A64 Instruction Set Architecture**, which is used in ARM-based processors.