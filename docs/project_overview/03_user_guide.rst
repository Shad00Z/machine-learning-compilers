.. _user-guide:

#############################
User Guide
#############################

.. note::

    This guide is for macOS and Linux only. As of now, we have not tested any support for Windows.

In order to use our tensor compiler, you will have to build from source.
Before we begin with instructions on how to do that, we will first guide you through all the dependencies you need.

*****************************
Installing Dependencies
*****************************

To build this project from source, you will need a few tools installed:

Git
===================================

To pull the code from GitHub, `git <https://git-scm.com/downloads>`_ is required.  
Alternatively, you can download the repository as a `.zip` file directly from GitHub.

C++ Compiler with OpenMP Support
===================================

You need a C++ compiler that supports `OpenMP <https://www.openmp.org>`_, as our tensor compiler uses parallel execution.

**On Linux (e.g., Ubuntu/Debian):**

.. code:: bash

    sudo apt install build-essential libomp-dev

This installs the GNU Compiler Collection (GCC), `make`, and the OpenMP development library.

**On macOS:**

While macOS comes with a default C++ compiler (Apple Clang), it **does not support OpenMP**. We recommend using the LLVM toolchain via `Homebrew <https://brew.sh>`_:

.. code:: bash

    brew install llvm libomp

.. note::

    If you are already using Homebrew's GCC, that will work also, but you will have to adjust our Makefile.
    It automatically switches to **LLVM** on macOS, because we could not figure out a stable way to distinguish
    between the default Apple Clang and Homebrew's GCC.

Make (Build System)
===================================

Our project uses Makefiles, so you will also need the ``make`` tool.

**On Linux:**

``make`` is typically included with ``build-essential``. If not, install it with:

.. code:: bash

    sudo apt install make

**On macOS:**

Install the Command Line Tools, which include ``make``. 
If you have already used Homebrew before, this will likely already be installed.
You can test this by running:

.. code:: bash

    make --v

If you get an error saying that ``make`` could not be found, run

.. code:: bash

    xcode-select --install

and try again.

*****************************
Building from Source
*****************************

The first to building from source is to actually obtain the sources, which you can do using **git**:

.. code:: bash

    git clone https://github.com/Shad00Z/machine-learning-compilers.git
    cd machine-learning-compilers

Next, you have a few options to build the project, which are all based on running ``make`` from inside the
``machine-learning-compilers`` directory.

Default Installation
===================================

The default installation packages our project as a **static library** and also builds test executables. 
All you need to do here is to invoke ``make`` from inside the ``machine-learning-compilers`` directory.
A ``lib`` folder will be created, containing a ``libmlc.a`` static library file.

You may also execute our unit tests to ensure everything works on your system. 
Should you experience any test failures, please open an issue containing the console log in our GitHub repository.

**On Linux:**

.. code:: bash

    ./build/linux/tests/unit-tests 

**On macOS (ARM64):**

.. code:: bash

    ./build/macOS-arm64/tests/unit-tests

**On macOS (Intel):**

.. code:: bash

    ./build/macOS-x86_64/tests/unit-tests

Library-only Installation
===================================

You may skip building our tests, by invoking either

.. code:: bash

    make static-library

to build a static library, or

.. code:: bash

    make shared-library

to build a shared library. 

In either case, the library file can be found inside the ``lib`` folder at the top of the repository.
We recommend building a static library, as we have not tested the usage of a shared library.

Executing Benchmarks
===================================

To build the benchmarks, simply run

.. code:: bash

    make benchmarks

To actually execute benchmarks, the ``benchmarks`` executable is used.
However, for this executable to do anything, you need to specify which benchmarks you would like to run.
A list of available benchmarks can be displayed using:

**On Linux:**

.. code:: bash

    ./build/linux/benchmarks help

**On macOS (ARM64):**

.. code:: bash

    ./build/macOS-arm64/benchmarks help

**On macOS (Intel):**

.. code:: bash

    ./build/macOS-x86_64/benchmarks help

For example if you wish to execute the **matmul** and **sigmoid** benchmarks on Linux, you would need to run

.. code:: bash

    ./build/linux/benchmarks matmul sigmoid

*****************************
Using our Tensor Compiler
*****************************

We believe that the best way to learn is by exploring examples.
So instead of explaining all the details here, we have provided several code samples in the ``example_usage`` directory at the root of our GitHub repository.
These examples demonstrate how to use the core functionality of our tensor compiler and should help you get started quickly.
If you would like to explore more advanced features, be sure to check out the **API** section.

Compiling and Running Examples
==============================

If you have installed all necessary dependencies, you can compile and run the examples using the shell scripts we have provided:

**On Linux:**

.. code:: bash

    ./run_linux.sh

**On macOS:**

.. code:: bash

    ./run_macos.sh

These scripts are also useful references, because they show how to compile and link your own code against our static library.

Compilation Example (Linux)
===========================

Here is the command we use in the Linux shell script to compile and link the ``EinsumExample`` file:

.. code-block:: bash

    g++ \
        -std=c++20 \
        -o EinsumExample \
        EinsumExample.cpp \
        -I../include \
        -L../lib \
        -lmlc \
        -fopenmp \
        -lomp

Explanation
===========================

- The first four lines are standard for compiling a C++ file with C++20 support.
- The ``-I../include`` flag tells the compiler where to find the public headers for our tensor compiler.
- The ``-L../lib`` flag tells the linker where to look for the compiled static library.
- The ``-lmlc`` flag links the binary against ``libmlc.a``, which is how your project accesses the functionality of our compiler.
- The last two flags enable **OpenMP** support. These assume that OpenMP is installed and accessible through your system paths.

If you are integrating the tensor compiler into your own project, you will need to add ``machine-learning-compilers/include`` to your compiler's include paths, and ``machine-learning-compilers/lib`` to your linker paths.

.. note::

   If you are using an IDE, make sure to add the include directory to your project's configuration. This will ensure that features like autocompletion work correctly.