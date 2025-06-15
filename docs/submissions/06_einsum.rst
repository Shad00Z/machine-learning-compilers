##############################
6. Einsum Trees
##############################

After creating our backend with the implementation of tensor objects, we now want to perform 
tensor computations on larger expressions in the form einsum trees. 

**********************************
6.1 Lowering
**********************************

Our first step is to accept expressions of the form ``[...],[...]->[...]``. 

6.1.1 Expression Parsing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To transform our string expression into something connected, we implemented an ``EinsumNode``. 

An ``EinsumNode`` holds information about a single node from the expression e.g. ``[...]``:

.. literalinclude:: ../../src/einsum/EinsumNode.h
    :language: cpp
    :lines: 13-38
    :lineno-match:
    :caption: information that an ``EinsumNode`` holds
    :dedent:

To disassemble our einsum expression, we perform a number of steps.

We initially check all allowed characters and then begin the connection of our single ``EinsumNode``s using 
the ``parse_einsum_expression_recursive`` function. The first split we perform on our expression, is when we 
find the rightmost arrow ``->``. 

If we find such an arrow, we split the expression into two pieces, where the left is our ``input`` for our right 
``output`` expression.

.. literalinclude:: ../../src/einsum/EinsumTree.cpp
    :language: cpp
    :lines: 52-53
    :lineno-match:
    :caption: splitting the expression into ``input`` and ``output``
    :dedent:

The second step we take, is to again split our ``input``, but this time at a ``,``. That means, we 
look for the ``,`` that is between two brackets ``],[`` and where the number of open and closed brackets, is the same.

If there is such a ``,``, we have more than one input:

.. literalinclude:: ../../src/einsum/EinsumTree.cpp
    :language: cpp
    :lines: 59-89
    :lineno-match:
    :caption: splitting the left ``input`` into two inputs
    :dedent:

We connect these constructed components, by recursively calling the ``parse_einsum_expression_recursive`` function, for 
the children of the current node: 

.. literalinclude:: ../../src/einsum/EinsumTree.cpp
    :language: cpp
    :lines: 105-108
    :lineno-match:
    :caption: connecting ``EinsumNode``s to the ``EinsumTree``
    :dedent:

For the current node, we are now calculating all information (size, strides, etc.) that would be needed to execute a primitive operation. 

6.1.2 Lowering to Tensor Backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For all nodes, we start by setting the dimension ids of our current node:

.. literalinclude:: ../../src/einsum/EinsumTree.cpp
    :language: cpp
    :lines: 113-119
    :lineno-match:
    :caption: set current node ``ids``
    :dedent:

The next step is to calculate all other information, based on the amount of children a node has:

.. literalinclude:: ../../src/einsum/EinsumTree.cpp
    :language: cpp
    :lines: 130-150
    :lineno-match:
    :caption: check tree structure
    :dedent:

This means for the case, where we have no children, we return early. 
Otherwise we now gather and sort all ids, that are needed to execute a contraction or a unary operation:

.. literalinclude:: ../../src/einsum/EinsumTree.cpp
    :language: cpp
    :lines: 152-183
    :lineno-match:
    :caption: gather all unique ``ids`` from tensors for respective operation
    :dedent:

After gathering all ``ids``, we iterate over them and calculate all other information:

.. literalinclude:: ../../src/einsum/EinsumTree.cpp
    :language: cpp
    :lines: 196-221
    :lineno-match:
    :caption: set all dimension types 
    :dedent:

.. literalinclude:: ../../src/einsum/EinsumTree.cpp
    :language: cpp
    :lines: 223-243
    :lineno-match:
    :caption: calculate strides for ``stride_in0``
    :dedent:

The calculation for the strides of ``stride_in1`` and ``stride_out`` would be similar.

6.1.3 Run Optimization Passes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After setting all the information, we were then able to call our ``optimize`` function:

.. literalinclude:: ../../src/einsum/EinsumTree.cpp
    :language: cpp
    :lines: 275-282
    :lineno-match:
    :caption: call ``optimize`` function
    :dedent:

Based on the optimizations we could now set the type of primitive operation we want to perform:

.. literalinclude:: ../../src/einsum/EinsumTree.cpp
    :language: cpp
    :lines: 284-308
    :lineno-match:
    :caption: set ``primitive`` type
    :dedent:

Finally, we would be able to call the ``setup`` function to initialize the operation:

.. literalinclude:: ../../src/einsum/EinsumTree.cpp
    :language: cpp
    :lines: 314-323
    :lineno-match:
    :caption: call ``setup`` function
    :dedent:

6.1.4 Einsum Tree Execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To execute a whole einums tree expression, we have an ``execute`` function as a common entry point. 
We initially provide the function with the ``root`` node and initialize the tensor output for this node with zero:

.. literalinclude:: ../../src/einsum/EinsumTree.cpp
    :language: cpp
    :lines: 348-360
    :lineno-match:
    :caption: fill the ``tensor_out``
    :dedent:

After again recursively calling the function, we check, if we are a leaf node: 

.. literalinclude:: ../../src/einsum/EinsumTree.cpp
    :language: cpp
    :lines: 375-401
    :lineno-match:
    :caption: copy ``input_tensor`` for leaf node
    :dedent:

If we are no leaf node, we again call the ``execute`` function for the children of the node:

.. literalinclude:: ../../src/einsum/EinsumTree.cpp
    :language: cpp
    :lines: 405-420
    :lineno-match:
    :caption: execute the ``children`` and execute the ``operation``
    :dedent:

6.1.5 Performance Benchmarks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To validate the correctness and effectiveness of our implementation we performed benchmarks. 
Our first benchmark was to compare our new einsum implementation to the performance of the :ref:`tensor optimization<5.5.6 Performance Benchmarks>` benchmark.

.. literalinclude:: ../../benchmarks/optimized_tensor_and_einsum_operation_benchmarks.txt
    :language: text
    :lines: 81-105
    :lineno-match:
    :caption: comparison of ``einsum`` with ``tensor optimization``
    :dedent:

Secondly we compared our implementation with two reference einsum expressions:

1. ``[[8,4],[7,3,8]->[7,3,4]],[[[2,6,7],[1,5,6]->[1,2,5,7]],[0,5]->[0,1,2,7]]->[0,1,2,3,4]``
2. ``[[[[3,6,8,9]->[8,6,9,3]],[[2,5,7,9]->[7,5,2,9]]->[7,8,5,6,2,3]],[0,4,5,6]->[0,4,7,8,2,3]],[1,4,7,8]->[0,1,2,3]``

.. literalinclude:: ../../benchmarks/einsum_benchmark.txt
    :language: text
    :lineno-match:
    :caption: benchmark for reference einsum expressions
    :dedent:
