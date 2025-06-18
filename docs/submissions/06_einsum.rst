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

We initially check all allowed characters and then begin the connection of our ``EinsumNode`` objects using 
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

The second step we take, is to split our ``input`` again, but this time at a ``,``. That means, we 
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

For all nodes, we start by setting the dimension IDs of our current node:

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
Otherwise we now gather and sort all IDs, that are needed to execute a contraction or a unary operation:

.. literalinclude:: ../../src/einsum/EinsumTree.cpp
    :language: cpp
    :lines: 152-183
    :lineno-match:
    :caption: gather all unique ``IDs`` from tensors for respective operation
    :dedent:

After gathering all ``IDs``, we iterate over them and calculate all other information:

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

To execute a whole einsum tree expression, we use an ``execute`` function as a common entry point. 
We initially provide the function with the ``root`` node and initialize the tensor output for this node with zero:

.. literalinclude:: ../../src/einsum/EinsumTree.cpp
    :language: cpp
    :lines: 347-359
    :lineno-match:
    :caption: fill the ``tensor_out``
    :dedent:

After each recursive call of the function, we check if we are a leaf node: 

.. literalinclude:: ../../src/einsum/EinsumTree.cpp
    :language: cpp
    :lines: 374-400
    :lineno-match:
    :caption: copy ``input_tensor`` for leaf node
    :dedent:

If we are not a leaf node, we call the ``execute`` function for the children of the node:

.. literalinclude:: ../../src/einsum/EinsumTree.cpp
    :language: cpp
    :lines: 404-419
    :lineno-match:
    :caption: execute the ``children`` and execute the ``operation``
    :dedent:

6.1.5 Performance Benchmarks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To validate the correctness and effectiveness of our implementation we performed benchmarks. 
Our first benchmark was to compare our new einsum implementation to the performance of the :ref:`tensor optimization<5.5.6 Performance Benchmarks>` benchmark.

.. literalinclude:: ../../benchmarks/optimized_tensor_and_einsum_operation_benchmarks_old.txt
    :language: text
    :lines: 81-105
    :lineno-match:
    :caption: comparison of ``einsum`` with ``tensor optimization``
    :dedent:

Secondly we compared our implementation with two reference einsum expressions:

1. ``[[8,4],[7,3,8]->[7,3,4]],[[[2,6,7],[1,5,6]->[1,2,5,7]],[0,5]->[0,1,2,7]]->[0,1,2,3,4]``
2. ``[[[[3,6,8,9]->[8,6,9,3]],[[2,5,7,9]->[7,5,2,9]]->[7,8,5,6,2,3]],[0,4,5,6]->[0,4,7,8,2,3]],[1,4,7,8]->[0,1,2,3]``

.. literalinclude:: ../../benchmarks/einsum_benchmark_old.txt
    :language: text
    :lineno-match:
    :caption: benchmark for reference einsum expressions
    :dedent:

**********************************
6.2 Optimization
**********************************

Being able to compute pre-optimized einsum trees is only the starting point for einsum tree execution. 
The general case would be that an einsum tree can be optimized to enhance the execution time and therefore improve the throughput. 
In the following, we will look at several of these problems and how to resolve them.


6.2.2 Swapping Operands
^^^^^^^^^^^^^^^^^^^^^^^^^

If the tree itself already has correctly ordered dimensions, we can skip the reordering optimization. 
However, it can still be the case, that the operands for an operation in the einsum tree have to be swapped. 

Consider the following einsum tree: ``[[7,3,8],[8,4]->[7,3,4]]``. 
At first glance this tree expression seems fine, however, as our execution demands, that:

1. the unit stride of our ``left_child`` and  the unit stride of our ``parent`` have to be the same (``dim_t::M``) and
2. the unit stride of our ``right_child`` is a ``dim_t::K``,

there seems to be a problem. For this example, the swapping of ``children`` / ``operands`` comes in handy. 
The swap of the children would transform the einsum tree:

1. from ``[[7,3,8],[8,4]->[7,3,4]]``
2. to ``[[8,4],[7,3,8]->[7,3,4]]``

In our implementation we look at possible swaps after parsing our einsum expression. 
The reason for that is, if we do it at this position, we can exploit the given order of the einsum expression and, 
more importantly, we did not initialize our einsum nodes yet:

.. code:: cpp

    mini_jit::einsum::EinsumNode *root_node = parse_einsum_expression_recursive(einsum_expression);

    // SWAP NODES
    swapNodes(root_node);

    initialize_einsum_nodes(root_node, dimension_sizes);
    return root_node;

This positioning of the node swap is important, because by executing it before our node initialization, we safe us another 'initialization' later on.

For example, considering our simple example ``[[7,3,8],[8,4]->[7,3,4]]`` the biggest problem would be, 
that after the einsum nodes for this tree are initalizated the ``dimension`` with ``id=4`` would be initialized as ``dim_t::N``. 
After swapping the children nodes, we would have to 'recompute' these dimensions, because the ``dimension`` with ``id=4`` would now have to be of type ``dim_t::M``.

We perform a swap of the children nodes with a ``swapNodes`` function.
A node swapping can only happen, if there are two children present. That means if we look at a leaf node, we simply return, 
and if we look at a node with one child, we call our ``swapNodes`` function only on one child and return:

.. code:: cpp

    if (einsum_node == nullptr || einsum_node->get_number_of_children() == 0)
    {
        return;
    }

    if (einsum_node->get_number_of_children() == 1)
    {
        swapNodes(einsum_node->leftChild);
        return;
    }

If a node has two children, we look at two things:

1. the unit strides of the ``right_child`` and the current ``parent`` are of the same ``dim_t`` and
2. the unit stride of the ``left_child`` exists somewhere in the ``right_child``.

If these two conditions are met, we swap the two children nodes:

.. code:: cpp

    int64_t l_unit_stride_root_node = einsum_node->output_dimension_ids.size() - 1;
    int64_t l_unit_stride_left_child = einsum_node->leftChild->output_dimension_ids.size() - 1;
    int64_t l_unit_stride_right_child = einsum_node->rightChild->output_dimension_ids.size() - 1;

    if (einsum_node->output_dimension_ids[l_unit_stride_root_node] == einsum_node->rightChild->output_dimension_ids[l_unit_stride_right_child] &&
        contains(einsum_node->rightChild->output_dimension_ids, einsum_node->leftChild->output_dimension_ids[l_unit_stride_left_child]))
    {
        EinsumNode *l_temp_node = einsum_node->leftChild;

        einsum_node->leftChild = einsum_node->rightChild;
        einsum_node->rightChild = l_temp_node;
    }

For the cases, where these conditions are not met, we rely on our other optimization techniques 
to find matching unit strides either by reordering or permuting single tree nodes.

The last step is to recursively call our ``swapNodes`` function on the children nodes, to guarantee, 
that all nodes of the tree are looked at:

.. code:: cpp

    // recursively swap children
    swapNodes(einsum_node->leftChild);
    swapNodes(einsum_node->rightChild);