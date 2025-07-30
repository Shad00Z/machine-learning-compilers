##############################
6. Einsum Trees
##############################

In this section, we explain how we expanded the capabilities of our tensor compiler by adding support for einsum trees.

.. _einsum-lowering:

**********************************
6.1 Lowering
**********************************

The first task was to parse einsum trees as string expressions of the form ``[...],[...]->[...]`` into tree objects.
A tree object should then be lowered to our tensor operation backend, meaning that contraction and permutation nodes had to be mapped to executable tensor operations.
Furthermore, we had to run a series of optimization passes on the einsum tree and benchmark its performance.

.. _einsum-parsing:

6.1.1 Expression Parsing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to transform string expressions into a tree of connected objects, we first implemented an ``EinsumNode`` class.
An ``EinsumNode`` represents a node in the einsum tree and has the form ``[...]`` in the string representation.

.. code-block:: cpp
    :caption: EinsumNode class

    /// The IDs of the dimensions in the output tensor
    std::vector<int64_t> m_output_dimension_ids;
    /// The IDs of the dimensions in the operation
    std::vector<int64_t> m_dimension_ids;
    /// The data type of the tensor
    mini_jit::dtype_t m_dtype = mini_jit::dtype_t::fp32;
    /// Primititve type for the first touch kernel
    mini_jit::ptype_t m_prim_first_touch = mini_jit::ptype_t::none;
    /// Primitive type for the main kernel
    mini_jit::ptype_t m_prim_main = mini_jit::ptype_t::none;
    /// Primitive type for the last touch kernel
    mini_jit::ptype_t m_prim_last_touch = mini_jit::ptype_t::none;
    /// Dimension types of the loops (m, n, k, c)
    std::vector<mini_jit::dim_t> m_dim_types;
    /// Execution types of the loops (seq, shared, prim)
    std::vector<mini_jit::exec_t> m_exec_types;
    /// Sizes of the dimensions (loops)
    std::vector<int64_t> m_dim_sizes;
    /// Strides of the first input tensor
    std::vector<int64_t> m_strides_in0;
    /// Strides of the second input tensor
    std::vector<int64_t> m_strides_in1;
    /// Strides of the output tensor
    std::vector<int64_t> m_strides_out;
    /// Size of the output tensor
    int64_t m_tensor_size = 1;
    /// The output tensor for this node
    void *m_tensor_out = nullptr;
    /// The tensor operation associated with this node
    mini_jit::TensorOperation m_operation;
    /// String representation of the einsum expression
    std::string m_tensor_expression = "";
    /// The left child node in the einsum tree
    EinsumNode *m_left_child = nullptr;
    /// The right child node in the einsum tree
    EinsumNode *m_right_child = nullptr;
    /// The number of operations performed by this node
    double m_computational_operations = 0.0;

To disassemble an einsum expression, we perform a number of steps:

We initially check all allowed characters and then begin the connection of our ``EinsumNode`` objects using 
the ``parse_einsum_expression_recursive`` function. 
The first split we perform on the input expression is at the rightmost arrow ``->``. 

.. code-block:: cpp
    :caption: Determining the position of the rightmost arrow

    size_t l_arrow_pos = einsum_expression.rfind("->");

If we find such an arrow, we split the expression into two pieces, where the left part is the ``input`` for the ``output`` expression on the right side. Note that here we already remove the brackets around the output expression.

.. code-block:: cpp
    :caption: Splitting the einsum expression at the arrow position

    l_inputs = einsum_expression.substr(0, l_arrow_pos);
    l_output = einsum_expression.substr(l_arrow_pos + 3, einsum_expression.size() - l_arrow_pos - 4);

The second step is to split the ``input`` again, but this time at a ``,`` that divides the ``input`` into two valid expressions. Specifically, we look for the ``,`` that is between two brackets ``],[`` and where the number of open and closed brackets is the same. If such a ``,`` exists, we have more than one input:

.. code-block:: cpp
    :caption: Splitting the input into two parts

    // if the first char is not a bracket, there is only one input
    int64_t l_split_input_pos = -1;
    if (l_inputs[0] == '[')
    {
        // split inputs by comma
        int64_t l_brackets = 0;
        int64_t l_current_pos = 0;
        for (char c : l_inputs)
        {
            switch (c)
            {
            case '[':
                l_brackets++;
                break;
            case ']':
                l_brackets--;
                break;
            case ',':
                if (l_brackets == 0)
                {
                    l_split_input_pos = l_current_pos;
                }
                break;
            default:
                break;
            }
            // return early if we found the comma already
            if (l_split_input_pos != -1)
            {
                break;
            }
            l_current_pos++;
        }
    }
    if (l_split_input_pos == -1)
    {
        l_left_input_expression = l_inputs.substr(1, l_inputs.size() - 2);
        l_right_input_expression = "";
    }
    else
    {
        // remove outer brackets
        l_left_input_expression = l_inputs.substr(1, l_split_input_pos - 2);
        l_right_input_expression = l_inputs.substr(l_split_input_pos + 2, l_inputs.size() - l_split_input_pos - 3);
    }

Lastly, we create an ``EinsumNode`` for the current output expression and recursively sets its children to the respective input expressions, thus creating a tree.

.. code-block:: cpp
    :caption: Creation of a new EinsumNode

    return new EinsumNode(get_dimensions_from_expression(l_output),
                          l_output,
                          parse_einsum_expression_recursive(l_left_input_expression),
                          parse_einsum_expression_recursive(l_right_input_expression));

.. _initialize-einsum-nodes:

6.1.2 Initializing the Einsum Nodes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After creating an einsum tree, the next step was to initialize the nodes with the correct values, such as the dimensions or strides. We implemented a ``initialize_einsum_nodes`` function for this.

Since we evaluate the tree from the leaves to the root, the first step is to call the function recursively on the children of the current node:

.. code:: cpp

    initialize_einsum_nodes(root_node->m_left_child, dimension_sizes);
    initialize_einsum_nodes(root_node->m_right_child, dimension_sizes);

Next, the actual computations take place.

The first step is to gather all dimension IDs that are used in the tensor operation. These are the dimension IDs of the current (output) node's tensor and the dimension IDs of the children's output tensors. 

.. code-block:: cpp
    :caption: Gathering all dimension IDs involved in the tensor operation

    //////////////////////////////////////////////////////////////////
    // GATHER AND SORT ALL USED IDS
    //////////////////////////////////////////////////////////////////
    // this is already given
    std::vector<int64_t> *l_output_dimension_ids = &root_node->m_output_dimension_ids;
    // these will be initialized
    std::vector<int64_t> *l_operation_dim_ids = &root_node->m_dimension_ids;
    std::vector<int64_t> *l_dim_sizes = &root_node->m_dim_sizes;
    // local vector of sizes of the output dimensions
    std::vector<int64_t> l_out_dim_sizes = root_node->m_dim_sizes;
    // add ids from output
    for (auto dim_id : root_node->m_output_dimension_ids)
    {
        l_operation_dim_ids->push_back(dim_id);
        l_dim_sizes->push_back(dimension_sizes[dim_id]);
        l_out_dim_sizes.push_back(dimension_sizes[dim_id]);
    }
    // add ids from children (for 1 child, the ids are the same as the output ids)
    if (root_node->get_number_of_children() == 2)
    {
        for (auto dim_id : root_node->m_left_child->m_output_dimension_ids)
        {
            // check if the dimension id is already in the list
            if (!contains(*l_operation_dim_ids, dim_id))
            {
                l_operation_dim_ids->push_back(dim_id);
                l_dim_sizes->push_back(dimension_sizes[dim_id]);
            }
        }
        for (auto dim_id : root_node->m_right_child->m_output_dimension_ids)
        {
            // check if the dimension id is already in the list
            if (!contains(*l_operation_dim_ids, dim_id))
            {
                l_operation_dim_ids->push_back(dim_id);
                l_dim_sizes->push_back(dimension_sizes[dim_id]);
            }
        }
    }

This information allows us to compute the size that the output tensor of the current node will have:

.. code-block:: cpp
    :caption: Computing the size of the output tensor

    root_node->m_tensor_size = 1;
    for (auto &dim_id : *l_output_dimension_ids)
    {
        root_node->m_tensor_size *= dimension_sizes[dim_id];
    }

Knowing all used IDs and sizes, we can initialize the vectors for the dimension and execution types, as well as the vectors for the strides. 

.. code-block:: cpp

    //////////////////////////////////////////////////////////////////
    // INIT VECTORS
    //////////////////////////////////////////////////////////////////

    std::vector<dim_t> *l_dim_types = &root_node->m_dim_types;
    l_dim_types->resize(l_operation_dim_ids->size(), dim_t::k);
    std::vector<exec_t> *l_exec_types = &root_node->m_exec_types;
    l_exec_types->resize(l_operation_dim_ids->size(), exec_t::seq);
    std::vector<int64_t> *l_strides_in0 = &root_node->m_strides_in0;
    l_strides_in0->resize(l_operation_dim_ids->size(), 0);
    std::vector<int64_t> *l_strides_in1 = &root_node->m_strides_in1;
    l_strides_in1->resize(l_operation_dim_ids->size(), 0);
    std::vector<int64_t> *l_strides_out = &root_node->m_strides_out;
    l_strides_out->resize(l_operation_dim_ids->size(), 0);

Computing dimension types can be done using the following rules:

* Dimension exists in output and left input: ``M``

* Dimension exists in output and right input: ``N``

* All other dimensions: ``K`` (in both inputs, but not in the output)

As for the strides, we simply need to multiply dimension sizes. Consider the following example, where a tensor has the expression ``abc``. Since ``c`` is the right most dimension, we assume a stride of 1. The next dimension to the left of it is ``b``, and we compute its stride as the product of the dimension sizes to the right of it. This results in the stride of ``b`` being the dimension size of ``c``. For ``a``, the stride is therefore the product of the dimension sizes of ``b`` and ``c``.

.. code-block:: cpp
    :caption: Setting the type and stride for each dimension

    for (size_t i = 0; i < l_operation_dim_ids->size(); i++)
    {
        //////////////////////////////////////////////////////////////////
        // SET TYPE
        //////////////////////////////////////////////////////////////////
        // output + left = M
        // output + right = N

        int64_t l_dim_id = (*l_operation_dim_ids)[i];

        if (root_node->get_number_of_children() == 2)
        {
            // Dimension M
            if (contains(*l_output_dimension_ids, l_dim_id) &&
                contains(root_node->m_left_child->m_output_dimension_ids, l_dim_id))
            {
                (*l_dim_types)[i] = dim_t::m;
            }
            // Dimension N
            else if (contains(*l_output_dimension_ids, l_dim_id) &&
                     contains(root_node->m_right_child->m_output_dimension_ids, l_dim_id))
            {
                (*l_dim_types)[i] = dim_t::n;
            }
        }
        else
        {
            (*l_dim_types)[i] = dim_t::c;
        }

        // stride_in0
        if (root_node->m_left_child != nullptr &&
            contains(root_node->m_left_child->m_output_dimension_ids, l_dim_id))
        {
            int64_t stride = 1;
            auto it = std::find(root_node->m_left_child->m_output_dimension_ids.begin(),
                                root_node->m_left_child->m_output_dimension_ids.end(),
                                l_dim_id);
            size_t index = std::distance(root_node->m_left_child->m_output_dimension_ids.begin(), it);
            for (size_t j = index + 1; j < root_node->m_left_child->m_output_dimension_ids.size(); ++j)
            {
                stride *= dimension_sizes[root_node->m_left_child->m_output_dimension_ids[j]];
            }
            (*l_strides_in0)[i] = stride;
        }

        // stride_in1
        if (root_node->m_right_child != nullptr &&
            contains(root_node->m_right_child->m_output_dimension_ids, l_dim_id))
        {
            int64_t stride = 1;
            auto it = std::find(root_node->m_right_child->m_output_dimension_ids.begin(),
                                root_node->m_right_child->m_output_dimension_ids.end(),
                                l_dim_id);
            size_t index = std::distance(root_node->m_right_child->m_output_dimension_ids.begin(), it);
            for (size_t j = index + 1; j < root_node->m_right_child->m_output_dimension_ids.size(); ++j)
            {
                stride *= dimension_sizes[root_node->m_right_child->m_output_dimension_ids[j]];
            }
            (*l_strides_in1)[i] = stride;
        }

        // stride_out
        if (contains(*l_output_dimension_ids, l_dim_id))
        {
            int64_t stride = 1;
            auto it = std::find(l_output_dimension_ids->begin(), l_output_dimension_ids->end(), l_dim_id);
            size_t index = std::distance(l_output_dimension_ids->begin(), it);
            for (size_t j = index + 1; j < l_output_dimension_ids->size(); ++j)
            {
                stride *= dimension_sizes[root_node->m_output_dimension_ids[j]];
            }
            (*l_strides_out)[i] = stride;
        }
    }

.. _einsum-node-optimizations:

6.1.3 Optimization Passes on Nodes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After creating and initializing an einsum tree, we run the previously implemented Optimizer on each node in the einsum tree:

.. code-block:: cpp

    void mini_jit::einsum::EinsumTree::optimize_einsum_nodes(EinsumNode *root_node,
                                                             int64_t thread_target,
                                                             int64_t max_kernel_size)
    {
        if (root_node == nullptr)
        {
            return;
        }
        // no optimizations for input nodes
        if (root_node->get_number_of_children() == 0)
        {
            return;
        }
        // optimize children
        optimize_einsum_nodes(root_node->m_left_child, thread_target, max_kernel_size);
        optimize_einsum_nodes(root_node->m_right_child, thread_target, max_kernel_size);
        // optimize current node
        mini_jit::ir::Optimizer::optimize(root_node->m_dim_types,
                                          root_node->m_exec_types,
                                          root_node->m_dim_sizes,
                                          root_node->m_strides_in0,
                                          root_node->m_strides_in1,
                                          root_node->m_strides_out,
                                          thread_target,
                                          max_kernel_size);
    }

This function first makes a recursive call on its children and then executes the optimizer on the previously set vectors.

.. _einsum-lowering-subchapter:

6.1.4 Lowering to Tensor Backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While the optimization step for each node was optional, this step is required to make the einsum tree executable.
We call a function ``lower_einsum_nodes_to_tensor_operations`` that sets up ``TensorOperation`` objects for each node, which can later be executed.

For every node, we first set the data type and initialize the number of computational operations to zero.

.. code:: cpp

    // operations for all nodes
    root_node->m_dtype = dtype;
    root_node->m_computational_operations = 0.0;

In case the currently evaluated node is a leaf node, this is all we do here and return.

.. note::

    The need for saving the computational operations comes from our benchmarks. The idea is to evaluate the nodes from the leaves to the root, and for each node to compute the computational operations in that node and to add the computational operations of the children on top. This way, each node knows the number of computational operations for the whole subtree which it is the root of. Consequently, the root node of the tree will hold the number of computational operations for the whole tree.

Next, we perform the recursive call before doing any calculations.
This way, we evaluate the tree from the leaves to the root.

.. code:: cpp

    // lower children
    lower_einsum_nodes_to_tensor_operations(root_node->m_left_child, dimension_sizes, dtype);
    lower_einsum_nodes_to_tensor_operations(root_node->m_right_child, dimension_sizes, dtype);

The actual computations for a node happen next. 

We start by identifying the type of the operation, based on the number of primitive dimensions. 
Conveniently, we use this step to also set the number of computational operations for the current tensor operation.

.. code-block:: cpp
    :caption: Identifying the primitive type for each tensor operation

    // lower current node
    int l_prim_count = std::count(root_node->m_exec_types.begin(), root_node->m_exec_types.end(), exec_t::prim);
    mini_jit::ptype_t l_main_ptype = mini_jit::ptype_t::none;
    if (l_prim_count == 2)
    {
        l_main_ptype = mini_jit::ptype_t::identity;
        root_node->m_computational_operations = 0.0; // no operations for identity
    }
    else if (l_prim_count == 3)
    {
        l_main_ptype = mini_jit::ptype_t::gemm;
        root_node->m_computational_operations = 2.0f;
        for (int64_t size : root_node->m_dim_sizes)
        {
            root_node->m_computational_operations *= size;
        }
    }
    else if (l_prim_count == 4)
    {
        l_main_ptype = mini_jit::ptype_t::brgemm;
        root_node->m_computational_operations = 2.0f;
        for (int64_t size : root_node->m_dim_sizes)
        {
            root_node->m_computational_operations *= size;
        }
    }

As mentioned above, we also need to add the computational operations of the children.

.. code:: cpp

    // add child ops
    root_node->m_computational_operations += root_node->m_left_child ? root_node->m_left_child->m_computational_operations : 0.0;
    root_node->m_computational_operations += root_node->m_right_child ? root_node->m_right_child->m_computational_operations : 0.0;

Lastly, all that is left is to call the setup function of the tensor operation using the identified operation type and the vectors we computed in :ref:`initialize-einsum-nodes`.

.. code-block:: cpp
    :caption: Setting up the Tensor Operation for an EinsumNode

    root_node->m_operation.setup(root_node->m_dtype,
                                 ptype_t::none,
                                 l_main_ptype,
                                 ptype_t::none,
                                 root_node->m_dim_types,
                                 root_node->m_exec_types,
                                 root_node->m_dim_sizes,
                                 root_node->m_strides_in0,
                                 root_node->m_strides_in1,
                                 root_node->m_strides_out);

.. _einsum-execution:

6.1.4 Einsum Tree Execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After parsing, optimizing and lowering the einsum tree, we are now able to execute the operations. 
For this, we use an ``execute`` function as a common entry point. 
We initially provide the function with the ``root`` node and initialize the tensor output for this node with zero. 
If this is the first execution, we allocate a new array and if not, we simply fill it with zeroes.

.. code-block:: cpp
    :caption: Management of intermediate result tensors

    const int64_t l_tensor_size = root_node->m_tensor_size;

    if (root_node->m_dtype == mini_jit::dtype_t::fp32)
    {
        if (root_node->m_tensor_out == nullptr)
        {
            root_node->m_tensor_out = new float[l_tensor_size]{0.0f};
        }
        else
        {
            std::fill(static_cast<float *>(root_node->m_tensor_out),
                      static_cast<float *>(root_node->m_tensor_out) + l_tensor_size,
                      0.0f);
        }
    }
    else if (root_node->m_dtype == mini_jit::dtype_t::fp64)
    {
        if (root_node->m_tensor_out == nullptr)
        {
            root_node->m_tensor_out = new double[l_tensor_size]{0.0};
        }
        else
        {
            std::fill(static_cast<double *>(root_node->m_tensor_out),
                      static_cast<double *>(root_node->m_tensor_out) + l_tensor_size,
                      0.0);
        }
    }

Next, there are two options for each node. Either it is an interior node or a leaf node. 
If the current node happens to be a leaf node, we retrieve the pointer to the input tensor from a map using the tensor expression, and simply copy it to the output tensor:

.. code-block:: cpp
    :caption: Copying the input tensors for leaf nodes

    if (root_node->get_number_of_children() == 0)
    {
        // check if input tensor is given
        auto it = tensor_inputs.find(root_node->m_tensor_expression);
        if (it != tensor_inputs.end())
        {
            if (root_node->m_dtype == mini_jit::dtype_t::fp32)
            {
                std::copy(
                    static_cast<const float *>(it->second),
                    static_cast<const float *>(it->second) + l_tensor_size,
                    static_cast<float *>(root_node->m_tensor_out));
            }
            else if (root_node->m_dtype == mini_jit::dtype_t::fp64)
            {
                std::copy(
                    static_cast<const double *>(it->second),
                    static_cast<const double *>(it->second) + l_tensor_size,
                    static_cast<double *>(root_node->m_tensor_out));
            }
        }
        else
        {
            throw std::invalid_argument("Error: No input tensor found for leaf node with expression: " +
                                        root_node->m_tensor_expression);
        }
    }

The argument ``tensor_inputs`` is a C++ map of the type ``std::map<std::string, void const *>``. For each input tensor expression, it holds the corresponding pointer to the input tensor. For example:

.. code:: cpp

    std::map<std::string, void const *> tensor_inputs;
    float *tensor_A = new float[SIZE_A];
    tensor_inputs["2,0"] = tensor_A;

If the current node happens to be an interior node, we can simply execute the tensor operation:

.. code:: cpp

    // compute children
    execute(root_node->m_left_child, dimension_sizes, tensor_inputs);
    execute(root_node->m_right_child, dimension_sizes, tensor_inputs);

    // execute operation
    auto l_ptr_right_child = root_node->m_right_child ? root_node->m_right_child->m_tensor_out : nullptr;
    root_node->m_operation.execute(root_node->m_left_child->m_tensor_out,
                                   l_ptr_right_child,
                                   root_node->m_tensor_out);

6.1.5 Performance Benchmarks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To validate the correctness and effectiveness of our implementation we performed benchmarks. 
Our first benchmark was to compare our new einsum implementation to the performance of the :ref:`tensor optimization<5.5.6 Performance Benchmarks>` benchmark.

.. literalinclude:: ../../benchmarks/optimized_tensor_and_einsum_operation_benchmarks_old.txt
    :language: text
    :lines: 81-105
    :caption: comparison of ``einsum`` with ``tensor optimization``
    :dedent:

Secondly we compared our implementation with two reference einsum expressions:

1. ``[[8,4],[7,3,8]->[7,3,4]],[[[2,6,7],[1,5,6]->[1,2,5,7]],[0,5]->[0,1,2,7]]->[0,1,2,3,4]``
2. ``[[[[3,6,8,9]->[8,6,9,3]],[[2,5,7,9]->[7,5,2,9]]->[7,8,5,6,2,3]],[0,4,5,6]->[0,4,7,8,2,3]],[1,4,7,8]->[0,1,2,3]``

.. literalinclude:: ../../benchmarks/einsum_benchmark.txt
    :language: text
    :caption: benchmark for reference einsum expressions
    :dedent:

.. _einsum-tree-optimizations:

**********************************
6.2 Einsum Tree Optimization
**********************************

Being able to compute pre-optimized einsum trees is only the starting point of our einsum tree support.
The general case would be that an einsum tree can be optimized to enhance the execution time and therefore improve the throughput. 
In this section, we consider several optimization passes that we implemented for einsum trees.

6.2.1 Swapping Operands
^^^^^^^^^^^^^^^^^^^^^^^^^

The first step towards an optimization pass for the einsum tree was a ``swap_nodes`` function, which would swap two input nodes if the dimensions did not fit our schema.

Consider the following einsum tree: ``[[7,3,8],[8,4]->[7,3,4]]``. 
At first glance this tree expression seems fine, however, as our execution demands, that:

1. the unit stride of our ``left_child`` and the unit stride of our ``parent`` have to be the same (``dim_t::M``) and
2. the unit stride of our ``right_child`` is a ``dim_t::K``,

The given expression however does not fulfill these demands.
In this example, the swapping of ``children`` / ``operands`` comes in handy and would transform the einsum tree:

1. from ``[[7,3,8],[8,4]->[7,3,4]]``
2. to ``[[8,4],[7,3,8]->[7,3,4]]``

In our implementation we look at possible swaps after parsing our einsum expression. 
The reason for that is, if we do it at this position, we can exploit the given order of the einsum expression and
more importantly, we did not initialize our einsum nodes yet:

.. code:: cpp

    mini_jit::einsum::EinsumNode *root_node = parse_einsum_expression_recursive(einsum_expression);

    // SWAP NODES
    swapNodes(root_node);

    initialize_einsum_nodes(root_node, dimension_sizes);
    return root_node;

This positioning of the node swap is important, because by executing it before our node initialization, we save a redundant 'initialization' later on.

For example, considering our simple example ``[[7,3,8],[8,4]->[7,3,4]]`` the biggest problem would be that after the einsum nodes for this tree are initialized, the ``dimension`` with ``id=4`` would be initialized as ``dim_t::N``.
After swapping the children nodes, we would have to 'recompute' these dimensions, 
because the ``dimension`` with ``id=4`` would now have to be of type ``dim_t::M``.

A node swapping can only happen, if there are two children present. That means if we look at a leaf node, we simply return, 
and if we look at a node with one child, we call our ``swap_nodes`` function only on one child and return:

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

For the cases, where the conditions are not met, we rely on our other optimization techniques 
to find matching unit strides either by reordering or permuting single tree nodes.

The last step is to recursively call our ``swap_nodes`` function on the children nodes to guarantee 
that all nodes of the tree are looked at:

.. code:: cpp

    // recursively swap children
    swap_nodes(einsum_node->leftChild);
    swap_nodes(einsum_node->rightChild);

6.2.2 Reordering Dimensions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As we tried to benchmark further unoptimized einsum trees, we realized that swapping operands alone was not sufficient. This is because during the identification process of primitive dimensions, our implementation looks for specific strides.
For example, our code requires that a primary ``M`` dimension, which should appear in the left input tensor and also in the output tensor, needs to have a unit stride in both tensors for contractions such as ``GEMM`` and ``BRGEMM``. 
However, should this primary ``M`` dimension not be in the right most position in the tensor expression, it will not have unit stride. 
Therefore, we need to perform a dimension reordering to make the tree executable in our implementation.

Our ``reorder_node_dimensions`` function starts by checking the number of children the currently evaluated node has.

.. code:: cpp

    if (root_node == nullptr || root_node->get_number_of_children() == 0)
    {
        // since we view root_node as the parent node where the children are reordered,
        // we do not need to do anything for leaf nodes
        return;
    }

    // one child -> identity operation and not a contraction
    // -> no need to reorder dimensions here
    if (root_node->get_number_of_children() == 1)
    {
        reorder_node_dimensions(root_node->m_left_child);
        return;
    }

In the code after this section, we are safe to assume that the current node has two children. Furthermore, we make the assumption that the current node is a parent and therefore the order of its dimensions is correct. 
In other words, the order of the dimensions of the current node specifies the correct dimension order to which we need to adapt the children.

We start by saving the index at which the dimension with unit stride in the left input and output tensor should be.

.. code-block:: cpp
    :caption: Determining the dimension ID that has unit stride

    int64_t l_unit_stride_root_node = root_node->m_output_dimension_ids.size() - 1;
    int64_t l_parent_dim_id = root_node->m_output_dimension_ids[l_unit_stride_root_node];
    int64_t l_unit_stride_left_child = root_node->m_left_child->m_output_dimension_ids.size() - 1;

Next, we perform a swap of the input tensors if required. This step makes the ``swap_nodes`` function redundant, which is why we do not actually use ``swap_nodes`` in our final implementation.

.. code-block:: cpp
    :caption: Swapping children if necessary

    // Find unit stride for M
    if (contains(root_node->m_right_child->m_output_dimension_ids,
                 root_node->m_output_dimension_ids[l_unit_stride_root_node]))
    {
        // swap children
        EinsumNode *l_temp_node = root_node->m_left_child;
        root_node->m_left_child = root_node->m_right_child;
        root_node->m_right_child = l_temp_node;
        l_unit_stride_left_child = root_node->m_left_child->m_output_dimension_ids.size() - 1;
    }

We can now assume that the primary ``M`` dimension should be present in the left input and also the output tensor. Furthermore, we assume that the primary ``M`` dimension has unit stride in the output tensor, because the output tensor is ordered correctly. 
With these assumptions, we now need to check where the primary ``M`` dimension is in the left input tensor and move it to the right most position if it is not already there.

.. code-block:: cpp
    :caption: Moving the M dimension to the right most position in the first input node

    auto l_dim_child_m_it = std::find_if(root_node->m_left_child->m_output_dimension_ids.begin(),
                                         root_node->m_left_child->m_output_dimension_ids.end(),
                                         [l_parent_dim_id](const int64_t dim_id)
                                         {
                                             return (dim_id == l_parent_dim_id);
                                         });

    // no M found
    if (l_dim_child_m_it == root_node->m_left_child->m_output_dimension_ids.end())
    {
        throw std::invalid_argument("EinsumTree: No M dimension found for child " +
                                    root_node->m_left_child->m_tensor_expression +
                                    " and parent " +
                                    root_node->m_tensor_expression);
    }
    // M is in output dims but not the right-most element
    else if (l_dim_child_m_it < root_node->m_left_child->m_output_dimension_ids.end() - 1)
    {
        EinsumNode *l_left_child_permute = new EinsumNode(root_node->m_left_child->m_output_dimension_ids,
                                                          root_node->m_left_child->m_tensor_expression,
                                                          root_node->m_left_child,
                                                          nullptr);

        // move M dimension to the right-most position
        // Calculate the position in the new vector and rotate
        size_t l_m_position = std::distance(root_node->m_left_child->m_output_dimension_ids.begin(), l_dim_child_m_it);
        auto l_new_m_it = l_left_child_permute->m_output_dimension_ids.begin() + l_m_position;
        std::rotate(l_new_m_it, l_new_m_it + 1, l_left_child_permute->m_output_dimension_ids.end());

        // update expression of the new permute node
        std::string l_new_expression = "";
        for (size_t i = 0; i < l_left_child_permute->m_output_dimension_ids.size(); i++)
        {
            if (i > 0)
            {
                l_new_expression += ",";
            }
            l_new_expression += std::to_string(l_left_child_permute->m_output_dimension_ids[i]);
        }
        l_left_child_permute->m_tensor_expression = l_new_expression;

        // insert into the tree
        root_node->m_left_child = l_left_child_permute;
    }

In order to actually perform the reordering, we insert a permutation node instead of editing the child nodes. The new permutation node becomes the new left child of the current node and the previous left child is now the child of the new permutation node.

Similar to the primary ``M`` dimension in the left input tensor, we need to ensure that the primary ``K`` dimension has unit stride in the right input tensor.

We start by finding the right most dimension in the left input tensor that also exists in the right input tensor. This is because the requirement for ``K`` dimensions is that they exist in both input tensors. Furthermore, it is beneficial for the primary ``K`` dimension to have a low stride (for shorter jumps in memory), which is why we choose the right most ``K``.

.. code-block:: cpp
    :caption: Finding a K dimension

    int64_t l_k_dim_index = l_unit_stride_left_child;
    for (int i = l_unit_stride_left_child; i >= 0; i--)
    {
        if (contains(root_node->m_right_child->m_output_dimension_ids,
                     root_node->m_left_child->m_output_dimension_ids[i]))
        {
            l_k_dim_index = i;
            break;
        }
    }

Having found such a ``K`` dimension, we now need to ensure that it is the right most dimension in the right input tensor, to ensure that it will have unit stride there. 
This process is exactly the same as shown above for the ``M`` dimension, where we also had to insert a permutation node and update the respective child pointers.

.. code-block:: cpp
    :caption: Moving the K dimension to the right most position in the second input node

    int64_t l_k_dim_id = root_node->m_left_child->m_output_dimension_ids[l_k_dim_index];
    auto l_dim_child_k_it = std::find_if(root_node->m_right_child->m_output_dimension_ids.begin(),
                                         root_node->m_right_child->m_output_dimension_ids.end(),
                                         [l_k_dim_id](const int64_t dim_id)
                                         {
                                             return (dim_id == l_k_dim_id);
                                         });

    // no K found
    if (l_dim_child_k_it == root_node->m_right_child->m_output_dimension_ids.end())
    {
        throw std::invalid_argument("EinsumTree: No K dimension found for child " +
                                    root_node->m_right_child->m_tensor_expression +
                                    " and parent " +
                                    root_node->m_tensor_expression);
    }
    // K is in output dims but not the right-most element
    else if (l_dim_child_k_it < root_node->m_right_child->m_output_dimension_ids.end() - 1)
    {
        EinsumNode *l_right_child_permute = new EinsumNode(root_node->m_right_child->m_output_dimension_ids,
                                                           root_node->m_right_child->m_tensor_expression,
                                                           root_node->m_right_child,
                                                           nullptr);

        // move K dimension to the right-most position
        // Calculate the position in the new vector and rotate
        size_t l_k_position = std::distance(root_node->m_right_child->m_output_dimension_ids.begin(), l_dim_child_k_it);
        auto l_new_k_it = l_right_child_permute->m_output_dimension_ids.begin() + l_k_position;
        std::rotate(l_new_k_it, l_new_k_it + 1, l_right_child_permute->m_output_dimension_ids.end());

        // update expression of the new permute node
        std::string l_new_expression = "";
        for (size_t i = 0; i < l_right_child_permute->m_output_dimension_ids.size(); i++)
        {
            if (i > 0)
            {
                l_new_expression += ",";
            }
            l_new_expression += std::to_string(l_right_child_permute->m_output_dimension_ids[i]);
        }
        l_right_child_permute->m_tensor_expression = l_new_expression;

        // insert into the tree
        root_node->m_right_child = l_right_child_permute;
    }

We can now say that the children are also ordered correctly, which is why we make the recusive call on the children in the last step. This way, the correct order of the parent is always guaranteed.

.. code:: cpp

    // recursively call children
    reorder_node_dimensions(root_node->m_left_child);
    reorder_node_dimensions(root_node->m_right_child);

6.2.3 Benchmarks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For this task, we were given three example einsum trees as string representations which we were supposed to benchmark. The examples were:

#. ``[[7,3,8],[8,4]->[7,3,4]],[[0,5],[[5,1,6],[6,2,7]->[5,1,2,7]]->[0,1,2,7]]->[0,1,2,3,4``

#. ``[[1,4,7,8],[[0,4,5,6],[[2,5,7,9],[3,6,8,9]->[2,5,7,3,6,8]]->[0,4,2,7,3,8]]->[0,1,2,3]``

#. ``[[2,7,3],[3,8,4]->[2,7,8,4]],[[4,9,0],[[0,5,1],[1,6,2]->[0,5,6,2]]->[4,9,5,6,2]]->[5,6,7,8,9]``

Our benchmarks returned the following results:

.. code:: text

    Running EinsumTree benchmark - Optimization Example #1
    Total time (s):                  3.23636
    Total reps:                      9
    Total floating point operations: 356487340032
    Estimated GFLOPS/sec:            110.151
    --------------------------------------------------
    Running EinsumTree benchmark - Optimization Example #2
    Total time (s):                  3.01429
    Total reps:                      195
    Total floating point operations: 599359488000
    Estimated GFLOPS/sec:            198.84
    --------------------------------------------------
    Running EinsumTree benchmark - Optimization Example #3
    Total time (s):                  3.0549
    Total reps:                      24
    Total floating point operations: 801840000000
    Estimated GFLOPS/sec:            262.477
    --------------------------------------------------
