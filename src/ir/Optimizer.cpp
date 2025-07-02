#include "Optimizer.h"
#include "IRConverter.h"
#include <algorithm>
#include <limits.h>
#include <iostream>

void mini_jit::ir::Optimizer::optimize(std::vector<mini_jit::ir::Dimension> &dimensions,
                                       int64_t thread_target,
                                       int64_t max_kernel_size,
                                       int64_t min_kernel_size)
{
    fuseDimensions(dimensions,
                   min_kernel_size);

    splitDimensions(dimensions,
                    max_kernel_size);

    identifyPrimitives(dimensions);

    // Verify that there are 2, 3 or 4 primitive dimensions
    int prim_count = std::count_if(dimensions.begin(), dimensions.end(),
                                   [](const mini_jit::ir::Dimension &dim)
                                   {
                                       return dim.exec_type == exec_t::prim;
                                   });
    if (prim_count < 2 || prim_count > 4)
    {
        throw std::invalid_argument("Optimizer: Expected 2, 3 or 4 primitive dimensions, found " + std::to_string(prim_count) + ". Try setting all dimensions to seq or undefined.");
    }

    createSharedLoops(dimensions,
                      thread_target);

    // TODO: Dimension Reordering?
}

void mini_jit::ir::Optimizer::optimize(std::vector<mini_jit::dim_t> &dim_types,
                                       std::vector<mini_jit::exec_t> &exec_types,
                                       std::vector<int64_t> &dim_sizes,
                                       std::vector<int64_t> &strides_in0,
                                       std::vector<int64_t> &strides_in1,
                                       std::vector<int64_t> &strides_out,
                                       int64_t thread_target,
                                       int64_t max_kernel_size,
                                       int64_t min_kernel_size)
{
    // Convert input vectors to a vector of Dimensions
    std::vector<mini_jit::ir::Dimension> dimensions;
    IRConverter::convertConfigToDimensions(dim_types,
                                           exec_types,
                                           dim_sizes,
                                           strides_in0,
                                           strides_in1,
                                           strides_out,
                                           dimensions);
    // Optimize the dimensions
    optimize(dimensions, 
             thread_target, 
             max_kernel_size, 
             min_kernel_size);
    // Convert the optimized dimensions back to the original format
    IRConverter::convertDimensionsToConfig(dimensions,
                                           dim_types,
                                           exec_types,
                                           dim_sizes,
                                           strides_in0,
                                           strides_in1,
                                           strides_out);
}

void mini_jit::ir::Optimizer::identifyPrimitives(std::vector<mini_jit::ir::Dimension> &dimensions)
{
    // Handle identity case first
    auto l_has_c_dim = std::any_of(dimensions.begin(), dimensions.end(),
                                   [](const mini_jit::ir::Dimension &dim)
                                   {
                                       return dim.type == dim_t::c;
                                   });

    // Handle binary case
    auto l_has_k_dim = std::any_of(dimensions.begin(), dimensions.end(),
                                   [](const mini_jit::ir::Dimension &dim)
                                   {
                                       return dim.type == dim_t::k;
                                   });

    if (l_has_c_dim)
    {
        // check that all dimensions are c
        if (!std::all_of(dimensions.begin(), dimensions.end(),
                         [](const mini_jit::ir::Dimension &dim)
                         {
                             return dim.type == dim_t::c;
                         }))
        {
            throw std::invalid_argument("Optimizer: All dimensions must be of type 'c' for unary operations.");
        }
        // check for existing primary dimensions
        int prim_c_count = std::count_if(dimensions.begin(), dimensions.end(),
                                         [](const mini_jit::ir::Dimension &dim)
                                         {
                                             return dim.type == dim_t::c && dim.exec_type == exec_t::prim;
                                         });
        if (prim_c_count == 2)
        {
            return; // primary dimensions already set
        }
        else if (prim_c_count != 0)
        {
            throw std::invalid_argument("Optimizer: Expected 0 or 2 primary dimensions of type 'c', found " + std::to_string(prim_c_count) + ". Try setting all dimensions to seq or undefined.");
        }

        /////////////////////////////////////////////////////////////////
        // FIND UNARY PRIM M
        /////////////////////////////////////////////////////////////////
        // req: unit stride in in0. Out might not be 1 if operation transposes.
        auto l_dim_m_it = std::find_if(dimensions.begin(), dimensions.end(),
                                       [](const mini_jit::ir::Dimension &dim)
                                       {
                                           return (dim.type == dim_t::c) &&
                                                  dim.stride_in0 == 1 &&
                                                  dim.stride_in1 == 0;
                                       });

        bool l_transpose = false;
        if (l_dim_m_it != dimensions.end())
        {
            // transpose if the output stride in M is not 1
            l_transpose = l_dim_m_it->stride_out != 1;
            // set dimension data
            l_dim_m_it->exec_type = exec_t::prim;
            if (l_dim_m_it != dimensions.end() - 1)
            {
                // move M to the back
                std::rotate(l_dim_m_it, l_dim_m_it + 1, dimensions.end());
            }
        }
        else
        {
            throw std::invalid_argument("Optimizer: No suitable primary dimension M found.");
        }

        if (l_transpose)
        {
            /////////////////////////////////////////////////////////////////
            // FIND UNARY PRIM N
            /////////////////////////////////////////////////////////////////
            // req: unit stride in out.
            auto l_dim_n_it = std::find_if(dimensions.begin(), dimensions.end(),
                                           [](const mini_jit::ir::Dimension &dim)
                                           {
                                               return (dim.type == dim_t::c) &&
                                                      dim.stride_out == 1 &&
                                                      dim.stride_in1 == 0;
                                           });

            if (l_dim_n_it != dimensions.end())
            {
                // set dimension data
                l_dim_n_it->exec_type = exec_t::prim;
                if (l_dim_n_it != dimensions.end() - 1)
                {
                    // move N to the back
                    std::rotate(l_dim_n_it, l_dim_n_it + 1, dimensions.end());
                }
            }
            else
            {
                throw std::invalid_argument("Optimizer: No suitable primary dimension N found.");
            }
        }
        else
        {
            // TODO: check if this is ok

            /////////////////////////////////////////////////////////////////
            // FIND UNARY PRIM N
            /////////////////////////////////////////////////////////////////
            // req: choose the one with the smallest stride in in0
            int l_n_dim_stride = INT_MAX;
            int l_n_dim_id = -1;
            for (size_t i = 0; i < dimensions.size(); i++)
            {
                if ((dimensions[i].type == dim_t::c) &&
                    dimensions[i].stride_in1 == 0 &&
                    (dimensions[i].exec_type == exec_t::undefined ||
                     dimensions[i].exec_type == exec_t::seq))
                {
                    int l_current_stride = dimensions[i].stride_in0;
                    if (l_current_stride < l_n_dim_stride)
                    {
                        l_n_dim_stride = l_current_stride;
                        l_n_dim_id = static_cast<int>(i);
                    }
                }
            }

            if (l_n_dim_id == -1)
            {
                throw std::invalid_argument("Optimizer: No suitable primary dimension N found.");
            }

            // set dimension data
            dimensions[l_n_dim_id].exec_type = exec_t::prim;
            // move N to the back
            if (l_n_dim_id != static_cast<int>(dimensions.size()) - 1)
            {
                std::rotate(dimensions.begin() + l_n_dim_id, dimensions.begin() + l_n_dim_id + 1, dimensions.end());
            }
        }

        // lastly, set all remaining dimensions to seq
        for (auto &dim : dimensions)
        {
            if (dim.exec_type == exec_t::undefined)
            {
                dim.exec_type = exec_t::seq;
            }
        }

        return; // all primary dimensions set
    }
    // BINARY CASE
    else if (!l_has_k_dim)
    {
        // check for existing primary dimensions
        int prim_count = std::count_if(dimensions.begin(), dimensions.end(),
                                         [](const mini_jit::ir::Dimension &dim)
                                         {
                                             return dim.type == dim_t::c && dim.exec_type == exec_t::prim;
                                         });
        if (prim_count == 2)
        {
            return; // primary dimensions already set
        }
        else if (prim_count != 0)
        {
            throw std::invalid_argument("Optimizer: Expected 0 or 2 primary dimensions, found " + std::to_string(prim_count) + ". Try setting all dimensions to seq or undefined.");
        }
        /////////////////////////////////////////////////////////////////
        // FIND PRIM M
        /////////////////////////////////////////////////////////////////
        // req: unit stride in all tensors
        auto l_dim_m_it = std::find_if(dimensions.begin(), dimensions.end(),
                                       [](const mini_jit::ir::Dimension &dim)
                                       {
                                           return dim.type == dim_t::m &&
                                                  dim.stride_in0 == 1 &&
                                                  dim.stride_in1 == 1 &&
                                                  dim.stride_out == 1;
                                       });

        if (l_dim_m_it != dimensions.end())
        {
            // set dimension data
            l_dim_m_it->type = dim_t::m;
            l_dim_m_it->exec_type = exec_t::prim;
            if (l_dim_m_it != dimensions.end() - 1)
            {
                // move M to the back
                std::rotate(l_dim_m_it, l_dim_m_it + 1, dimensions.end());
            }
        }
        else
        {
            throw std::invalid_argument("Optimizer: No suitable primary dimension M found.");
        }

        /////////////////////////////////////////////////////////////////
        // FIND PRIM N
        /////////////////////////////////////////////////////////////////
        // req: choose the one with smallest stride
        int l_n_dim_stride = INT_MAX;
        int l_n_dim_id = -1;
        for (size_t i = 0; i < dimensions.size(); i++)
        {
            if (dimensions[i].type == dim_t::n &&
                dimensions[i].stride_in0 == dimensions[i].stride_in1)
            {
                if (dimensions[i].stride_in0 < l_n_dim_stride)
                {
                    l_n_dim_stride = dimensions[i].stride_in0;
                    l_n_dim_id = static_cast<int>(i);
                }
            }
        }

        if (l_n_dim_id == -1)
        {
            throw std::invalid_argument("Optimizer: No suitable primary dimension N found.");
        }

        // set dimension data
        dimensions[l_n_dim_id].type = dim_t::n;
        dimensions[l_n_dim_id].exec_type = exec_t::prim;
        // move N to the back
        if (l_n_dim_id != static_cast<int>(dimensions.size()) - 1)
        {
            std::rotate(dimensions.begin() + l_n_dim_id, dimensions.begin() + l_n_dim_id + 1, dimensions.end());
        }
    }
    // TERNARY CASE
    else
    {
        /////////////////////////////////////////////////////////////////
        // FIND PRIM BR (second K)
        /////////////////////////////////////////////////////////////////
        // req: no unit stride in in1, stride_out has to be 0
        auto l_dim_BR_it = std::find_if(dimensions.begin(), dimensions.end(),
                                        [](const mini_jit::ir::Dimension &dim)
                                        {
                                            return dim.type == dim_t::k &&
                                                   dim.stride_in1 != 1 &&
                                                   dim.stride_out == 0;
                                        });

        if (l_dim_BR_it != dimensions.end())
        {
            // set dimension data
            l_dim_BR_it->type = dim_t::k;
            l_dim_BR_it->exec_type = exec_t::prim;
            if (l_dim_BR_it != dimensions.end() - 1)
            {
                // move BR(K) to the back
                std::rotate(l_dim_BR_it, l_dim_BR_it + 1, dimensions.end());
            }
        }
        /////////////////////////////////////////////////////////////////
        // FIND PRIM M
        /////////////////////////////////////////////////////////////////
        // req: unit stride in in0 and out
        auto l_dim_m_it = std::find_if(dimensions.begin(), dimensions.end(),
                                       [](const mini_jit::ir::Dimension &dim)
                                       {
                                           return dim.type == dim_t::m &&
                                                  dim.stride_in0 == 1 &&
                                                  dim.stride_in1 == 0 &&
                                                  dim.stride_out == 1;
                                       });

        if (l_dim_m_it != dimensions.end())
        {
            // set dimension data
            l_dim_m_it->type = dim_t::m;
            l_dim_m_it->exec_type = exec_t::prim;
            if (l_dim_m_it != dimensions.end() - 1)
            {
                // move M to the back
                std::rotate(l_dim_m_it, l_dim_m_it + 1, dimensions.end());
            }
        }
        else
        {
            throw std::invalid_argument("Optimizer: No suitable primary dimension M found.");
        }

        /////////////////////////////////////////////////////////////////
        // FIND PRIM N
        /////////////////////////////////////////////////////////////////
        // req: choose the one with the smallest strides, stride_in0 has to be 0
        int l_n_dim_strides = INT_MAX;
        int l_n_dim_id = -1;
        for (size_t i = 0; i < dimensions.size(); i++)
        {
            if (dimensions[i].type == dim_t::n &&
                dimensions[i].stride_in0 == 0)
            {
                int l_current_strides = dimensions[i].stride_in1 + dimensions[i].stride_out;
                if (l_current_strides < l_n_dim_strides)
                {
                    l_n_dim_strides = l_current_strides;
                    l_n_dim_id = static_cast<int>(i);
                }
            }
        }

        if (l_n_dim_id == -1)
        {
            throw std::invalid_argument("Optimizer: No suitable primary dimension N found.");
        }

        // set dimension data
        dimensions[l_n_dim_id].type = dim_t::n;
        dimensions[l_n_dim_id].exec_type = exec_t::prim;
        // move N to the back
        if (l_n_dim_id != static_cast<int>(dimensions.size()) - 1)
        {
            std::rotate(dimensions.begin() + l_n_dim_id, dimensions.begin() + l_n_dim_id + 1, dimensions.end());
        }
        /////////////////////////////////////////////////////////////////
        // FIND PRIM K
        /////////////////////////////////////////////////////////////////
        // req: unit stride in in1, stride_out has to be 0
        auto l_dim_k_it = std::find_if(dimensions.begin(), dimensions.end(),
                                       [](const mini_jit::ir::Dimension &dim)
                                       {
                                           return dim.type == dim_t::k &&
                                                  dim.stride_in1 == 1 &&
                                                  dim.stride_out == 0;
                                       });

        if (l_dim_k_it != dimensions.end())
        {
            // set dimension data
            l_dim_k_it->type = dim_t::k;
            l_dim_k_it->exec_type = exec_t::prim;
            if (l_dim_k_it != dimensions.end() - 1)
            {
                // move K to the back
                std::rotate(l_dim_k_it, l_dim_k_it + 1, dimensions.end());
            }
        }
        else
        {
            throw std::invalid_argument("Optimizer: No suitable primary dimension K found.");
        }
    }
}

void mini_jit::ir::Optimizer::splitDimensions(std::vector<mini_jit::ir::Dimension> &dimensions,
                                              int64_t max_kernel_size)
{
    // Dimensions should be split if they are too large (> max_kernel_size)
    for (size_t i = 0; i < dimensions.size(); i++)
    {
        if (dimensions[i].size > max_kernel_size)
        {
            int64_t l_size_dim_0 = 0;
            int64_t l_size_dim_1 = 0;
            findBestSplit(dimensions[i].size,
                          max_kernel_size,
                          dimensions[i].type,
                          l_size_dim_0,
                          l_size_dim_1);
            if (l_size_dim_0 > 1)
            {
                // create a new seq dimension
                mini_jit::ir::Dimension l_dim_new(dimensions[i].type,
                                                  exec_t::seq,
                                                  l_size_dim_0,
                                                  dimensions[i].stride_in0 * l_size_dim_1,
                                                  dimensions[i].stride_in1 * l_size_dim_1,
                                                  dimensions[i].stride_out * l_size_dim_1);
                // update the original dimension size
                dimensions[i].size = l_size_dim_1;
                // insert the new dimension at the start
                dimensions.insert(dimensions.begin(), l_dim_new);
                i++; // skip over new dimension
            }
        }
    }
}

void mini_jit::ir::Optimizer::createSharedLoops(std::vector<mini_jit::ir::Dimension> &dimensions,
                                                int64_t thread_target)
{
    int64_t l_num_threads = 1;

    // Count the number of possible iterations for shared loops
    for (size_t i = 0; i < dimensions.size(); i++)
    {
        if (dimensions[i].exec_type == exec_t::shared)
        {
            // increase thread number for each existing shared dimension
            l_num_threads *= dimensions[i].size;
        }
    }

    if (l_num_threads >= thread_target)
    {
        // make sure that the shared loops are at the front
        std::stable_partition(dimensions.begin(), dimensions.end(),
                              [](const mini_jit::ir::Dimension &dim)
                              {
                                  return dim.exec_type == exec_t::shared;
                              });
        // no need to create more shared loops
        return;
    }
    
    // Creation of new shared loops:
    for (size_t i = 0; i < dimensions.size(); i++)
    {
        // if the dimension can be set to shared and we did not reach the target number of threads yet
        // we set the dimension to shared
        // also dont parallelize the k dimension (see class slides)
        if ((dimensions[i].exec_type == exec_t::seq || dimensions[i].exec_type == exec_t::undefined) &&
            dimensions[i].type != dim_t::k &&
            l_num_threads * dimensions[i].size <= thread_target)
        {
            dimensions[i].exec_type = exec_t::shared;
            l_num_threads *= dimensions[i].size;
        }
    }

    // Move all shared loops to the front
    std::stable_partition(dimensions.begin(), dimensions.end(),
                          [](const mini_jit::ir::Dimension &dim)
                          {
                              return dim.exec_type == exec_t::shared;
                          });
}

void mini_jit::ir::Optimizer::findBestSplit(int64_t i_size,
                                            int64_t i_max_kernel_size,
                                            dim_t i_type,
                                            int64_t &o_size_0,
                                            int64_t &o_size_1)
{
    o_size_0 = 1;
    o_size_1 = i_size;
    // Optimal sizes:y
    //  M: 16x
    //  N: 4y
    //  K: z

    if (i_type == dim_t::m)
    {
        // multiples of (multiples of) 4 are efficient (LDP, STP)
        for (int64_t i = 16; i > 4; i -= 4)
        {
            findLargestMultipleOfDivisor(i, i_size, i_max_kernel_size, o_size_0, o_size_1);
            if (o_size_0 > 1)
            {
                return;
            }
        }
        // split by 2
        findLargestMultipleOfDivisor(2, i_size, i_max_kernel_size, o_size_0, o_size_1);
        if (o_size_0 > 1)
        {
            return;
        }
    }
    // for n, we want multiples of 4
    else if (i_type == dim_t::n)
    {
        // split by 4
        findLargestMultipleOfDivisor(4, i_size, i_max_kernel_size, o_size_0, o_size_1);
        if (o_size_0 > 1)
        {
            return;
        }
        // split by 2
        findLargestMultipleOfDivisor(2, i_size, i_max_kernel_size, o_size_0, o_size_1);
        if (o_size_0 > 1)
        {
            return;
        }
    }
    // k doesnt really matter
    else if (i_type == dim_t::k)
    {
        // split by 2
        findLargestMultipleOfDivisor(2, i_size, i_max_kernel_size, o_size_0, o_size_1);
        if (o_size_0 > 1)
        {
            return;
        }
    }
    else if (i_type == dim_t::c)
    {
        // identity uses M=8 and N=1

        // split by 8
        findLargestMultipleOfDivisor(8, i_size, i_max_kernel_size, o_size_0, o_size_1);
        if (o_size_0 > 1)
        {
            return;
        }

        // split by 4
        findLargestMultipleOfDivisor(4, i_size, i_max_kernel_size, o_size_0, o_size_1);
        if (o_size_0 > 1)
        {
            return;
        }

        // if 8 and 4 did not work, we try 2
        findLargestMultipleOfDivisor(2, i_size, i_max_kernel_size, o_size_0, o_size_1);
        if (o_size_0 > 1)
        {
            return;
        }
    }
    else
    {
        // undefined
        return;
    }
}

void mini_jit::ir::Optimizer::findLargestMultipleOfDivisor(int64_t i_divisor,
                                                           int64_t i_size,
                                                           int64_t i_max_size,
                                                           int64_t &o_size_0,
                                                           int64_t &o_size_1)
{
    o_size_0 = 1;
    o_size_1 = i_size;

    if (i_divisor <= 0 || i_size <= 0 || i_max_size <= 0)
    {
        return;
    }

    // start: largest multiple of i_divisor < i_max_size
    int64_t l_max_divisible = (i_max_size / i_divisor) * i_divisor;

    for (int64_t l_m = l_max_divisible; l_m >= i_divisor; l_m -= i_divisor)
    {
        // we found an m that divides i_size! it is also the largest
        if (i_size % l_m == 0)
        {
            o_size_1 = l_m;
            o_size_0 = i_size / l_m;
            return;
        }
    }
}

void mini_jit::ir::Optimizer::fuseDimensions(std::vector<mini_jit::ir::Dimension> &dimensions,
                                             int64_t min_kernel_size)
{
    // Dimensions should be fused if they are small enough (< min_kernel_size)
    // Config object: Two dimensions X and Y can be fused can be fused if for all tensors: stride(X) = |Y| ⨉ stride(Y).
    // For both dimensions X and Y, the type has to be the same, and the execution type (exec.type) has to be the same or undefined.
    for (size_t i = 0; i < dimensions.size(); i++)
    {
        mini_jit::ir::Dimension &l_dim_0 = dimensions[i];
        if (l_dim_0.size < min_kernel_size)
        {
            // find a dimension that can be fused with the current one
            for (size_t j = 0; j < dimensions.size(); j++)
            {
                if (i == j) continue; // skip self

                mini_jit::ir::Dimension &l_dim_1 = dimensions[j];
                if (l_dim_0.type == l_dim_1.type &&
                    (l_dim_0.exec_type == l_dim_1.exec_type ||
                     l_dim_0.exec_type == exec_t::undefined ||
                     l_dim_1.exec_type == exec_t::undefined) &&
                    l_dim_1.stride_in0 == l_dim_0.size * l_dim_0.stride_in0 &&
                    l_dim_1.stride_in1 == l_dim_0.size * l_dim_0.stride_in1 &&
                    l_dim_1.stride_out == l_dim_0.size * l_dim_0.stride_out)
                {
                    // fuse the two dimensions and keep the smaller stride
                    l_dim_0.size += l_dim_1.size;
                    l_dim_0.stride_in0 = std::min(l_dim_0.stride_in0, l_dim_1.stride_in0);
                    l_dim_0.stride_in1 = std::min(l_dim_0.stride_in1, l_dim_1.stride_in1);
                    l_dim_0.stride_out = std::min(l_dim_0.stride_out, l_dim_1.stride_out);
                    // remove the fused dimension
                    dimensions.erase(dimensions.begin() + j);
                    j--; // adjust index after erasing
                }
            }
        }
    }
}