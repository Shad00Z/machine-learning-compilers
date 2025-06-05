#include "Optimizer.h"
#include <algorithm>
#include <limits.h>

void mini_jit::ir::Optimizer::optimize(std::vector<mini_jit::ir::Dimension> &dimensions,
                                       int64_t thread_target,
                                       int64_t max_kernel_size)
{
    identifyPrimitives(dimensions);

    // Verify that there are 3 or 4 primitive dimensions
    int prim_count = std::count_if(dimensions.begin(), dimensions.end(),
                                   [](const mini_jit::ir::Dimension &dim)
                                   {
                                       return dim.exec_type == exec_t::prim;
                                   });
    if (prim_count < 3 || prim_count > 4)
    {
        throw std::invalid_argument("Optimizer: Expected 3 or 4 primitive dimensions, found " + std::to_string(prim_count) + ". Try setting all dimensions to seq or undefined.");
    }

    splitDimensions(dimensions,
                    max_kernel_size);

    createSharedLoops(dimensions,
                      thread_target);

    // TODO: Dimension Fusion, Reorder?
}

void mini_jit::ir::Optimizer::identifyPrimitives(std::vector<mini_jit::ir::Dimension> &dimensions)
{
    /////////////////////////////////////////////////////////////////
    // FIND PRIM BR (second K)
    /////////////////////////////////////////////////////////////////
    // req: no unit stride in in1, stride_out has to be 0
    auto l_dim_BR_it = std::find_if(dimensions.begin(), dimensions.end(),
                                    [](const mini_jit::ir::Dimension &dim)
                                    {
                                        return (dim.type == dim_t::k || dim.type == dim_t::undefined) &&
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
                                       return (dim.type == dim_t::m || dim.type == dim_t::undefined) &&
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
        if ((dimensions[i].type == dim_t::n || dimensions[i].type == dim_t::undefined) &&
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
                                       return (dim.type == dim_t::k || dim.type == dim_t::undefined) &&
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

void mini_jit::ir::Optimizer::splitDimensions(std::vector<mini_jit::ir::Dimension> &dimensions,
                                              int64_t max_kernel_size)
{
    // Primitive dimensions should be split if they are too large (> 1024)
    for (size_t i = 0; i < dimensions.size(); i++)
    {
        if (dimensions[i].exec_type == exec_t::prim && dimensions[i].size > 1024)
        {
            int64_t l_size_seq = 0;
            int64_t l_size_prim = 0;
            findBestSplit(dimensions[i].size,
                          max_kernel_size,
                          dimensions[i].type,
                          l_size_seq,
                          l_size_prim);
            if (l_size_seq > 1)
            {
                // create a new seq dimension
                mini_jit::ir::Dimension l_dim_seq(dimensions[i].type,
                                                  exec_t::seq,
                                                  l_size_seq,
                                                  dimensions[i].stride_in0 * l_size_prim,
                                                  dimensions[i].stride_in1 * l_size_prim,
                                                  dimensions[i].stride_out * l_size_prim);
                // update the prim dimension size
                dimensions[i].size = l_size_prim;
                // insert the new seq dimension at the start
                dimensions.insert(dimensions.begin(), l_dim_seq);
                i++; // skip over new seq dimension
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