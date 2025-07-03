#ifndef MINI_JIT_INSTRUCTIONS_BASE_RET_H
#define MINI_JIT_INSTRUCTIONS_BASE_RET_H

#include <cstdint>

namespace mini_jit
{
    namespace instructions
    {
        namespace base
        {
            /**
             * @brief Generates a RET instruction.
             *
             * @return instruction.
             */
            constexpr uint32_t ret()
            {
                return 0xd65f03c0;
            }
        }
    }
}

#endif // MINI_JIT_INSTRUCTIONS_BASE_RET_H