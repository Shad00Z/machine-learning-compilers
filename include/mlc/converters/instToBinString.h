#ifndef MINI_JIT_CONVERTERS_INST_TO_BIN_STRING_H
#define MINI_JIT_CONVERTERS_INST_TO_BIN_STRING_H

#include <bitset>
#include <cstdint>
#include <string>

namespace mini_jit
{
    namespace converters
    {
        /**
         * @brief Converts the given instruction to a binary string.
         *
         * @param inst instruction.
         *
         * @return binary string.
         **/
        inline const std::string to_string_bin(uint32_t inst)
        {
            std::string l_res = "0b";
            l_res += std::bitset<32>(inst).to_string();

            return l_res;
        }
    } // namespace converters
} // namespace mini_jit

#endif // MINI_JIT_CONVERTERS_INST_TO_BIN_STRING_H