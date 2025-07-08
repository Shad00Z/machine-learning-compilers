#ifndef MINI_JIT_CONVERTERS_INST_TO_HEX_STRING_H
#define MINI_JIT_CONVERTERS_INST_TO_HEX_STRING_H

#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

namespace mini_jit
{
    namespace converters
    {
        /**
         * @brief Converts the given instruction to a hex string.
         *
         * @param inst instruction.
         *
         * @return hex string.
         **/
        inline const std::string to_string_hex(uint32_t inst)
        {
            std::stringstream l_ss;
            l_ss << "0x" << std::hex
                 << std::setfill('0')
                 << std::setw(8)
                 << inst;

            return l_ss.str();
        }
    } // namespace converters
} // namespace mini_jit

#endif // MINI_JIT_CONVERTERS_INST_TO_HEX_STRING_H