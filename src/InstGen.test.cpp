#include <catch2/catch.hpp>
#include "InstGen.h"

TEST_CASE("Tests the ret instruction generation", "[RET]")
{
    mini_jit::InstGen l_gen;
    uint32_t l_ins = mini_jit::InstGen::ret();
    std::string l_hex = l_gen.to_string_hex(l_ins);
    REQUIRE(l_hex == "0xd65f03c0");
}

TEST_CASE("Tests the Base CBNZ instruction generation", "[CBNZ]")
{
    mini_jit::InstGen l_gen;
    uint32_t l_ins = mini_jit::InstGen::base_br_cbnz(mini_jit::InstGen::x0, 0);
    std::string l_hex = l_gen.to_string_hex(l_ins);
    REQUIRE(l_hex == "0xb5000000");
}

TEST_CASE("Tests the Base ORR (shifted register) instruction generation", "[ORR]")
{
    mini_jit::InstGen l_gen;
    uint32_t l_ins = mini_jit::InstGen::base_orr_shifted_reg(mini_jit::InstGen::x1,
                                                             mini_jit::InstGen::x0,
                                                             mini_jit::InstGen::x0,
                                                             0x0,
                                                             0x0);
    std::string l_hex = l_gen.to_string_hex(l_ins);
    REQUIRE(l_hex == "0xaa000001");
}

TEST_CASE("Tests the Base MOV (register) instruction generation", "[MOV]")
{
    mini_jit::InstGen l_gen;
    uint32_t l_ins = mini_jit::InstGen::mov_reg(mini_jit::InstGen::x2,
                                                mini_jit::InstGen::x1);
    std::string l_hex = l_gen.to_string_hex(l_ins);
    REQUIRE(l_hex == "0xaa0103e2");
}