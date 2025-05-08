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

TEST_CASE("Tests the Base MOV (register) instruction generation", "[MOV_REG]")
{
    mini_jit::InstGen l_gen;
    uint32_t l_ins = mini_jit::InstGen::mov_reg(mini_jit::InstGen::x2,
                                                mini_jit::InstGen::x1);
    std::string l_hex = l_gen.to_string_hex(l_ins);
    REQUIRE(l_hex == "0xaa0103e2");
}

TEST_CASE("Tests the Base MOV (immediate) instruction generation", "[MOV_IMM]")
{
    mini_jit::InstGen l_gen;
    uint32_t l_ins = mini_jit::InstGen::mov_imm(mini_jit::InstGen::x0, 15);
    std::string l_hex = l_gen.to_string_hex(l_ins);
    REQUIRE(l_hex == "0xd28001e0");
}

TEST_CASE("Tests the Base LDR instruction generation", "[Base LDR]")
{
    mini_jit::InstGen l_gen;

    uint32_t l_ins = mini_jit::InstGen::base_ldr_imm_uoff(mini_jit::InstGen::x1, mini_jit::InstGen::x0, 16);
    std::string l_hex = l_gen.to_string_hex(l_ins);
    REQUIRE(l_hex == "0xf9400801");
}

TEST_CASE("Tests the Base LDP instruction generation", "[Base LDP]")
{
    mini_jit::InstGen l_gen;

    uint32_t l_ins = mini_jit::InstGen::base_ldp_soff(mini_jit::InstGen::x2, mini_jit::InstGen::x3, mini_jit::InstGen::x0, -16);
    std::string l_hex = l_gen.to_string_hex(l_ins);
    REQUIRE(l_hex == "0xa97f0c02");

    l_ins = mini_jit::InstGen::base_ldp_post(mini_jit::InstGen::x2, mini_jit::InstGen::x3, mini_jit::InstGen::x0, 16);
    l_hex = l_gen.to_string_hex(l_ins);
    REQUIRE(l_hex == "0xa8c10c02");

    l_ins = mini_jit::InstGen::base_ldp_pre(mini_jit::InstGen::x2, mini_jit::InstGen::x3, mini_jit::InstGen::x0, 16);
    l_hex = l_gen.to_string_hex(l_ins);
    REQUIRE(l_hex == "0xa9c10c02");
}

TEST_CASE("Tests the Neon LDP instruction generation", "[Neon LDP]")
{
    mini_jit::InstGen l_gen;

    uint32_t l_ins = mini_jit::InstGen::neon_ldp_soff(mini_jit::InstGen::v1, mini_jit::InstGen::v2, mini_jit::InstGen::x0, -16, mini_jit::InstGen::d);
    std::string l_hex = l_gen.to_string_hex(l_ins);
    REQUIRE(l_hex == "0x6d7f0801");

    l_ins = mini_jit::InstGen::neon_ldp_post(mini_jit::InstGen::v1, mini_jit::InstGen::v2, mini_jit::InstGen::x0, 16, mini_jit::InstGen::d);
    l_hex = l_gen.to_string_hex(l_ins);
    REQUIRE(l_hex == "0x6cc10801");

    l_ins = mini_jit::InstGen::neon_ldp_pre(mini_jit::InstGen::v1, mini_jit::InstGen::v2, mini_jit::InstGen::x0, 16, mini_jit::InstGen::d);
    l_hex = l_gen.to_string_hex(l_ins);
    REQUIRE(l_hex == "0x6dc10801");
}
