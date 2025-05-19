#include <catch2/catch.hpp>

#include "converters/instToBinString.h"
#include "converters/instToHexString.h"

#include "registers/gp_registers.h"
#include "instructions/ret.h"
#include "instructions/base/all_base_instructions.h"

using gpr_t = mini_jit::registers::gpr_t;
using namespace mini_jit::converters;
namespace inst = mini_jit::instructions;
namespace base = inst::base;

TEST_CASE("Tests the ret instruction generation", "[RET]")
{
    uint32_t l_ins = inst::ret();
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0xd65f03c0");
}

TEST_CASE("Tests the Base CBNZ instruction generation", "[CBNZ]")
{
    uint32_t l_ins = base::cbnz(gpr_t::x0, 0);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0xb5000000");
}

TEST_CASE("Tests the Base ORR (shifted register) instruction generation", "[ORR]")
{
    uint32_t l_ins = base::orr(gpr_t::x1,
                               gpr_t::x0,
                               gpr_t::x0,
                               0x0,
                               0x0);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0xaa000001");
}

TEST_CASE("Tests the Base MOV (register) instruction generation", "[MOV_REG]")
{
    uint32_t l_ins = base::mov(gpr_t::x2,
                               gpr_t::x1);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0xaa0103e2");
}

TEST_CASE("Tests the Base MOV (SP) instruction generation", "[MOV_SP]")
{
    uint32_t l_ins = base::movSP(gpr_t::x0,
                                 gpr_t::sp);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x910003e0");
}

TEST_CASE("Tests the Base MOV (immediate) instruction generation", "[MOV_IMM]")
{
    uint32_t l_ins = base::mov(gpr_t::x0, 15);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0xd28001e0");
}

TEST_CASE("Tests the Base LDR instruction generation", "[Base LDR]")
{
    uint32_t l_ins = base::ldr(gpr_t::x1, gpr_t::x0, 16);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0xf9400801");
}

TEST_CASE("Tests the Base LDP instruction generation", "[Base LDP]")
{
    uint32_t l_ins = base::ldp(gpr_t::x2, gpr_t::x3, gpr_t::x0, -16);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0xa97f0c02");

    l_ins = base::ldpPost(gpr_t::x2, gpr_t::x3, gpr_t::x0, 16);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0xa8c10c02");

    l_ins = base::ldpPre(gpr_t::x2, gpr_t::x3, gpr_t::x0, 16);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0xa9c10c02");
}

TEST_CASE("Tests the Base STR instruction generation", "[Base STR]")
{
    uint32_t l_ins = base::str(gpr_t::x1, gpr_t::x0, 16);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0xf9000801");

    l_ins = base::strPost(gpr_t::x1, gpr_t::x0, 16);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0xf8010401");
}

TEST_CASE("Tests the Base STP instruction generation", "[Base STP]")
{
    uint32_t l_ins = base::stp(gpr_t::x2, gpr_t::x3, gpr_t::x0, -16);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0xa93f0c02");

    l_ins = base::stpPost(gpr_t::x2, gpr_t::x3, gpr_t::x0, 16);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0xa8810c02");

    l_ins = base::stpPre(gpr_t::x2, gpr_t::x3, gpr_t::x0, 16);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0xa9810c02");
}

TEST_CASE("Tests the MUL (register) instruction generation", "[MUL_REG]")
{
    uint32_t l_ins = base::mul(gpr_t::x2, gpr_t::x0, gpr_t::x1);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x9b017c02");
}

TEST_CASE("Tests the ADD (shifted register) instruction generation", "[ADD_SREG]")
{
    uint32_t l_ins = base::add(gpr_t::x2, gpr_t::x0, gpr_t::x1, 4, 0);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x8b011002");
}

TEST_CASE("Tests the ADD (immediate) instruction generation", "[ADD_IMM]")
{
    uint32_t l_ins = base::add(gpr_t::x2, gpr_t::x0, 16, 0);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x91004002");
}

TEST_CASE("Tests the LSL (immediate) instruction generation", "[LSL_IMM]")
{
    uint32_t l_ins = base::lsl(gpr_t::x3, gpr_t::x3, 2);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0xd37ef463");
}