#include <catch2/catch.hpp>

#include "converters/instToBinString.h"
#include "converters/instToHexString.h"

#include "registers/gp_registers.h"
#include "registers/simd_fp_registers.h"
#include "instructions/ret.h"
#include "instructions/simd_fp/all_simd_fp_instructions.h"

using gpr_t = mini_jit::registers::gpr_t;
using simd_fp_t = mini_jit::registers::simd_fp_t;
using neon_size_spec_t = mini_jit::registers::neon_size_spec_t;
using namespace mini_jit::converters;
namespace inst = mini_jit::instructions;
namespace simd_fp = inst::simd_fp;

TEST_CASE("Tests the Neon LDR instruction generation", "[Neon LDR]")
{
    uint32_t l_ins = simd_fp::ldr(simd_fp_t::v28, gpr_t::x6, 0, neon_size_spec_t::s);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0xbd4000dc");
}

TEST_CASE("Tests the Neon LDP instruction generation", "[Neon LDP]")
{
    uint32_t l_ins = simd_fp::ldp(simd_fp_t::v1, simd_fp_t::v2, gpr_t::x0, -16, neon_size_spec_t::d);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x6d7f0801");

    l_ins = simd_fp::ldpPost(simd_fp_t::v1, simd_fp_t::v2, gpr_t::x0, 16, neon_size_spec_t::d);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x6cc10801");

    l_ins = simd_fp::ldpPre(simd_fp_t::v1, simd_fp_t::v2, gpr_t::x0, 16, neon_size_spec_t::d);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x6dc10801");
}

TEST_CASE("Tests the Neon STP instruction generation", "[Neon STP]")
{
    uint32_t l_ins = simd_fp::stp(simd_fp_t::v1, simd_fp_t::v2, gpr_t::x0, -16, neon_size_spec_t::d);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x6d3f0801");

    l_ins = simd_fp::stpPost(simd_fp_t::v1, simd_fp_t::v2, gpr_t::x0, 16, neon_size_spec_t::d);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x6c810801");

    l_ins = simd_fp::stpPre(simd_fp_t::v1, simd_fp_t::v2, gpr_t::x0, 16, neon_size_spec_t::d);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x6d810801");
}

TEST_CASE("Tests the Neon FMLA (by element) instruction generation", "[Neon_FMLA_ELEM]")
{
    uint32_t l_ins = simd_fp::fmlaElem(simd_fp_t::v4, simd_fp_t::v0, simd_fp_t::v28, arr_spec_t::s4);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4f9c1004");
}
