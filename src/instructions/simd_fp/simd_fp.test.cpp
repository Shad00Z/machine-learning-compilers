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

TEST_CASE("Tests the Neon LD1 (single structure) with a lane index instruction generation", "[Neon LD1 Single Structure Index]")
{
    uint32_t l_ins = simd_fp::ld1(simd_fp_t::v0, gpr_t::x0, 3, neon_size_spec_t::s);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4d409000");

    l_ins = simd_fp::ld1(simd_fp_t::v5, gpr_t::x1, 1, neon_size_spec_t::d);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4d408425");

    CHECK_THROWS_AS(simd_fp::ld1(simd_fp_t::v0, gpr_t::x0, 4, neon_size_spec_t::s), std::out_of_range);
    CHECK_THROWS_AS(simd_fp::ld1(simd_fp_t::v0, gpr_t::x0, 2, neon_size_spec_t::d), std::out_of_range);
}

TEST_CASE("Tests the Neon LD1 (single structure) with a lane index and a register post-index instruction generation", "[Neon LD1 Single Structure Index Post-Index Register]")
{
    uint32_t l_ins = simd_fp::ld1(simd_fp_t::v0, gpr_t::x0, 3, neon_size_spec_t::s, gpr_t::x1);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4dc19000");
}

TEST_CASE("Tests the Neon LD1 (single structure) with a lane index and a post-index immediate instruction generation", "[Neon LD1 Single Structure Index Post-Index Immediate]")
{
    uint32_t l_ins = simd_fp::ld1(simd_fp_t::v0, gpr_t::x0, 3, neon_size_spec_t::s, 4);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4ddf9000");

    l_ins = simd_fp::ld1(simd_fp_t::v0, gpr_t::x0, 1, neon_size_spec_t::d, 8);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4ddf8400");

    CHECK_THROWS_AS(simd_fp::ld1(simd_fp_t::v0, gpr_t::x1, 1, neon_size_spec_t::d, 4), std::invalid_argument);
    CHECK_THROWS_AS(simd_fp::ld1(simd_fp_t::v0, gpr_t::x1, 1, neon_size_spec_t::s, 8), std::invalid_argument);
}
