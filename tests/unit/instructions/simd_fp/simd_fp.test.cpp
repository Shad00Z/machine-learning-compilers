#include <catch2/catch.hpp>
#include <mlc/converters/instToBinString.h>
#include <mlc/converters/instToHexString.h>
#include <mlc/instructions/simd_fp/all_simd_fp_instructions.h>
#include <mlc/registers/gp_registers.h>
#include <mlc/registers/simd_fp_registers.h>

using gpr_t            = mini_jit::registers::gpr_t;
using simd_fp_t        = mini_jit::registers::simd_fp_t;
using neon_size_spec_t = mini_jit::registers::neon_size_spec_t;
using arr_spec_t       = mini_jit::registers::arr_spec_t;
using namespace mini_jit::converters;
namespace inst    = mini_jit::instructions;
namespace simd_fp = inst::simd_fp;

TEST_CASE("Tests the Neon FABS instruction generation", "[Neon_FABS]")
{
    uint32_t    l_ins = simd_fp::fabsVec(simd_fp_t::v5, simd_fp_t::v4, arr_spec_t::s2);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x0ea0f885");

    l_ins = simd_fp::fabsVec(simd_fp_t::v7, simd_fp_t::v6, arr_spec_t::s4);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4ea0f8c7");

    l_ins = simd_fp::fabsVec(simd_fp_t::v9, simd_fp_t::v8, arr_spec_t::d2);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4ee0f909");

    l_ins = simd_fp::fabsScalar(simd_fp_t::v11, simd_fp_t::v10, neon_size_spec_t::s);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x1e20c14b");

    l_ins = simd_fp::fabsScalar(simd_fp_t::v13, simd_fp_t::v12, neon_size_spec_t::d);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x1e60c18d");
}

TEST_CASE("Tests the Neon FCVTMS instruction generation", "[Neon_FCVTMS]")
{
    uint32_t    l_ins = simd_fp::fcvtmsVec(simd_fp_t::v5, simd_fp_t::v4, arr_spec_t::s2);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x0e21b885");

    l_ins = simd_fp::fcvtmsVec(simd_fp_t::v7, simd_fp_t::v6, arr_spec_t::s4);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4e21b8c7");

    l_ins = simd_fp::fcvtmsVec(simd_fp_t::v9, simd_fp_t::v8, arr_spec_t::d2);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4e61b909");

    l_ins = simd_fp::fcvtmsScalar(simd_fp_t::v11, simd_fp_t::v10, neon_size_spec_t::s);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x5e21b94b");

    l_ins = simd_fp::fcvtmsScalar(simd_fp_t::v13, simd_fp_t::v12, neon_size_spec_t::d);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x5e61b98d");
}

TEST_CASE("Tests the Neon SCVTF instruction generation", "[Neon_SCVTF]")
{
    uint32_t    l_ins = simd_fp::scvtfVec(simd_fp_t::v5, simd_fp_t::v4, arr_spec_t::s2);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x0e21d885");

    l_ins = simd_fp::scvtfVec(simd_fp_t::v7, simd_fp_t::v6, arr_spec_t::s4);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4e21d8c7");

    l_ins = simd_fp::scvtfVec(simd_fp_t::v8, simd_fp_t::v7, arr_spec_t::s4);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4e21d8e8");

    l_ins = simd_fp::scvtfVec(simd_fp_t::v9, simd_fp_t::v8, arr_spec_t::d2);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4e61d909");

    l_ins = simd_fp::scvtfScalar(simd_fp_t::v11, simd_fp_t::v10, neon_size_spec_t::s);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x5e21d94b");

    l_ins = simd_fp::scvtfScalar(simd_fp_t::v13, simd_fp_t::v12, neon_size_spec_t::d);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x5e61d98d");
}

TEST_CASE("Tests the Neon FRINTN instruction generation", "[Neon_FRINTN]")
{
    uint32_t    l_ins = simd_fp::frintnVec(simd_fp_t::v5, simd_fp_t::v4, arr_spec_t::s2);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x0e218885");

    l_ins = simd_fp::frintnVec(simd_fp_t::v7, simd_fp_t::v6, arr_spec_t::s4);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4e2188c7");

    l_ins = simd_fp::frintnVec(simd_fp_t::v8, simd_fp_t::v7, arr_spec_t::s4);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4e2188e8");

    l_ins = simd_fp::frintnVec(simd_fp_t::v9, simd_fp_t::v8, arr_spec_t::d2);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4e618909");

    l_ins = simd_fp::frintnScalar(simd_fp_t::v11, simd_fp_t::v10, neon_size_spec_t::s);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x1e24414b");

    l_ins = simd_fp::frintnScalar(simd_fp_t::v13, simd_fp_t::v12, neon_size_spec_t::d);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x1e64418d");
}

TEST_CASE("Tests the Neon FRINTM instruction generation", "[Neon_FRINTM]")
{
    uint32_t    l_ins = simd_fp::frintmVec(simd_fp_t::v5, simd_fp_t::v4, arr_spec_t::s2);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x0e219885");

    l_ins = simd_fp::frintmVec(simd_fp_t::v7, simd_fp_t::v6, arr_spec_t::s4);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4e2198c7");

    l_ins = simd_fp::frintmVec(simd_fp_t::v8, simd_fp_t::v7, arr_spec_t::s4);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4e2198e8");

    l_ins = simd_fp::frintmVec(simd_fp_t::v9, simd_fp_t::v8, arr_spec_t::d2);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4e619909");

    l_ins = simd_fp::frintmScalar(simd_fp_t::v11, simd_fp_t::v10, neon_size_spec_t::s);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x1e25414b");

    l_ins = simd_fp::frintmScalar(simd_fp_t::v13, simd_fp_t::v12, neon_size_spec_t::d);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x1e65418d");
}

TEST_CASE("Tests the Neon LDR instruction generation", "[Neon_LDR]")
{
    uint32_t    l_ins = simd_fp::ldr(simd_fp_t::v28, gpr_t::x6, 0, neon_size_spec_t::s);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0xbd4000dc");

    l_ins = simd_fp::ldrPost(simd_fp_t::v28, gpr_t::x6, 16, neon_size_spec_t::s);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0xbc4104dc");

    l_ins = simd_fp::ldrPost(simd_fp_t::v12, gpr_t::x9, 24, neon_size_spec_t::d);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0xfc41852c");

    l_ins = simd_fp::ldrPost(simd_fp_t::v0, gpr_t::x20, 8, neon_size_spec_t::d);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0xfc408680");

    l_ins = simd_fp::ldrPost(simd_fp_t::v7, gpr_t::x11, 32, neon_size_spec_t::q);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x3cc20567");

    l_ins = simd_fp::ldrReg(simd_fp_t::v10, gpr_t::x5, gpr_t::w10, 0, neon_size_spec_t::s);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0xbc6a48aa");
}

TEST_CASE("Tests the Neon LDP instruction generation", "[Neon_LDP]")
{
    uint32_t    l_ins = simd_fp::ldp(simd_fp_t::v1, simd_fp_t::v2, gpr_t::x0, -16, neon_size_spec_t::d);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x6d7f0801");

    l_ins = simd_fp::ldpPost(simd_fp_t::v1, simd_fp_t::v2, gpr_t::x0, 16, neon_size_spec_t::d);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x6cc10801");

    l_ins = simd_fp::ldpPre(simd_fp_t::v1, simd_fp_t::v2, gpr_t::x0, 16, neon_size_spec_t::d);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x6dc10801");
}

TEST_CASE("Tests the Neon STR instruction generation", "[Neon_STR]")
{
    uint32_t    l_ins = simd_fp::str(simd_fp_t::v0, gpr_t::x12, 0, neon_size_spec_t::s);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0xbd000180");

    l_ins = simd_fp::strPost(simd_fp_t::v28, gpr_t::x6, 16, neon_size_spec_t::s);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0xbc0104dc");

    l_ins = simd_fp::strPost(simd_fp_t::v12, gpr_t::x9, 24, neon_size_spec_t::d);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0xfc01852c");

    l_ins = simd_fp::strPost(simd_fp_t::v0, gpr_t::x20, 8, neon_size_spec_t::d);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0xfc008680");

    l_ins = simd_fp::strPost(simd_fp_t::v7, gpr_t::x11, 32, neon_size_spec_t::q);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x3c820567");
}

TEST_CASE("Tests the Neon STP instruction generation", "[Neon_STP]")
{
    uint32_t    l_ins = simd_fp::stp(simd_fp_t::v1, simd_fp_t::v2, gpr_t::x0, -16, neon_size_spec_t::d);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x6d3f0801");

    l_ins = simd_fp::stpPost(simd_fp_t::v1, simd_fp_t::v2, gpr_t::x0, 16, neon_size_spec_t::d);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x6c810801");

    l_ins = simd_fp::stpPre(simd_fp_t::v1, simd_fp_t::v2, gpr_t::x0, 16, neon_size_spec_t::d);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x6d810801");
}

TEST_CASE("Tests the Neon FADD (scalar) instruction generation", "[Neon_FADD_SCALAR]")
{
    uint32_t    l_ins = simd_fp::faddScalar(simd_fp_t::v0, simd_fp_t::v1, simd_fp_t::v2, neon_size_spec_t::s);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x1e222820");

    l_ins = simd_fp::faddScalar(simd_fp_t::v2, simd_fp_t::v3, simd_fp_t::v4, neon_size_spec_t::d);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x1e642862");
}

TEST_CASE("Tests the Neon FADD (vector) instruction generation", "[Neon_FADD_VEC]")
{
    uint32_t    l_ins = simd_fp::faddVec(simd_fp_t::v0, simd_fp_t::v1, simd_fp_t::v2, arr_spec_t::s2);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x0e22d420");

    l_ins = simd_fp::faddVec(simd_fp_t::v4, simd_fp_t::v7, simd_fp_t::v6, arr_spec_t::s4);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4e26d4e4");

    l_ins = simd_fp::faddVec(simd_fp_t::v9, simd_fp_t::v6, simd_fp_t::v8, arr_spec_t::d2);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4e68d4c9");
}

TEST_CASE("Tests the Neon FDIV (scalar) instruction generation", "[Neon_FDIV_SCALAR]")
{
    uint32_t    l_ins = simd_fp::fdivScalar(simd_fp_t::v0, simd_fp_t::v1, simd_fp_t::v2, neon_size_spec_t::s);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x1e221820");

    l_ins = simd_fp::fdivScalar(simd_fp_t::v2, simd_fp_t::v3, simd_fp_t::v4, neon_size_spec_t::d);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x1e641862");
}

TEST_CASE("Tests the Neon FDIV (vector) instruction generation", "[Neon_FDIV_VEC]")
{
    uint32_t    l_ins = simd_fp::fdivVec(simd_fp_t::v0, simd_fp_t::v1, simd_fp_t::v2, arr_spec_t::s2);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x2e22fc20");

    l_ins = simd_fp::fdivVec(simd_fp_t::v4, simd_fp_t::v7, simd_fp_t::v6, arr_spec_t::s4);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x6e26fce4");

    l_ins = simd_fp::fdivVec(simd_fp_t::v9, simd_fp_t::v6, simd_fp_t::v8, arr_spec_t::d2);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x6e68fcc9");
}

TEST_CASE("Tests the Neon FMOV (scalar, immediate) instruction generation", "[Neon_FMOV_SCALAR]")
{
    uint32_t    l_ins = simd_fp::fmovIntScalar(simd_fp_t::v0, 1, neon_size_spec_t::s);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x1e2e1000");

    l_ins = simd_fp::fmovIntScalar(simd_fp_t::v3, 7, neon_size_spec_t::s);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x1e239003");

    l_ins = simd_fp::fmovIntScalar(simd_fp_t::v5, 13, neon_size_spec_t::d);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x1e655005");

    l_ins = simd_fp::fmovIntScalar(simd_fp_t::v7, 21, neon_size_spec_t::d);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x1e66b007");
}

TEST_CASE("Tests the Neon UMOV instruction generation", "[Neon_UMOV_SCALAR]")
{
    uint32_t    l_ins = simd_fp::umov(gpr_t::w0, simd_fp_t::v1, 1, neon_size_spec_t::s);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x0e0c3c20");

    l_ins = simd_fp::umov(gpr_t::w3, simd_fp_t::v7, 2, neon_size_spec_t::s);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x0e143ce3");

    l_ins = simd_fp::umov(gpr_t::w3, simd_fp_t::v7, 3, neon_size_spec_t::s);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x0e1c3ce3");

    l_ins = simd_fp::umov(gpr_t::x3, simd_fp_t::v7, 0, neon_size_spec_t::d);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4e083ce3");

    l_ins = simd_fp::umov(gpr_t::x3, simd_fp_t::v7, 1, neon_size_spec_t::d);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4e183ce3");
}

TEST_CASE("Tests the Neon FMOV (vector, immediate) instruction generation", "[Neon_FMOV_VEC]")
{
    uint32_t    l_ins = simd_fp::fmovIntVec(simd_fp_t::v0, 1, arr_spec_t::s2);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x0f03f600");

    l_ins = simd_fp::fmovIntVec(simd_fp_t::v0, 2, arr_spec_t::s2);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x0f00f400");

    l_ins = simd_fp::fmovIntVec(simd_fp_t::v7, 9, arr_spec_t::s4);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4f01f447");

    l_ins = simd_fp::fmovIntVec(simd_fp_t::v11, 13, arr_spec_t::d2);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x6f01f54b");

    l_ins = simd_fp::fmovVec(simd_fp_t::v0, 0b01110000, arr_spec_t::s4);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4f03f600");

    l_ins = simd_fp::fmovVec(simd_fp_t::v1, 0b01100000, arr_spec_t::s4);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4f03f401");
}

TEST_CASE("Tests the Neon FMLA (by element) instruction generation", "[Neon_FMLA_ELEM]")
{
    uint32_t    l_ins = simd_fp::fmlaElem(simd_fp_t::v4, simd_fp_t::v0, simd_fp_t::v28, arr_spec_t::s4);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4f9c1004");
}

TEST_CASE("Tests the Neon FRECPE instruction generation", "[Neon_FRECPE]")
{
    uint32_t    l_ins = simd_fp::frecpeVec(simd_fp_t::v0, simd_fp_t::v1, arr_spec_t::s4);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4ea1d820");

    l_ins = simd_fp::frecpeScalar(simd_fp_t::v0, simd_fp_t::v2, neon_size_spec_t::s);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x5ea1d840");
}

TEST_CASE("Tests the Neon FRECPS instruction generation", "[Neon_FRECPS]")
{
    uint32_t    l_ins = simd_fp::frecpsVec(simd_fp_t::v2, simd_fp_t::v1, simd_fp_t::v0, arr_spec_t::s4);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4e20fc22");

    l_ins = simd_fp::frecpsScalar(simd_fp_t::v3, simd_fp_t::v2, simd_fp_t::v0, neon_size_spec_t::s);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x5e20fc43");
}

TEST_CASE("Tests the Neon FMUL (vector) instruction generation", "[Neon_FMUL_VEC]")
{
    uint32_t    l_ins = simd_fp::fmulVec(simd_fp_t::v2, simd_fp_t::v1, simd_fp_t::v1, arr_spec_t::s4);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x6e21dc22");
}

TEST_CASE("Tests the Neon FMUL (scalar) instruction generation", "[Neon_FMUL_SCALAR]")
{
    uint32_t    l_ins = simd_fp::fmulScalar(simd_fp_t::v0, simd_fp_t::v0, simd_fp_t::v1, neon_size_spec_t::s);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x1e210800");

    l_ins = simd_fp::fmulScalar(simd_fp_t::v2, simd_fp_t::v3, simd_fp_t::v4, neon_size_spec_t::d);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x1e640862");
}

TEST_CASE("Tests the Neon FSUB (scalar) instruction generation", "[Neon_FSUB_SCALAR]")
{
    uint32_t    l_ins = simd_fp::fsubScalar(simd_fp_t::v0, simd_fp_t::v1, simd_fp_t::v2, neon_size_spec_t::s);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x1e223820");

    l_ins = simd_fp::fsubScalar(simd_fp_t::v2, simd_fp_t::v3, simd_fp_t::v4, neon_size_spec_t::d);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x1e643862");
}

TEST_CASE("Tests the Neon FSUB (vector) instruction generation", "[Neon_FSUB_VEC]")
{
    uint32_t    l_ins = simd_fp::fsubVec(simd_fp_t::v0, simd_fp_t::v1, simd_fp_t::v2, arr_spec_t::s2);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x0ea2d420");

    l_ins = simd_fp::fsubVec(simd_fp_t::v4, simd_fp_t::v7, simd_fp_t::v6, arr_spec_t::s4);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4ea6d4e4");

    l_ins = simd_fp::fsubVec(simd_fp_t::v9, simd_fp_t::v6, simd_fp_t::v8, arr_spec_t::d2);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4ee8d4c9");
}

TEST_CASE("Tests the Neon INS (element) instruction generation", "[Neon_INS_SCALAR]")
{
    uint32_t    l_ins = simd_fp::ins(simd_fp_t::v0, simd_fp_t::v1, 0, 1, neon_size_spec_t::s);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x6e042420");

    l_ins = simd_fp::ins(simd_fp_t::v3, simd_fp_t::v2, 1, 0, neon_size_spec_t::s);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x6e0c0443");

    l_ins = simd_fp::ins(simd_fp_t::v3, simd_fp_t::v2, 2, 3, neon_size_spec_t::s);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x6e146443");

    l_ins = simd_fp::ins(simd_fp_t::v4, simd_fp_t::v1, 1, 0, neon_size_spec_t::d);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x6e180424");

    l_ins = simd_fp::ins(simd_fp_t::v4, simd_fp_t::v1, 0, 1, neon_size_spec_t::d);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x6e084424");
}

TEST_CASE("Tests the Neon LD1 (single structure) with a lane index instruction generation", "[Neon LD1 Single Structure Index]")
{
    uint32_t    l_ins = simd_fp::ld1(simd_fp_t::v0, gpr_t::x0, 3, neon_size_spec_t::s);
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
    uint32_t    l_ins = simd_fp::ld1(simd_fp_t::v0, gpr_t::x0, 3, neon_size_spec_t::s, gpr_t::x1);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4dc19000");
}

TEST_CASE("Tests the Neon LD1 (single structure) with a lane index and a post-index immediate instruction generation", "[Neon LD1 Single Structure Index Post-Index Immediate]")
{
    uint32_t    l_ins = simd_fp::ld1(simd_fp_t::v0, gpr_t::x0, 3, neon_size_spec_t::s, 4);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4ddf9000");

    l_ins = simd_fp::ld1(simd_fp_t::v0, gpr_t::x0, 1, neon_size_spec_t::d, 8);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4ddf8400");

    CHECK_THROWS_AS(simd_fp::ld1(simd_fp_t::v0, gpr_t::x1, 1, neon_size_spec_t::d, 4), std::invalid_argument);
    CHECK_THROWS_AS(simd_fp::ld1(simd_fp_t::v0, gpr_t::x1, 1, neon_size_spec_t::s, 8), std::invalid_argument);
}

TEST_CASE("Tests the Neon ST1 (single structure) with a lane index instruction generation", "[Neon ST1 Single Structure Index]")
{
    uint32_t    l_ins = simd_fp::st1(simd_fp_t::v0, gpr_t::x0, 3, neon_size_spec_t::s);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4d009000");

    l_ins = simd_fp::st1(simd_fp_t::v5, gpr_t::x1, 1, neon_size_spec_t::d);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4d008425");

    CHECK_THROWS_AS(simd_fp::st1(simd_fp_t::v0, gpr_t::x0, 4, neon_size_spec_t::s), std::out_of_range);
    CHECK_THROWS_AS(simd_fp::st1(simd_fp_t::v0, gpr_t::x0, 2, neon_size_spec_t::d), std::out_of_range);
}

TEST_CASE("Tests the Neon ST1 (single structure) with a lane index and a register post-index instruction generation", "[Neon ST1 Single Structure Index Post-Index Register]")
{
    uint32_t    l_ins = simd_fp::st1(simd_fp_t::v0, gpr_t::x0, 3, neon_size_spec_t::s, gpr_t::x1);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4d819000");
}

TEST_CASE("Tests the Neon ST1 (single structure) with a lane index and a post-index immediate instruction generation", "[Neon ST1 Single Structure Index Post-Index Immediate]")
{
    uint32_t    l_ins = simd_fp::st1(simd_fp_t::v0, gpr_t::x0, 3, neon_size_spec_t::s, 4);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4d9f9000");

    l_ins = simd_fp::st1(simd_fp_t::v0, gpr_t::x0, 1, neon_size_spec_t::d, 8);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4d9f8400");

    CHECK_THROWS_AS(simd_fp::st1(simd_fp_t::v0, gpr_t::x1, 1, neon_size_spec_t::d, 4), std::invalid_argument);
    CHECK_THROWS_AS(simd_fp::st1(simd_fp_t::v0, gpr_t::x1, 1, neon_size_spec_t::s, 8), std::invalid_argument);
}

TEST_CASE("Tests the Neon MOV (from general-purpose register) instruction generation", "[Neon MOV GPR]")
{
    uint32_t    l_ins = simd_fp::mov(simd_fp_t::v0, gpr_t::wzr, 3, neon_size_spec_t::s);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4e1c1fe0");

    l_ins = simd_fp::mov(simd_fp_t::v2, gpr_t::wzr, 3, neon_size_spec_t::s);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4e1c1fe2");

    l_ins = simd_fp::mov(simd_fp_t::v0, gpr_t::w1, 3, neon_size_spec_t::s);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4e1c1c20");

    l_ins = simd_fp::mov(simd_fp_t::v8, gpr_t::x2, 0, neon_size_spec_t::d);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4e081c48");

    CHECK_THROWS_AS(simd_fp::mov(simd_fp_t::v0, gpr_t::wzr, 3, neon_size_spec_t::q), std::invalid_argument);
}

TEST_CASE("Tests the Neon FMADD instruction generation", "[Neon FMADD]")
{
    uint32_t    l_ins = simd_fp::fmadd(simd_fp_t::v1, simd_fp_t::v25, simd_fp_t::v29, simd_fp_t::v1, neon_size_spec_t::s);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x1f1d0721");

    l_ins = simd_fp::fmadd(simd_fp_t::v1, simd_fp_t::v25, simd_fp_t::v29, simd_fp_t::v1, neon_size_spec_t::d);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x1f5d0721");
}

TEST_CASE("Tests the Neon FMAX instruction generation", "[Neon FMAX]")
{
    // vector
    uint32_t    l_ins = simd_fp::fmaxVec(simd_fp_t::v3, simd_fp_t::v0, simd_fp_t::v1, arr_spec_t::s4);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4e21f403");

    l_ins = simd_fp::fmaxVec(simd_fp_t::v24, simd_fp_t::v31, simd_fp_t::v13, arr_spec_t::s2);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x0e2df7f8");

    CHECK_THROWS_AS(simd_fp::fmaxVec(simd_fp_t::v24, simd_fp_t::v31, simd_fp_t::v13, arr_spec_t::b8), std::invalid_argument);
    CHECK_THROWS_AS(simd_fp::fmaxVec(simd_fp_t::v24, simd_fp_t::v31, simd_fp_t::v13, arr_spec_t::b16), std::invalid_argument);

    // scalar
    l_ins = simd_fp::fmaxScalar(simd_fp_t::v3, simd_fp_t::v0, simd_fp_t::v1, neon_size_spec_t::s);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x1e214803");

    l_ins = simd_fp::fmaxScalar(simd_fp_t::v24, simd_fp_t::v31, simd_fp_t::v13, neon_size_spec_t::d);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x1e6d4bf8");

    CHECK_THROWS_AS(simd_fp::fmaxScalar(simd_fp_t::v24, simd_fp_t::v31, simd_fp_t::v13, neon_size_spec_t::q), std::invalid_argument);
}

TEST_CASE("Tests the Neon FMIN instruction generation", "[Neon FMIN]")
{
    // vector
    uint32_t    l_ins = simd_fp::fminVec(simd_fp_t::v3, simd_fp_t::v0, simd_fp_t::v1, arr_spec_t::s4);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4ea1f403");

    l_ins = simd_fp::fminVec(simd_fp_t::v24, simd_fp_t::v31, simd_fp_t::v13, arr_spec_t::s2);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x0eadf7f8");

    CHECK_THROWS_AS(simd_fp::fminVec(simd_fp_t::v24, simd_fp_t::v31, simd_fp_t::v13, arr_spec_t::b8), std::invalid_argument);
    CHECK_THROWS_AS(simd_fp::fminVec(simd_fp_t::v24, simd_fp_t::v31, simd_fp_t::v13, arr_spec_t::b16), std::invalid_argument);

    // scalar
    l_ins = simd_fp::fminScalar(simd_fp_t::v3, simd_fp_t::v0, simd_fp_t::v1, neon_size_spec_t::s);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x1e215803");

    l_ins = simd_fp::fminScalar(simd_fp_t::v24, simd_fp_t::v31, simd_fp_t::v13, neon_size_spec_t::d);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x1e6d5bf8");

    CHECK_THROWS_AS(simd_fp::fminScalar(simd_fp_t::v24, simd_fp_t::v31, simd_fp_t::v13, neon_size_spec_t::q), std::invalid_argument);
}

TEST_CASE("Tests the Neon EOR instruction generation", "[Neon EOR]")
{
    uint32_t    l_ins = simd_fp::eor(simd_fp_t::v2, simd_fp_t::v0, simd_fp_t::v1, arr_spec_t::b8);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x2e211c02");

    l_ins = simd_fp::eor(simd_fp_t::v2, simd_fp_t::v0, simd_fp_t::v1, arr_spec_t::b16);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x6e211c02");

    CHECK_THROWS_AS(simd_fp::eor(simd_fp_t::v2, simd_fp_t::v0, simd_fp_t::v1, arr_spec_t::s2), std::invalid_argument);
    CHECK_THROWS_AS(simd_fp::eor(simd_fp_t::v2, simd_fp_t::v0, simd_fp_t::v1, arr_spec_t::s4), std::invalid_argument);
    CHECK_THROWS_AS(simd_fp::eor(simd_fp_t::v2, simd_fp_t::v0, simd_fp_t::v1, arr_spec_t::d2), std::invalid_argument);
}

TEST_CASE("Tests the TRN1 and TRN2 instruction generation", "[Neon TRN]")
{
    uint32_t    l_ins = simd_fp::trn1(simd_fp_t::v17, simd_fp_t::v11, simd_fp_t::v29, arr_spec_t::s2);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x0e9d2971");

    l_ins = simd_fp::trn2(simd_fp_t::v12, simd_fp_t::v7, simd_fp_t::v0, arr_spec_t::d2);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4ec068ec");
}

TEST_CASE("Tests the ZIP1 and ZIP2 instruction generation", "[Neon ZIP]")
{
    uint32_t    l_ins = simd_fp::zip1(simd_fp_t::v4, simd_fp_t::v9, simd_fp_t::v23, arr_spec_t::s4);
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4e973924");

    l_ins = simd_fp::zip2(simd_fp_t::v6, simd_fp_t::v4, simd_fp_t::v11, arr_spec_t::d2);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4ecb7886");

    l_ins = simd_fp::zip2(simd_fp_t::v11, simd_fp_t::v6, simd_fp_t::v7, arr_spec_t::s4);
    l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x4e8778cb");
}
