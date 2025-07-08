#include <catch2/catch.hpp>
#include <cstdint>
#include <mlc/converters/instToBinString.h>
#include <mlc/converters/instToHexString.h>
#include <string>

using namespace mini_jit::converters;

TEST_CASE("Tests the hex string generation", "[InstToHexString]")
{
    uint32_t    l_ins = 0x12345678;
    std::string l_hex = to_string_hex(l_ins);
    REQUIRE(l_hex == "0x12345678");
}

TEST_CASE("Tests the binary string generation", "[InstToBinaryString]")
{
    uint32_t    l_ins = 0x12345678;
    std::string l_bin = to_string_bin(l_ins);
    REQUIRE(l_bin == "0b00010010001101000101011001111000");
}