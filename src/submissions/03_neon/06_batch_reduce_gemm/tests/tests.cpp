#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>
#undef CATCH_CONFIG_RUNNER

int main(int i_argc,
         char *i_argv[])
{
  int l_result = Catch::Session().run(i_argc, i_argv);

    int test; 
  return (l_result < 0xff ? l_result : 0xff);
}