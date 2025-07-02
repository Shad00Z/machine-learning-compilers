######################################################################
# Makefile for Machine Learning Compilers
# Author: lgrumbach, lobitz
######################################################################

# TOOLS
CXX = g++
LD = g++
CPP_VERSION = c++20

# LIBS
LIBS = 

# DIRECTORIES
SRC_DIR = src
TEST_DIR = tests
UNIT_TESTS_DIR = $(TEST_DIR)/unit
INT_TESTS_DIR = $(TEST_DIR)/integration
BIN_DIR_ROOT = build
KERNELS_DIR = $(BIN_DIR_ROOT)/kernels
LIB_DIR = 
INC_DIR = include
SUB_DIR = $(SRC_DIR)/submissions

# COMPILER FLAGS
CXXFLAGS  = -std=$(CPP_VERSION)
CXXFLAGS += -O2
CXXFLAGS += -g
CXXFLAGS += -Wall
CXXFLAGS += -Wextra
CXXFLAGS += -Wpedantic
# Suppress warnings from Catch2 and standard library regex
CXXFLAGS += -Wno-maybe-uninitialized

# LINKER FLAGS
LDFLAGS = 

# SANITIZER FLAGS
SAN_CXX_FLAGS  = -g
SAN_CXX_FLAGS += -fsanitize=float-divide-by-zero
SAN_CXX_FLAGS += -fsanitize=bounds
SAN_CXX_FLAGS += -fsanitize=address
SAN_CXX_FLAGS += -fsanitize=undefined
SAN_CXX_FLAGS += -fno-omit-frame-pointer

SAN_LD_FLAGS  = -g
SAN_LD_FLAGS += -fsanitize=address
SAN_LD_FLAGS += -fsanitize=undefined

# TARGET OS
ifeq ($(OS),Windows_NT)
	OS = windows
else
	UNAME := $(shell uname -s)
	ifneq (,$(findstring _NT,$(UNAME)))
		OS = windows
	else ifeq ($(UNAME),Darwin)
		OS = macOS
	else ifeq ($(UNAME),Linux)
		OS = linux
	else
    	$(error OS not supported by this Makefile)
	endif
endif
ARCH := $(shell uname -m)

# Check which compiler is used on macos
ifeq ($(OS),macOS)
ifneq (,$(findstring Apple clang,$(shell $(CXX) --version)))
$(warning Apple clang does not support OpenMP. Switching to LLVM)

LLVM_LOCATION = $(shell brew --prefix llvm)
CXX = $(LLVM_LOCATION)/bin/clang++
LD = $(LLVM_LOCATION)/bin/clang++
endif
endif

# OS-SPECIFIC DIRECTORIES
BIN_DIR := $(BIN_DIR_ROOT)/$(OS)
ifeq ($(OS),windows)
	# Windows 32-bit
	ifeq ($(win32),1)
		BIN_DIR := $(BIN_DIR)32
	# Windows 64-bit
	else
		BIN_DIR := $(BIN_DIR)64
	endif
else ifeq ($(OS),macOS)
	BIN_DIR := $(BIN_DIR)-$(ARCH)
endif

# Threads
LDFLAGS += -lpthread
ifneq ($(OS),macOS)
	CXXFLAGS += -pthread
endif

# OPENMP
ifeq ($(OS),macOS)
	CXXFLAGS += -Xpreprocessor
	LDFLAGS += -Xpreprocessor

	LIBOMP_LOCATION = $(shell brew --prefix libomp)
	CXXFLAGS += -I$(LIBOMP_LOCATION)/include
	LDFLAGS += -L$(LIBOMP_LOCATION)/lib
	LDFLAGS += -lomp
endif

CXXFLAGS += -fopenmp
LDFLAGS += -fopenmp

# INCLUDES
# INCFLAGS = -I$(LIB_DIR)
INCFLAGS = -I$(INC_DIR)
INCFLAGS += -I$(TEST_DIR)
INCFLAGS += -I/usr/local/include
ifeq ($(OS),windows)
	INCFLAGS += $(shell C:\\msys64\\usr\\bin\\find.exe -L $(SRC_DIR) -type d -path "$(SUB_DIR)" -prune -o -type d -exec echo -I{} \;)
else ifeq ($(OS),macOS)
	INCFLAGS += $(shell find $(SRC_DIR) -path "$(SUB_DIR)" -prune -o -type d -exec echo -I{} \;)
	ifeq ($(ARCH),arm64)
		INCFLAGS += -I/opt/homebrew/include
	else ifeq ($(ARCH),x86_64)
		INCFLAGS += -I/usr/local/include
	endif
else ifeq ($(OS),linux)
	INCFLAGS += $(shell find $(SRC_DIR) -path "$(SUB_DIR)" -prune -o -type d -exec echo -I{} \;)
endif

# LINKER LIBRARIES
ifeq ($(OS),macOS)
	ifeq ($(ARCH),arm64)
		LDFLAGS += -L/opt/homebrew/lib
	else ifeq ($(ARCH),x86_64)
		LDFLAGS += -L/usr/local/lib
	endif
endif

# GATHER ALL SOURCES
ifeq ($(OS),macOS)
	SRC = $(shell find $(SRC_DIR) -name "*.cpp" ! -name "*.bench.cpp")
	UNIT_TEST_SRC = $(shell find $(UNIT_TESTS_DIR) -name "*.cpp")
#	INT_TEST_SRC = $(shell find $(INT_TESTS_DIR) -name "*.cpp")
	BENCH_SRC = $(shell find $(SRC_DIR) -name "*.bench.cpp")
	SUBMISSIONS = $(shell find $(SUB_DIR) -type f)
else ifeq ($(OS),linux)
	SRC = $(shell find $(SRC_DIR) -name "*.cpp" ! -name "*.bench.cpp")
	UNIT_TEST_SRC = $(shell find $(UNIT_TESTS_DIR) -name "*.cpp")
#	INT_TEST_SRC = $(shell find $(INT_TESTS_DIR) -name "*.cpp")
	BENCH_SRC = $(shell find $(SRC_DIR) -name "*.bench.cpp")
	SUBMISSIONS = $(shell find $(SUB_DIR) -type f)
else ifeq ($(OS),windows)
	find_files = $(foreach n,$1,$(shell C:\\\msys64\\\usr\\\bin\\\find.exe -L $2 -name "$n"))
	SRC = $(call find_files,*.cpp,$(SRC_DIR))
	UNIT_TEST_SRC = $(call find_files,*.cpp,$(UNIT_TESTS_DIR))
#	INT_TEST_SRC = $(call find_files,*.cpp,$(INT_TESTS_DIR))
	BENCH_SRC = $(call find_files,*.bench.cpp,$(SRC_DIR))
	SUBMISSIONS = $(call find_files,*,$(SUB_DIR))
endif

# MAIN FILES FOR ENTRY POINTS
TESTS_MAIN_SRC = $(TEST_DIR)/tests.cpp
BENCH_MAIN_SRC = $(SRC_DIR)/benchmarks.cpp

# COMMON SOURCES (EXCEPT MAIN FILES)
COMMON_SRC = $(filter-out $(SUBMISSIONS) $(BENCH_MAIN_SRC) $(BENCH_SRC), $(SRC))

# DEP
COMMON_DEP = $(COMMON_SRC:%.cpp=$(BIN_DIR)/%.d)
TESTS_MAIN_DEP = $(TESTS_MAIN_SRC:%.cpp=$(BIN_DIR)/%.d)
UNIT_TEST_DEP = $(UNIT_TEST_SRC:%.cpp=$(BIN_DIR)/%.d)
INT_TEST_DEP = $(INT_TEST_SRC:%.cpp=$(BIN_DIR)/%.d)
BENCH_MAIN_DEP = $(BENCH_MAIN_SRC:%.cpp=$(BIN_DIR)/%.d)
BENCH_DEP = $(BENCH_SRC:%.cpp=$(BIN_DIR)/%.d)
-include $(COMMON_DEP)
-include $(TESTS_MAIN_DEP)
-include $(UNIT_TEST_DEP)
-include $(INT_TEST_DEP)
-include $(BENCH_MAIN_DEP)
-include $(BENCH_DEP)

# Convert sources to object files
COMMON_OBJ = $(COMMON_SRC:%.cpp=$(BIN_DIR)/%.o)
TESTS_MAIN_OBJ = $(TESTS_MAIN_SRC:%.cpp=$(BIN_DIR)/%.o)
UNIT_TEST_OBJ = $(UNIT_TEST_SRC:%.cpp=$(BIN_DIR)/%.o)
INT_TEST_OBJ = $(INT_TEST_SRC:%.cpp=$(BIN_DIR)/%.o)
BENCH_MAIN_OBJ = $(BENCH_MAIN_SRC:%.cpp=$(BIN_DIR)/%.o)
BENCH_OBJ = $(BENCH_SRC:%.cpp=$(BIN_DIR)/%.o)

# TARGETS
default: tests benchmarks

$(BIN_DIR):
	mkdir -p $@

createdirs: $(BIN_DIR)

$(BIN_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) -o $@ -MMD -c $< $(CXXFLAGS) $(INCFLAGS)

benchmarks: createdirs $(COMMON_OBJ) $(BENCH_MAIN_OBJ) $(BENCH_OBJ)
	$(LD) -o $(BIN_DIR)/benchmarks $(COMMON_OBJ) $(BENCH_MAIN_OBJ) $(BENCH_OBJ) $(LDFLAGS) $(LIBS)

unit-tests: createdirs $(COMMON_OBJ) $(TESTS_MAIN_OBJ) $(UNIT_TEST_OBJ)
	$(LD) -o $(BIN_DIR)/$(TEST_DIR)/unit-tests $(COMMON_OBJ) $(TESTS_MAIN_OBJ) $(UNIT_TEST_OBJ) $(LDFLAGS) $(LIBS)

int-tests: createdirs $(COMMON_OBJ) $(TESTS_MAIN_OBJ) $(INT_TEST_OBJ)
	$(LD) -o $(BIN_DIR)/$(TEST_DIR)/int-tests $(COMMON_OBJ) $(TESTS_MAIN_OBJ) $(INT_TEST_OBJ) $(LDFLAGS) $(LIBS)

tests: unit-tests int-tests

unit-tests-san: CXXFLAGS := $(SAN_CXX_FLAGS) $(CXXFLAGS)
unit-tests-san: LDFLAGS := $(SAN_LD_FLAGS) $(LDFLAGS)
unit-tests-san: createdirs $(COMMON_OBJ) $(TESTS_MAIN_OBJ) $(UNIT_TEST_OBJ)
	$(LD) -o $(BIN_DIR)/$(TEST_DIR)/unit-tests-san $(COMMON_OBJ) $(TESTS_MAIN_OBJ) $(UNIT_TEST_OBJ) $(LDFLAGS) $(LIBS)

int-tests-san: CXXFLAGS := $(SAN_CXX_FLAGS) $(CXXFLAGS)
int-tests-san: LDFLAGS := $(SAN_LD_FLAGS) $(LDFLAGS)
int-tests-san: createdirs $(COMMON_OBJ) $(TESTS_MAIN_OBJ) $(INT_TEST_OBJ)
	$(LD) -o $(BIN_DIR)/$(TEST_DIR)/int-tests-san $(COMMON_OBJ) $(TESTS_MAIN_OBJ) $(INT_TEST_OBJ) $(LDFLAGS) $(LIBS)

tests-san: unit-tests-san int-tests-san

.PHONY: clean

clean:
	rm -rf $(BIN_DIR)
	rm -rf $(KERNELS_DIR)
