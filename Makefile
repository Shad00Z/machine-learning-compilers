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
BIN_DIR_ROOT = build
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
INCFLAGS += -I$(SRC_DIR)
INCFLAGS += $(shell find $(SRC_DIR) -type d -exec echo -I{} \;)
INCFLAGS += -I$(TEST_DIR)
INCFLAGS += -I/usr/local/include
ifeq ($(ARCH),arm64)
	INCFLAGS += -I/opt/homebrew/include
endif

# LINKER LIBRARIES
ifeq ($(OS),macOS)
	ifeq ($(ARCH),arm64)
		LDFLAGS += -L/opt/homebrew/lib
	else ifeq ($(ARCH),x86_64)
		LDFLAGS += -L/usr/local/lib
	endif
endif

# DIRECTORY COPY COMMAND
ifeq ($(OS),windows)
	COPY_DIRS_CMD = cmd /c 'robocopy $(SRC_DIR) $(BIN_DIR)/$(SRC_DIR) /e /xd submissions /xf * /mt /NFL /NDL /NJH /NJS /nc /ns /np & exit 0'
else ifeq ($(OS),macOS)
	COPY_DIRS_CMD = rsync -a --exclude 'submissions/' --include '*/' --exclude '*' "$(SRC_DIR)" "$(BIN_DIR)" && mkdir -p $(BIN_DIR)/$(TEST_DIR)/unit $(BIN_DIR)/$(TEST_DIR)/integration
else ifeq ($(OS),linux)
	COPY_DIRS_CMD = rsync -a --exclude 'submissions/' --include '*/' --exclude '*' "$(SRC_DIR)" "$(BIN_DIR)" && mkdir -p $(BIN_DIR)/$(TEST_DIR)/unit $(BIN_DIR)/$(TEST_DIR)/integration
endif

# GATHER ALL SOURCES
ifeq ($(OS),macOS)
	SRC = $(shell find src -name "*.cpp" ! -name "*.test.cpp" ! -name "*.bench.cpp")
	TESTS_MAIN_SRC = tests/unit/tests.cpp
	TEST_SRC = $(shell find tests/unit -name "*.cpp" ! -name "tests.cpp")
	BENCH_SRC = $(shell find src -name "*.bench.cpp")
	SUBMISSIONS = $(shell find $(SUB_DIR) -type f)
else ifeq ($(OS),linux)
	SRC = $(shell find src -name "*.cpp" ! -name "*.test.cpp" ! -name "*.bench.cpp")
	TESTS_MAIN_SRC = tests/unit/tests.cpp
	TEST_SRC = $(shell find tests/unit -name "*.cpp" ! -name "tests.cpp")
	BENCH_SRC = $(shell find src -name "*.bench.cpp")
	SUBMISSIONS = $(shell find $(SUB_DIR) -type f)
else ifeq ($(OS),windows)
	find_files = $(foreach n,$1,$(shell C:\\\msys64\\\usr\\\bin\\\find.exe -L $2 -name "$n"))
	SRC = $(call find_files,*.cpp,src)
	TESTS_MAIN_SRC = tests/unit/tests.cpp
	TEST_SRC = $(filter-out tests/unit/tests.cpp,$(call find_files,*.cpp,tests/unit))
	BENCH_SRC = $(call find_files,*.bench.cpp,src)
	SUBMISSIONS = $(call find_files,*,src/submissions)
endif

# MAIN FILES FOR ENTRY POINTS
TESTS_MAIN_SRC = $(TEST_DIR)/unit/tests.cpp
TEST_SRC = $(shell find tests/unit -name "*.cpp" ! -name "tests.cpp")
BENCH_MAIN_SRC = $(SRC_DIR)/benchmarks.cpp

# COMMON SOURCES (EXCEPT MAIN FILES)
COMMON_SRC = $(filter-out $(TESTS_MAIN_SRC) $(SUBMISSIONS) $(TEST_SRC) $(BENCH_MAIN_SRC) $(BENCH_SRC), $(SRC))
NOSUB_TEST_SRC = $(filter-out $(SUBMISSIONS), $(TEST_SRC))

# DEP
COMMON_DEP = $(COMMON_SRC:%.cpp=$(BIN_DIR)/%.d)
TESTS_MAIN_DEP = $(TESTS_MAIN_SRC:%.cpp=$(BIN_DIR)/%.d)
BENCH_MAIN_DEP = $(BENCH_MAIN_SRC:%.cpp=$(BIN_DIR)/%.d)
NOSUB_TEST_DEP = $(NOSUB_TEST_SRC:%.cpp=$(BIN_DIR)/%.d)
BENCH_DEP = $(BENCH_SRC:%.cpp=$(BIN_DIR)/%.d)
-include $(COMMON_DEP)
-include $(TESTS_MAIN_DEP)
-include $(BENCH_MAIN_DEP)
-include $(NOSUB_TEST_DEP)
-include $(BENCH_DEP)

# Convert sources to object files
COMMON_OBJ = $(COMMON_SRC:%.cpp=$(BIN_DIR)/%.o)
TESTS_MAIN_OBJ = $(TESTS_MAIN_SRC:%.cpp=$(BIN_DIR)/%.o)
TEST_OBJ = $(TEST_SRC:%.cpp=$(BIN_DIR)/%.o)
BENCH_MAIN_OBJ = $(BENCH_MAIN_SRC:%.cpp=$(BIN_DIR)/%.o)
BENCH_OBJ = $(BENCH_SRC:%.cpp=$(BIN_DIR)/%.o)

# TARGETS
default: tests benchmarks

$(BIN_DIR):
	mkdir -p $@

createdirs: $(BIN_DIR)
	$(COPY_DIRS_CMD)

$(BIN_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) -o $@ -MMD -c $< $(CXXFLAGS) $(INCFLAGS)

# Rule for test files to prevent duplicate symbols
$(BIN_DIR)/$(TEST_DIR)/%.o: $(TEST_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) -o $@ -MMD -c $< $(CXXFLAGS) $(INCFLAGS)

tests: createdirs $(COMMON_OBJ) $(TESTS_MAIN_OBJ) $(TEST_OBJ)
	@mkdir -p $(dir $(BIN_DIR)/tests/unit/tests)
	$(LD) -o $(BIN_DIR)/tests/unit/tests $(COMMON_OBJ) $(TESTS_MAIN_OBJ) $(TEST_OBJ) $(LDFLAGS) $(LIBS)

benchmarks: createdirs $(COMMON_OBJ) $(BENCH_MAIN_OBJ) $(BENCH_OBJ)
	$(LD) -o $(BIN_DIR)/benchmarks $(COMMON_OBJ) $(BENCH_MAIN_OBJ) $(BENCH_OBJ) $(LDFLAGS) $(LIBS)

san: CXXFLAGS := $(SAN_CXX_FLAGS) $(CXXFLAGS)
san: LDFLAGS := $(SAN_LD_FLAGS) $(LDFLAGS)

san: createdirs $(COMMON_OBJ) $(TESTS_MAIN_OBJ) $(TEST_OBJ)
	@mkdir -p $(dir $(BIN_DIR)/tests_san)
	$(LD) -o $(BIN_DIR)/tests_san $(COMMON_OBJ) $(TESTS_MAIN_OBJ) $(TEST_OBJ) $(LDFLAGS) $(LIBS)

.PHONY: clean

run_tests: tests
	$(BIN_DIR)/tests
	rm -rf *.bin

clean:
	rm -rf $(BIN_DIR)
	rm -rf *.bin
