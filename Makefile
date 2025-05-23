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
BIN_DIR_ROOT = build
LIB_DIR = 
INC_DIR = include
SUB_DIR = $(SRC_DIR)/submissions

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

# INCLUDES
# INCFLAGS = -I$(LIB_DIR)
INCFLAGS = -I$(INC_DIR)
INCFLAGS += -I$(SRC_DIR)
INCFLAGS += -I/usr/local/include
ifeq ($(ARCH),arm64)
	INCFLAGS += -I/opt/homebrew/include
endif

# COMPILER FLAGS
CXXFLAGS  = -std=$(CPP_VERSION)
CXXFLAGS += -O2
CXXFLAGS += -g
CXXFLAGS += -Wall
CXXFLAGS += -Wextra
CXXFLAGS += -Wpedantic

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
	COPY_DIRS_CMD = rsync -a --exclude 'submissions/' --include '*/' --exclude '*' "$(SRC_DIR)" "$(BIN_DIR)"
else ifeq ($(OS),linux)
	COPY_DIRS_CMD = rsync -a --exclude 'submissions/' --include '*/' --exclude '*' "$(SRC_DIR)" "$(BIN_DIR)"
endif

# GATHER ALL SOURCES
ifeq ($(OS),macOS)
	SRC = $(shell find src -name "*.cpp")
	TEST_SRC = $(shell find src -name "*.test.cpp")
	BENCH_SRC = $(shell find src -name "*.bench.cpp")
	SUBMISSIONS = $(shell find $(SUB_DIR) -type f)
else ifeq ($(OS),linux)
	SRC = $(shell find src -name "*.cpp")
	TEST_SRC = $(shell find src -name "*.test.cpp")
	BENCH_SRC = $(shell find src -name "*.bench.cpp")
	SUBMISSIONS = $(shell find $(SUB_DIR) -type f)
else ifeq ($(OS),windows)
	find_files = $(foreach n,$1,$(shell C:\\\msys64\\\usr\\\bin\\\find.exe -L $2 -name "$n"))
	SRC = $(call find_files,*.cpp,src)
	TEST_SRC = $(call find_files,*.test.cpp,src)
	BENCH_SRC = $(call find_files,*.bench.cpp,src)
	SUBMISSIONS = $(call find_files,*,src/submissions)
endif

# MAIN FILES FOR ENTRY POINTS
TESTS_MAIN_SRC = $(SRC_DIR)/tests.cpp
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
TESTS_OBJ = $(TESTS_MAIN_SRC:%.cpp=$(BIN_DIR)/%.o)
NOSUB_TEST_OBJ = $(NOSUB_TEST_SRC:%.cpp=$(BIN_DIR)/%.o)
BENCH_MAIN_OBJ = $(BENCH_MAIN_SRC:%.cpp=$(BIN_DIR)/%.o)
BENCH_OBJ = $(BENCH_SRC:%.cpp=$(BIN_DIR)/%.o)

# TARGETS
default: tests benchmarks

$(BIN_DIR):
	mkdir -p $@

createdirs: $(BIN_DIR)
	$(COPY_DIRS_CMD)

$(BIN_DIR)/%.o: %.cpp
	$(CXX) -o $@ -MMD -c $< $(CXXFLAGS) $(INCFLAGS)

tests: createdirs $(COMMON_OBJ) $(TESTS_OBJ) $(NOSUB_TEST_OBJ)
	$(LD) -o $(BIN_DIR)/tests $(COMMON_OBJ) $(TESTS_OBJ) $(NOSUB_TEST_OBJ) $(LDFLAGS) $(LIBS)

benchmarks: createdirs $(COMMON_OBJ) $(BENCH_MAIN_OBJ) $(BENCH_OBJ)
	$(LD) -o $(BIN_DIR)/benchmarks $(COMMON_OBJ) $(BENCH_MAIN_OBJ) $(BENCH_OBJ) $(LDFLAGS) $(LIBS)

.PHONY: clean

run_tests: tests
	$(BIN_DIR)/tests
	rm -rf *.bin

clean:
	rm -rf $(BIN_DIR)
	rm -rf *.bin
