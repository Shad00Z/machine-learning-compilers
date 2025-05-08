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
	SUBMISSIONS = $(shell find $(SUB_DIR) -type f)
else ifeq ($(OS),linux)
	SRC = $(shell find src -name "*.cpp")
	TEST_SRC = $(shell find src -name "*.test.cpp")
	SUBMISSIONS = $(shell find $(SUB_DIR) -type f)
else ifeq ($(OS),windows)
	find_files = $(foreach n,$1,$(shell C:\\\msys64\\\usr\\\bin\\\find.exe -L $2 -name "$n"))
	SRC = $(call find_files,*.cpp,src)
	TEST_SRC = $(call find_files,*.test.cpp,src)
	SUBMISSIONS = $(call find_files,*,src/submissions)
endif

# MAIN FILES FOR ENTRY POINTS
INSTGEN_EXAMPLES_MAIN_SRC = $(SRC_DIR)/instgen_examples.cpp
KERNEL_EXAMPLES_MAIN_SRC = $(SRC_DIR)/kernel_examples.cpp
TESTS_MAIN_SRC = $(SRC_DIR)/tests.cpp

# COMMON SOURCES (EXCEPT MAIN FILES)
COMMON_SRC = $(filter-out $(INSTGEN_EXAMPLES_MAIN_SRC) $(KERNEL_EXAMPLES_MAIN_SRC) $(TESTS_MAIN_SRC) $(SUBMISSIONS) $(TEST_SRC), $(SRC))
NOSUB_TEST_SRC = $(filter-out $(SUBMISSIONS), $(TEST_SRC))

# DEP
COMMON_DEP = $(COMMON_SRC:%.cpp=$(BIN_DIR)/%.d)
INSTGEN_EXAMPLES_MAIN_DEP = $(INSTGEN_EXAMPLES_MAIN_SRC:%.cpp=$(BIN_DIR)/%.d)
KERNEL_EXAMPLES_MAIN_DEP = $(KERNEL_EXAMPLES_MAIN_SRC:%.cpp=$(BIN_DIR)/%.d)
TESTS_MAIN_DEP = $(TESTS_MAIN_SRC:%.cpp=$(BIN_DIR)/%.d)
-include $(COMMON_DEP)
-include $(INSTGEN_EXAMPLES_MAIN_DEP)
-include $(KERNEL_EXAMPLES_MAIN_DEP)
-include $(TESTS_MAIN_DEP)

# Convert sources to object files
COMMON_OBJ = $(COMMON_SRC:%.cpp=$(BIN_DIR)/%.o)
INSTGEN_OBJ = $(INSTGEN_EXAMPLES_MAIN_SRC:%.cpp=$(BIN_DIR)/%.o)
KERNEL_OBJ = $(KERNEL_EXAMPLES_MAIN_SRC:%.cpp=$(BIN_DIR)/%.o)
TESTS_OBJ = $(TESTS_MAIN_SRC:%.cpp=$(BIN_DIR)/%.o)
NOSUB_TEST_OBJ = $(NOSUB_TEST_SRC:%.cpp=$(BIN_DIR)/%.o)

# TARGETS
default: tests instgen_examples kernel_examples

$(BIN_DIR):
	mkdir -p $@

createdirs: $(BIN_DIR)
	$(COPY_DIRS_CMD)

$(BIN_DIR)/%.o: %.cpp
	$(CXX) -o $@ -MMD -c $< $(CXXFLAGS) $(INCFLAGS)

tests: createdirs $(COMMON_OBJ) $(TESTS_OBJ) $(NOSUB_TEST_OBJ)
	$(LD) -o $(BIN_DIR)/tests $(COMMON_OBJ) $(TESTS_OBJ) $(NOSUB_TEST_OBJ) $(LDFLAGS) $(LIBS)

instgen_examples: createdirs $(COMMON_OBJ) $(INSTGEN_OBJ)
	$(LD) -o $(BIN_DIR)/instgen_examples $(COMMON_OBJ) $(INSTGEN_OBJ) $(LDFLAGS) $(LIBS)

kernel_examples: createdirs $(COMMON_OBJ) $(KERNEL_OBJ)
	$(LD) -o $(BIN_DIR)/kernel_examples $(COMMON_OBJ) $(KERNEL_OBJ) $(LDFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -rf $(BIN_DIR)/$(SRC_DIR)
