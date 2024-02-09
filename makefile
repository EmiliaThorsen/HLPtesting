
# Much of this was taken from makefiletutorial.com

TARGET = main

# DEBUG = pona
# WERROR = tawa
WALL = tonsi

SRC_DIR := src
BUILD_DIR := build

ifdef DEBUG
	BUILD_DIR = debug
$(info $(shell echo DEBUG))
endif

AS := nasm
CC := gcc
LD := gcc

SRCS := $(shell find $(SRC_DIR) -name *.cpp -or -name *.c -or -name *.s | sed 's`'$(SRC_DIR)'/``')
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

INC_DIRS := $(shell find $(SRC_DIR) -type d)
C_INC_FLAGS := $(addprefix -I, $(INC_DIRS))
AS_INC_FLAGS := $(addprefix -i, $(INC_DIRS))

CCFLAGS += -march=native $(C_INC_FLAGS) -MMD -MP -O4
CXXFLAGS += -march=native $(C_INC_FLAGS) -MMD -MP -O4
ASFLAGS += -felf64 $(AS_INC_FLAGS) -MD -MP
LDFLAGS += -z noexecstack

ifdef WERROR
	CCFLAGS += -Werror
	ASFLAGS += -Werror
endif

ifdef WALL
	CCFLAGS += -Wall
	ASFLAGS += -Wall
	CXXFLAGS += -Wall
endif

ifdef DEBUG
	CCFLAGS += -g -DDEBUG
	CXXFLAGS += -g -DDEBUG
	ASFLAGS += -g -dDEBUG
	LDFLAGS += -g -no-pie
	SHORTFLAGS := -g
endif

$(BUILD_DIR)/$(TARGET): $(OBJS)
	@echo link
	@$(LD) $(LDFLAGS) $^ -o $@
	@if [ -f '-MP' ]; then rm -- -MP; fi

$(BUILD_DIR)/%.c.o: $(SRC_DIR)/%.c
	@echo $(CC) $(SHORTFLAGS) $(patsubst $(BUILD_DIR)/%.o, %, $@)
	@mkdir -p $(dir $@)
	@$(CC) $(CCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.s.o: $(SRC_DIR)/%.s
	@echo $(AS) $(SHORTFLAGS) $(patsubst $(BUILD_DIR)/%.o, %, $@)
	@mkdir -p $(dir $@)
	@$(AS) $(ASFLAGS) $< -o $@

$(BUILD_DIR)/%.cpp.o: $(SRC_DIR)/%.cpp
	@echo $(CXX) $(SHORTFLAGS) $(patsubst $(BUILD_DIR)/%.o, %, $@)
	@mkdir -p $(dir $@)
	@$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -r $(BUILD_DIR)

-include $(DEPS)
