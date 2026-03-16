override ROOT_DIR = $(patsubst %/,%,$(dir $(abspath $(firstword $(MAKEFILE_LIST)))))

# Customization points:
#  - BIN_ROOT: at make-time, override the binary directory
#  - OBJ_ROOT: at make-time, override the object file directory
#  - DEP_ROOT: at make-time, override the dependency file directory
BIN_ROOT := bin
OBJ_ROOT := .csconfig
DEP_ROOT := $(OBJ_ROOT)

# vcpkg integration
TRIPLET_DIR = $(patsubst %/,%,$(firstword $(filter-out $(ROOT_DIR)/vcpkg_installed/vcpkg/, $(wildcard $(ROOT_DIR)/vcpkg_installed/*/))))
override LDFLAGS  += -L$(TRIPLET_DIR)/lib -L$(TRIPLET_DIR)/lib/manual-link
override LDLIBS   += -lCLI11 -llzma -lz -lbz2 -lfmt

.PHONY: all clean test
.DEFAULT_GOAL := all
.SECONDEXPANSION:

executable_name := $(BIN_ROOT)/champsim
test_main_name  := test/bin/000-test-main

### Source discovery
src_sources    := $(wildcard src/*.cc)
module_sources := $(shell find branch btb prefetcher replacement -name "*.cc" 2>/dev/null)
test_sources   := $(wildcard test/cpp/src/*.cc)

### Object files (mirror source tree under OBJ_ROOT)
src_objs    := $(patsubst %.cc,$(OBJ_ROOT)/%.o,$(src_sources))
module_objs := $(patsubst %.cc,$(OBJ_ROOT)/%.o,$(module_sources))
test_objs   := $(patsubst test/cpp/src/%.cc,$(OBJ_ROOT)/test/%.o,$(test_sources))
all_objs    := $(src_objs) $(module_objs) $(test_objs)

### Options files
absolute.options:
	@echo "-I$(realpath inc) -isystem $(realpath $(TRIPLET_DIR)/include)" > $@

base_options   := absolute.options global.options
module_options := absolute.options global.options module.options

# Reverse the options list so that more-specific options are applied last
reverse = $(if $(wordlist 2,2,$(1)),$(call reverse,$(wordlist 2,$(words $1),$1)) $(firstword $(1)),$(1))
attach_options = $(call reverse, $(addprefix @,$(filter %.options, $^)))

### Recipes
define obj_recipe
	$(CXX) $(attach_options) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $(filter %.cc, $^)
endef

DEPFLAGS = -MM -MT $@ -MT $(@:.d=.o)
define dep_recipe
	$(CXX) $(attach_options) $(DEPFLAGS) $(CPPFLAGS) -MF $@ $(filter %.cc, $^)
endef

### Targets
all: $(executable_name)

### Directory creation
$(sort $(dir $(all_objs))) $(BIN_ROOT)/ test/bin/:
	mkdir -p $@

### Compilation rules - core sources
$(OBJ_ROOT)/src/%.o: src/%.cc $(base_options) | $$(dir $$@)
	$(obj_recipe)
$(DEP_ROOT)/src/%.d: src/%.cc $(base_options) | $$(dir $$@)
	$(dep_recipe)

### Compilation rules - modules (branch, btb, prefetcher, replacement)
$(OBJ_ROOT)/branch/%.o: branch/%.cc $(module_options) | $$(dir $$@)
	$(obj_recipe)
$(DEP_ROOT)/branch/%.d: branch/%.cc $(module_options) | $$(dir $$@)
	$(dep_recipe)

$(OBJ_ROOT)/btb/%.o: btb/%.cc $(module_options) | $$(dir $$@)
	$(obj_recipe)
$(DEP_ROOT)/btb/%.d: btb/%.cc $(module_options) | $$(dir $$@)
	$(dep_recipe)

$(OBJ_ROOT)/prefetcher/%.o: prefetcher/%.cc $(module_options) | $$(dir $$@)
	$(obj_recipe)
$(DEP_ROOT)/prefetcher/%.d: prefetcher/%.cc $(module_options) | $$(dir $$@)
	$(dep_recipe)

$(OBJ_ROOT)/replacement/%.o: replacement/%.cc $(module_options) | $$(dir $$@)
	$(obj_recipe)
$(DEP_ROOT)/replacement/%.d: replacement/%.cc $(module_options) | $$(dir $$@)
	$(dep_recipe)

### Compilation rules - tests
$(OBJ_ROOT)/test/%.o: test/cpp/src/%.cc $(base_options) | $$(dir $$@)
	$(obj_recipe)
$(DEP_ROOT)/test/%.d: test/cpp/src/%.cc $(base_options) | $$(dir $$@)
	$(dep_recipe)

### Linking
$(executable_name): $(src_objs) $(module_objs) | $(BIN_ROOT)/
	$(CXX) $(LDFLAGS) -o $@ $^ $(LOADLIBES) $(LDLIBS)

# Tests exclude main.o (000-test-main.cc provides its own main and globals)
$(test_main_name): override CXXFLAGS += -g3 -Og
$(test_main_name): override LDLIBS += -lCatch2Main -lCatch2
$(test_main_name): $(filter-out $(OBJ_ROOT)/src/main.o,$(src_objs)) $(test_objs) $(module_objs) | test/bin/
	$(CXX) $(LDFLAGS) -o $@ $^ $(LOADLIBES) $(LDLIBS)

### Tests
ifdef TEST_NUM
selected_test = -\# "[$(addprefix \#,$(filter $(addsuffix %,$(TEST_NUM)), $(patsubst %.cc,%,$(notdir $(test_sources)))))]"
endif
test: $(test_main_name)
	$(test_main_name) $(selected_test)

### Clean
clean:
	@-find $(OBJ_ROOT) $(DEP_ROOT) -type f \( -name "*.o" -o -name "*.d" \) -delete 2>/dev/null
	@-$(RM) $(executable_name) $(test_main_name)

### Auto-dependencies
ifeq (,$(filter clean, $(MAKECMDGOALS)))
-include $(patsubst $(OBJ_ROOT)/%.o,$(DEP_ROOT)/%.d,$(all_objs))
endif
