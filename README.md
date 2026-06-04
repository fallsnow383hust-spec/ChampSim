# ChampSim

![GitHub](https://img.shields.io/github/license/ChampSim/ChampSim)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ChampSim/ChampSim/test.yml)
![GitHub forks](https://img.shields.io/github/forks/ChampSim/ChampSim)
[![Coverage Status](https://coveralls.io/repos/github/ChampSim/ChampSim/badge.svg?branch=develop)](https://coveralls.io/github/ChampSim/ChampSim?branch=develop)

ChampSim is a trace-based simulator for a microarchitecture study. If you have questions about how to use ChampSim, we encourage you to search the threads in the Discussions tab or start your own thread. If you are aware of a bug or have a feature request, open a new Issue.

# Using ChampSim

ChampSim is the result of academic research. If you use this software in your work, please cite it using the following reference:

    Gober, N., Chacon, G., Wang, L., Gratz, P. V., Jimenez, D. A., Teran, E., Pugsley, S., & Kim, J. (2022). The Championship Simulator: Architectural Simulation for Education and Competition. https://doi.org/10.48550/arXiv.2210.14324

If you use ChampSim in your work, you may submit a pull request modifying `PUBLICATIONS_USING_CHAMPSIM.bib` to have it featured in [the documentation](https://champsim.github.io/ChampSim/master/Publications-using-champsim.html).

# Download dependencies

ChampSim uses [vcpkg](https://vcpkg.io) to manage its dependencies. In this repository, vcpkg is included as a submodule. You can download the dependencies with
```
git submodule update --init
vcpkg/bootstrap-vcpkg.sh
vcpkg/vcpkg install
```

# Compile

ChampSim is a single pre-compiled binary. Build it with:
```
$ make
```
or, for a parallel build:
```
$ make -j$(nproc)
```
The binary is placed at `bin/champsim`.

# Download DPC-3 trace

Traces used for the 3rd Data Prefetching Championship (DPC-3) can be found here. (https://dpc3.compas.cs.stonybrook.edu/champsim-traces/speccpu/) A set of traces used for the 2nd Cache Replacement Championship (CRC-2) can be found from this link. (http://bit.ly/2t2nkUj)

Storage for these traces is kindly provided by Daniel Jimenez (Texas A&M University) and Mike Ferdman (Stony Brook University). If you find yourself frequently using ChampSim, it is highly encouraged that you maintain your own repository of traces, in case the links ever break.

# Run simulation

Execute the binary directly. Use `--config` to supply a JSON configuration file:
```
$ bin/champsim --config my_config.json --warmup-instructions 200000000 --simulation-instructions 500000000 ~/path/to/traces/600.perlbench_s-210B.champsimtrace.xz
```
If `--config` is omitted, ChampSim loads `champsim_config.json` from the current directory by default. You can also pipe a config via stdin with `--config -`.

The number of warmup and simulation instructions given will be the number of instructions retired. Note that the statistics printed at the end of the simulation include only the simulation phase.

ChampSim supports two configuration formats:
- **Legacy format** (`champsim_config.json`): A shorthand format that auto-generates the full module hierarchy from high-level keys like `"branch_predictor"`, `"L1D"`, `"num_cores"`, etc.
- **Explicit format** (`champsim_config_explicit.json`): Gives full control over every module in the hierarchy. Set `"environment": "ENVIRONMENT"` and define all modules in a `"children"` array. See the [documentation](https://champsim.github.io/ChampSim/master/Explicit-configuration-format.html) for details.

# Add your own branch predictor, data prefetchers, and replacement policy

Modules are C++ classes that inherit from an interface base class in the `champsim::modules` namespace. The four user-facing interfaces are `prefetcher`, `replacement`, `branch_predictor`, and `btb`.

**Create a new module**
```
$ mkdir prefetcher/mypref
```

Create `prefetcher/mypref/mypref.h`:
```cpp
#include "modules.h"

struct mypref : champsim::modules::prefetcher {
    mypref(champsim::modules::ModuleBuilder builder);
    void prefetcher_initialize() override {}
    uint32_t prefetcher_cache_operate(champsim::address addr, champsim::address ip,
        bool cache_hit, bool useful_prefetch, access_type type, uint32_t metadata_in) override;
    uint32_t prefetcher_cache_fill(champsim::address, long, long, bool,
        champsim::address, uint32_t metadata_in) override { return metadata_in; }
    void prefetcher_cycle_operate() override {}
    void prefetcher_final_stats() override {}
    void prefetcher_branch_operate(champsim::address, uint8_t, champsim::address) override {}
};
```

Create `prefetcher/mypref/mypref.cc`:
```cpp
#include "mypref.h"
#include "cache.h"

// Register the module so the simulator can find it by name
champsim::modules::prefetcher::register_module<mypref> mypref_register("mypref");

mypref::mypref(champsim::modules::ModuleBuilder builder) {
    // Read parameters from the JSON config if needed:
    // int degree = builder.get_parameter<int>("degree", true, 3);
}

uint32_t mypref::prefetcher_cache_operate(champsim::address addr, champsim::address ip,
    bool cache_hit, bool useful_prefetch, access_type type, uint32_t metadata_in) {
    // Your prefetching logic here
    return metadata_in;
}
```

**Compile and test**

Add your prefetcher to a configuration file:
```json
{
    "L2C": {
        "prefetcher": "mypref"
    }
}
```
Note that the example prefetcher is an L2 prefetcher. You might design a prefetcher for a different level.

```
$ make
$ bin/champsim --config my_config.json --warmup-instructions 200000000 --simulation-instructions 500000000 600.perlbench_s-210B.champsimtrace.xz
```

All interface methods are **pure virtual** and must be overridden (use empty stubs `override {}` where no behavior is needed). See the [module system documentation](https://champsim.github.io/ChampSim/master/Modules.html) for complete interface signatures and details.

# How to create traces

Program traces are available in a variety of locations, however, many ChampSim users wish to trace their own programs for research purposes.
Example tracing utilities are provided in the `tracer/` directory.

# Evaluate Simulation

ChampSim measures the IPC (Instruction Per Cycle) value as a performance metric. <br>
There are some other useful metrics printed out at the end of simulation. <br>

Good luck and be a champion! <br>
