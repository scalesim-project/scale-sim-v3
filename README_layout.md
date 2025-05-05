# Layout Implementation in SCALE-Sim v3

## Overview
SCALE-Sim v3  models the memory system and layout organization of neural network accelerators. This document provides detailed information about the layout modeling features and implementation details.

## Key Components

The layout modeling system consists of two main components:

1. **Layout Modeling**: Supports various data organizations in the on-chip buffer
2. **Memory Latency Evaluation with Layout Consideration**: Evaluates actual bank conflicts under various bank organizations

## New Features

### On-Chip Buffer Bank Conflict Evaluation
- Introduces two new configurations:
  - `OnChipMemoryBanks`: Controls the total number of banks
  - `OnChipMemoryBankPorts`: Controls the total number of ports per bank

### Layout Modeling
- Supports per-layer layout specifications
- Each topology has its layout specification organized in a single CSV file
- Controlled via:
  - `IfmapCustomLayout` and `FilterCustomLayout` (True/False)
  - True: Requires a layout to be specified
  - False: Evaluates bank conflict assuming precompiled order

## Layout Specification

Layout is specified using three configuration parameters for each tensor:

1. **Intraline Factor**: Specifies how much data each line across all banks holds
   ```
   ifmap_intraline_factor = [2, 2, 16]  # For dimensions X, Y, C
   ```

2. **Intraline Order**: Specifies the order of dimensions within a line
   ```
   ifmap_intraline_order = [0, 1, 2]  # Order: x->y->c
   ```

3. **Interline Order**: Specifies the order of dimensions across lines
   ```
   ifmap_interline_order = [1, 2, 0]  # Order: c->x->y
   ```

## Memory System Architecture

### On-Chip Buffer Organization
- Organized in 2D structure
- Each horizontal row represents a single set containing the first row of all on-chip banks
- Total bandwidth across all banks equals the overall bandwidth
- Each bank has its own ports for fine-grained data access

### Buffer Components
1. **Active Buffer**: Holds data currently being used
2. **Prefetch Buffer**: Holds data to be used in the future
- Both buffers share the same memory space, each taking half of the total memory

## Simulation Flow

1. **Initialization**:
   - Parse input configurations
   - Initialize simulator and memory system
   - Set up layer parameters

2. **Prefetch Phase**:
   - Load prefetched ifmap and filter data into memory system
   - Create hashed buffer for efficient access
   - Handle data reformatting when matrix dimensions don't match buffer bandwidth

3. **Memory Request Serving**:
   - Process incoming memory requests
   - Handle bank conflicts and access patterns
   - Track stall cycles and latency

## Configuration Parameters

### Memory System Parameters
- `Bandwidth`: Number of data elements DRAM can provide per cycle
- `MemoryBanks`: Number of memory banks
- `active_buf_frac`: Percentage of total on-chip buffer allocated for active buffer
- `hit_latency`: Latency of accessing data from on-chip buffer
- `total_size_elems`: Total number of elements the on-chip buffer can hold

### Layout Parameters
- `IfmapCustomLayout`: Enable/disable custom ifmap layout
- `FilterCustomLayout`: Enable/disable custom filter layout
- `OnChipMemoryBanks`: Number of on-chip memory banks
- `OnChipMemoryBankPorts`: Number of ports per on-chip memory bank

## Usage Example

```bash
python3 ./scalesim/scale.py -c configs/scale_layout_bk4.cfg -l layout/MPQCRS.cfg -t topologies/conv_nets/alexnet_part.csv -p ./ >> log
```

## Limitations

1. Single-layer simulation only (no inter-layer fusion or pipelining)
2. No consideration of layout reorganization between layers
3. All tensors (ifmap, filter, ofmap) use the same bandwidth
4. Currently only models on-chip buffer reading without writing
5. Only input layouts are specified, not output layouts

## Future Work

1. Implement inter-layer layout reordering evaluation
2. Support different bandwidths for different tensors
3. Add output layout specification
4. Implement write operation modeling
5. Support more flexible bank organization configurations 