#!/bin/bash
# Multi-NPU Training Launch Script for SFM on Ascend

set -e

# Default values
NPUS_PER_NODE=4
NODES=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29500
QUICK_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --npus)
            NPUS_PER_NODE="$2"
            shift 2
            ;;
        --nodes)
            NODES="$2"
            shift 2
            ;;
        --node_rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --master_addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --npus N         Number of NPUs per node (default: 4)"
            echo "  --nodes N        Number of nodes (default: 1)"
            echo "  --node_rank N    Rank of this node (default: 0)"
            echo "  --master_addr IP Master node address (default: 127.0.0.1)"
            echo "  --master_port PORT Master port (default: 29500)"
            echo "  --quick          Quick test mode"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Calculate world size
WORLD_SIZE=$((NPUS_PER_NODE * NODES))

echo "========================================"
echo "State-Flow Machine - Distributed Training"
echo "========================================"
echo "NPUs per node: $NPUS_PER_NODE"
echo "Nodes: $NODES"
echo "World size: $WORLD_SIZE"
echo "Node rank: $NODE_RANK"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "========================================"

# Build torchrun command
TORCHRUN_CMD="torchrun"

if [ "$NODES" -gt 1 ]; then
    TORCHRUN_CMD="$TORCHRUN_CMD --nnodes=$NODES --nproc_per_node=$NPUS_PER_NODE"
    TORCHRUN_CMD="$TORCHRUN_CMD --node_rank=$NODE_RANK"
    TORCHRUN_CMD="$TORCHRUN_CMD --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
else
    TORCHRUN_CMD="$TORCHRUN_CMD --nproc_per_node=$NPUS_PER_NODE"
fi

# Add training script
if [ "$QUICK_MODE" = true ]; then
    TORCHRUN_CMD="$TORCHRUN_CMD train_distributed.py --quick"
else
    TORCHRUN_CMD="$TORCHRUN_CMD train_distributed.py"
fi

echo "Running: $TORCHRUN_CMD"
echo ""

cd experiments/exp0_state_tracking
$TORCHRUN_CMD
