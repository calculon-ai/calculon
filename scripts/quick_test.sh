#!/bin/bash

set -e

# Single megatron run: 1T, 8-64-6, a100
echo "Testing 'megatron'"
PYTHONPATH=. ./bin/calculon megatron examples/1T.json examples/megatron_execution.json examples/a100.json - > /dev/null
echo "  passed"

# Validation
echo "Testing 'megatron-validation'"
PYTHONPATH=. ./bin/calculon megatron-validation > /dev/null
echo "  passed"
# Optimal search
echo "Testing 'megatron-optimal-execution'"
PYTHONPATH=. ./bin/calculon megatron-optimal-execution examples/22B.json 8 8 examples/h100.json > /dev/null
echo "  passed"

