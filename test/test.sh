#!/bin/bash

set -e

export PYTHONPATH=.

# CLI interface infrastructure
echo -e "### Testing top level --help"
./bin/calculon --help > /dev/null
commands=$(./bin/calculon --help | head -n 2 | tail -n 1 | tr '{' ' ' | tr '}' ' ' | tr ',' ' ')
for command in $commands; do
    if [ $command == 'v' ] || [ $command == 'version' ]; then
	echo -e "### Testing \"$command\""
	./bin/calculon $command
    else
	echo -e "### Testing \"$command\" --help"
	./bin/calculon $command --help > /dev/null
    fi
done
echo -e "\n\n"

# Model size calculations
echo -e "### Testing megatron-parameter-calculator"
for model in models/*json; do
    ./bin/calculon megatron-parameter-calculator -a 15 $model
done
echo -e "\n\n"

# Model tests
echo -e "### Testing megatron"
for model in models/*json; do
    echo $model
    ./bin/calculon megatron $model examples/3072_t4_p64_d12_mbs4_full.json systems/a100_80e.json - > /dev/null
    ./bin/calculon megatron $model examples/3072_t4_p64_d12_mbs4_full.json systems/a100_80e.json /tmp/calculon_stats.json -p /tmp/calculon_peers.json
done
echo -e "\n\n"

# Megatron validation
echo -e "### Testing megatron-validation"
./bin/calculon mv -v
echo -e "\n\n"

# Megatron optimal execution
echo -e "### Testing megatron-optimal-execution (float16)"
./bin/calculon moe models/turing-530B.json 5128 2520 float16 systems/h100_80g_nvl8.json -e /tmp/calculon_exe.json -s /tmp/calculon_stats.json
echo -e "\n"

echo -e "### Testing megatron-optimal-execution (float8) (using -m)"
./bin/calculon moe models/turing-530B.json 5128 2520 float8 systems/h100_80g_nvl8.json -e /tmp/calculon_exe.json -s /tmp/calculon_stats.json -m
echo -e "\n\n"
