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
echo -e "### Testing llm-parameter-calculator"
for model in models/*json; do
    ./bin/calculon llm-parameter-calculator -a 15 $model
done
echo -e "\n\n"

# Model tests
echo -e "### Testing llm"
for model in models/*json; do
    echo $model
    ./bin/calculon llm $model examples/3072_t4_p64_d12_mbs4_full.json systems/a100_80e.json - > /dev/null
    ./bin/calculon llm $model examples/3072_t4_p64_d12_mbs4_full.json systems/a100_80e.json /tmp/calculon_stats.json -p /tmp/calculon_peers.json
done
echo -e "\n\n"

# Llm validation
echo -e "### Testing llm-validation"
./bin/calculon lv -v
echo -e "\n\n"

# Llm optimal execution
echo -e "### Testing llm-optimal-execution (float16) (using -f)"
./bin/calculon loe models/turing-530B.json 5128 2520 float16 systems/h100_80g_nvl8.json /tmp/calculon_530B_fp16.json -t 3 -f False --no-tp-overlap
echo -e "\n"

echo -e "### Testing llm-optimal-execution (float8) (using -m)"
./bin/calculon loe models/turing-530B.json 5128 2520 float8 systems/h100_80g_nvl8.json /tmp/calculon_530B_fp8.csv.gz -t 10 -m
echo -e "\n\n"

# Llm all executions
echo -e "### Testing llm-all-executions (float8)"
./bin/calculon lae models/turing-530B.json 5128 2520 float8 systems/h100_80g_nvl8.json /tmp/calculon_530B_fp8_all.csv.gz
echo -e "\n\n"

