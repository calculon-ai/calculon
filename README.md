[![DOI](https://zenodo.org/badge/660734586.svg)](https://zenodo.org/badge/latestdoi/660734586)
# Calculon - Co-design for large scale parallel applications

## Running

Run Calculon like this:
``` sh
$> PYTHONPATH=. ./bin/ <args>
```

Calculon is a hierarchical command line. To see the commands it accepts, use `--help` or `-h`:
``` sh
$> PYTHONPATH=. ./bin/ -h
```

You can also see how to use any command specifically by using `--help` or `-h` on the command:
``` sh
$> PYTHONPATH=. ./bin/ llm -h
```

## LLM Example

Run a single calculation for LLM (~1 sec):
``` sh
$> PYTHONPATH=. ./bin/ llm models/megatron-1T.json examples/3072_t4_p64_d12_mbs4_full.json systems/a100_80g.json -
```

Run a system execution optimizer for LLM (~1 min):
``` sh
$> PYTHONPATH=. ./bin/ llm-optimal-execution models/turing-530B.json 5128 2520 float16 systems/a100_80g.json output.json -m
```
`opt_exe.json` will contain the optimal way to run Turing-530B across 5128 A100 GPUs.

To store results from all successful runs from the same experiment, run a special system optimizer (~1 min):
``` sh
$> PYTHONPATH=. ./bin/ llm-all-executions models/turing-530B.json 5128 2520 float16 systems/a100_80g.json all_output.csv
```

## Testing and validation (optional)
To make sure that the current build is working, use

``` sh
$> make test
```
To validate Calculon performance modeling against Megatron run on NVIDIA's Selene A100-based supercomputer with results published in ["Sequence parallelism" paper](https://arxiv.org/abs/2205.05198), use

``` sh
$> PYTHONPATH=. ./bin/calculon llm-validation
```

## Publications

* Calculon: A Methodology and Tool for High-Level Co-Design of Systems and Large Language Models\
Mikhail Isaev, Nic McDonald, Larry Dennison, Richard Vuduc\
[Paper](https://dl.acm.org/doi/pdf/10.1145/3581784.3607102)

* Scaling Infrastructure to Support Multi-Trillion Parameter LLM Training\
Mikhail Isaev, Nic McDonald, Richard Vuduc\
[Paper](https://openreview.net/pdf?id=rqn2v1Ltgn0)
