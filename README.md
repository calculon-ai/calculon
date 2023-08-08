[![DOI](https://zenodo.org/badge/660734586.svg)](https://zenodo.org/badge/latestdoi/660734586)
# Calculon - Co-design for large scale parallel applications

## Installing (optional)

``` sh
$> make install
$> calculon -h
```

## Testing and validation (optional)
To make sure that the current build is working, use

``` sh
$> make test
```
To validate Calculon performance modeling against Megatron run on NVIDIA's Selene A100-based supercomputer with results published in ["Sequence parallelism" paper](https://arxiv.org/abs/2205.05198), use

``` sh
$> calculon llm-validation
```

## Running

If Calculon is installed (see above), run it like this:
``` sh
$> calculon <args>
```

If Calculon is NOT installed, you can run it locally like this:

``` sh
$> PYTHONPATH=. ./bin/calculon <args>
```

Calculon is a hierarchical command line. To see the commands it accepts, use `--help` or `-h`:
``` sh
$> calculon -h
```

You can also see how to use any command specifically by using `--help` or `-h` on the command:
``` sh
$> calculon llm -h
```

## LLM Example

Run a single calculation for LLM (~1 sec):
``` sh
$> calculon llm models/megatron-1T.json examples/3072_t4_p64_d12_mbs4_full.json systems/a100_80g.json -
```

Run a system execution optimizer for LLM (~1 min):
``` sh
$> calculon llm-optimal-execution models/turing-530B.json 5128 2520 float16 systems/a100_80g.json output.json -m
```
`opt_exe.json` will contain the optimal way to run Turing-530B across 5128 A100 GPUs.

To store results from all successful runs from the same experiment, run a special system optimizer (~1 min):
``` sh
$> calculon llm-all-executions models/turing-530B.json 5128 2520 float16 systems/a100_80g.json all_output.csv
```
