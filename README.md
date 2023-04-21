# Calculon - Co-design for large scale parallel applications

## Installing (optional)

``` sh
$> make install
$> calculon -h
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

Run a system execution optimizer for LLM (~5 mins):
``` sh
$> calculon llm-optimal-execution models/megatron-1T.json 4096 3072 float16 systems/a100_80g.json -e opt_exe.json -s opt_stats.json -m
```
`opt_exe.json` will contain the optimal way to run Megatron-1T across 4096 A100 GPUs.
