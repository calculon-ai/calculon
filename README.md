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
$> calculon megatron -h
```

## Megatron Example

Run a single calculation for Megatron (~1 sec):
``` sh
$> calculon megatron examples/1T.json examples/megatron_execution.json examples/a100.json -
```

Run a system execution optimizer for Megatron (~5 mins):
``` sh
$> calculon megatron-optimal-execution examples/1T.json 4096 3072 examples/a100.json -e opt_exe.json -s opt_stats.json -r opt_raw.json
```
`opt_exe.json` will contain the optimal way to run Megatron-1T across 4096 A100 GPUs.

You can see a 3D representation of the parallelism split using this command:
``` sh
$> ./scripts/3dplot.py opt_raw.json
```
