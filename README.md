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
$> PYTHONPATH=. ./bin/calculon -h
```

## Megatron Example

``` sh
$> calculon megatron examples/megatron_application.json examples/megatron_execution.json examples/a100.json
```
