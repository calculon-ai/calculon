from layers import *
from megatron import *

sw_config = {
  'n' : 65536,
  't' : 32,
  'p' : 32,
  'd' : 64,
  'batch_size' : 4096,
  'minibatch_size' : 1,
  'datatype' : 'bfloat16',
  'act_recompute' : 'partial',
  'pipeline_interleaving' : 4,
  'optim_sharding' : True,
  'sharp' : True,
  'seq_par' : True,
  'point_to_point_rs_ag' : True,
  'dp_overlap' : True,
  'weight_offload' : True,
  'activations_offload' : True,
  'optimizer_offload' : True,
  'training' : True,
}

a100 = {
  'matrix_tflops' : 312,     # TFlops
  'matrix_flop_eff': 0.9,
  'vector_tflops' : 78,      # TFlops
  'vector_flop_eff' : 0.9,
  'mem_tier1_bw' : 2*1024,    # GB/s
  'mem_tier1_cap' : 80,       # GB
  'mem_tier1_eff' : 0.9,
  'mem_tier2_bw' : 512,       # GB/s
  'mem_tier2_cap' : 1024,     # GB
  'mem_tier2_eff' : 0.9,
  'net_tier1_bw' : 300,       # GB/s
  'net_tier1_size' : 32,      # NVLink GPUs
  'net_tier1_eff' : 0.8,
  'net_tier2_bw' : 25,        # GB/s
  'net_tier2_size' : 65536,      # IB GPUs
  'net_tier2_eff' : 0.9,
}

h100 = {
  'matrix_tflops' : 1000,     # TFlops
  'matrix_flop_eff': 0.9,
  'vector_tflops' : 120,      # TFlops
  'vector_flop_eff' : 0.9,
  'mem_tier1_bw' : 3*1024,    # GB/s
  'mem_tier1_cap' : 80,       # GB
  'mem_tier1_eff' : 0.9,
  'mem_tier2_bw' : 512,       # GB/s
  'mem_tier2_cap' : 1024,     # GB
  'mem_tier2_eff' : 0.9,
  'net_tier1_bw' : 450,       # GB/s
  'net_tier1_size' : 32,      # NVLink GPUs
  'net_tier1_eff' : 0.8,
  'net_tier2_bw' : 50,        # GB/s
  'net_tier2_size' : 64,      # IB GPUs
  'net_tier2_eff' : 0.9,
}

model = Megatron("Megatron-1T", 25600, 2048, 4096, 160, 128)
model.compile(sw_config)
model.run(a100)
model.display_stats()
