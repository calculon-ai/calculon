class System:
  """Configuration for a system."""
  def __init__(self, kvs):
    # bw = GB/s
    # cap = GB
    self.matrix_tflops = kvs['matrix_tflops']
    self.matrix_flop_eff = kvs['matrix_flop_eff']
    self.vector_tflops = kvs['vector_tflops']
    self.vector_flop_eff = kvs['vector_flop_eff']
    self.mem_tier1_bw = kvs['mem_tier1_bw']
    self.mem_tier1_cap = kvs['mem_tier1_cap']
    self.mem_tier1_eff = kvs['mem_tier1_eff']
    self.mem_tier2_bw = kvs['mem_tier2_bw']
    self.mem_tier2_cap = kvs['mem_tier2_cap']
    self.mem_tier2_eff = kvs['mem_tier2_eff']
    self.net_tier1_bw = kvs['net_tier1_bw']
    self.net_tier1_size = kvs['net_tier1_size']
    self.net_tier1_eff = kvs['net_tier1_eff']
    self.net_tier2_bw = kvs['net_tier2_bw']
    self.net_tier2_size = kvs['net_tier2_size']
    self.net_tier2_eff = kvs['net_tier2_eff']
