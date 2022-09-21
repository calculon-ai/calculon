from layers import *

class Megatron: # stems from class (ParaGraph)
    '''
    A Megatron class that implements transformer with tensor, pipeline, and data
    parallelism.
    We should 
    1. Initialize the model with certain model parameters
    2. Compile it with certain optimizations and parallelization strategies
    3. Run on particular hardware system
    '''
    # TODO refactor to be a member of Application class
    def __init__(self, name, hidden, seq_size, batch_size,
                 attn_heads, num_layers):
        self.name = name
        self.hidden = hidden
        self.seq_size = seq_size
        self.batch_size = batch_size
        self.attn_heads = attn_heads
        self.num_layers = num_layers

        # TODO generalize layers to be a graph
        self.megatron_block = []

        # HW parameters to populate during run
        self.vector_throughput = 0
        self.matrix_throughput = 0
        self.mem_throughput = 0
        self.offload_throughput = 0
        self.tp_net_throughput = 0
        self.dp_net_throughput = 0
        self.pp_net_throughput = 0

        # metrics collected after run for each minibatch
        self.minibatch_fw_flops = 0
        self.minibatch_fw_flops_time = 0
        self.minibatch_fw_mem_accessed = 0
        self.minibatch_fw_mem_time = 0
        self.minibatch_bw_flops = 0
        self.minibatch_bw_flops_time = 0
        self.minibatch_bw_mem_accessed = 0
        self.minibatch_bw_mem_time = 0
        self.minibatch_recompute_mem_saving = 0
        self.minibatch_recompute_time = 0
        self.minibatch_fw_tp_size = 0
        self.minibatch_fw_tp_time = 0
        self.minibatch_bw_tp_size = 0
        self.minibatch_bw_tp_time = 0
        self.minibatch_fw_pp_size = 0
        self.minibatch_fw_pp_time = 0
        self.minibatch_bw_pp_size = 0
        self.minibatch_bw_pp_time = 0

        # metrics collected after run for each batch on a single GPU
        self.gpu_weight_space = 0
        self.gpu_act_space = 0
        self.gpu_act_checkpoint_size = 0
        self.gpu_weight_grad_space = 0
        self.gpu_act_grad_space = 0
        self.gpu_optim_space = 0
        self.gpu_fw_flops = 0
        self.gpu_fw_flops_time = 0
        self.gpu_fw_mem_accessed = 0
        self.gpu_fw_mem_time = 0
        self.gpu_bw_flops = 0
        self.gpu_bw_flops_time = 0
        self.gpu_bw_mem_accessed = 0
        self.gpu_bw_mem_time = 0
        self.gpu_recompute_time = 0
        self.gpu_tp_comm_size = 0
        self.gpu_tp_comm_time = 0
        self.gpu_pp_comm_size = 0
        self.gpu_pp_comm_time = 0
        self.gpu_dp_comm_size = 0
        self.gpu_dp_comm_time = 0

        self.fw_time = 0
        self.bw_time = 0
        self.recompute_time = 0
        self.bubble_time = 0
        self.tp_comm_time = 0
        self.pp_comm_time = 0
        self.dp_comm_time = 0
        self.total_time = 0
        self.gpu_mem_cap_req = 0
        self.offload_mem_bw_req = 0
        self.total_weight_space = 0
        self.total_act_space = 0
        self.total_weight_grad_space = 0
        self.total_act_grad_space = 0
        self.total_optim_space = 0

    def _build_attn_block(self):
        recompute_flag = False
        recompute_attn_flag = False
        if self.act_recompute == "full":
            recompute_flag = True
        if self.act_recompute == "full" or self.act_recompute == "partial":
            recompute_attn_flag = True
        if self.seq_par:
            self.megatron_block.append(LinearNorm("AttnBlock_LinearNorm",
                                                  self.seq_par_activation_size,
                                                  self.hidden,
                                                  needs_recompute=\
                                                  recompute_flag))
        else: 
            self.megatron_block.append(LinearNorm("AttnBlock_LinearNorm",
                                                  self.hidden,
                                                  self.activation_size,
                                                  needs_recompute=\
                                                  recompute_flag))
        self.megatron_block.append(Fork("AttnBlock_Fork",
                                         self.activation_size, 3))
        self.megatron_block.append(Linear("AttnBlock_Key",
                                          self.batch_seq,
                                          self.hidden,
                                          self.hidden / self.t,
                                          activation_reuse=True,
                                          needs_recompute=recompute_attn_flag))
        self.megatron_block.append(Linear("AttnBlock_Query",
                                          self.batch_seq,
                                          self.hidden,
                                          self.hidden / self.t,
                                          activation_reuse=True,
                                          needs_recompute=recompute_attn_flag))
        self.megatron_block.append(Linear("AttnBlock_Value",
                                          self.batch_seq,
                                          self.hidden,
                                          self.hidden / self.t,
                                          activation_reuse=True,
                                          needs_recompute=recompute_attn_flag))
        self.megatron_block.append(MatMul("AttnBlock_Multihead_Key_Query",
                                          self.batch_seq,
                                          self.hidden / self.t,
                                          self.batch_seq,
                                          needs_recompute=recompute_attn_flag))
        self.megatron_block.append(SoftMax("AttnBlock_Multihead_SoftMax",
                                           self.activation_size,
                                          needs_recompute=recompute_attn_flag))
        self.megatron_block.append(DropOut("AttnBlock_Multihead_DropOut",
                                           self.activation_size,
                                          needs_recompute=recompute_attn_flag))
        self.megatron_block.append(MatMul("AttnBlock_Multihead_Attn",
                                          self.batch_seq,
                                          self.hidden / self.t,
                                          self.batch_seq,
                                          needs_recompute=recompute_attn_flag))
        self.megatron_block.append(Linear("AttnBlock_MLP",
                                          self.batch_seq,
                                          self.hidden / self.t,
                                          self.hidden,
                                          needs_recompute=recompute_flag))
        if self.seq_par:
            self.megatron_block.append(DropOut("AttnBlock_DropOut",
                                           self.seq_par_activation_size,
                                           needs_recompute=recompute_flag))
            self.megatron_block.append(ElementWise("AttnBlock_Residual",
                                                   self.seq_par_activation_size,
                                                   self.seq_par_activation_size,
                                                   needs_recompute=\
                                                   recompute_flag))
        else: 
            self.megatron_block.append(DropOut("AttnBlock_DropOut",
                                               self.activation_size,
                                               needs_recompute=recompute_flag))
            self.megatron_block.append(ElementWise("AttnBlock_Residual",
                                                   self.activation_size,
                                                   self.activation_size,
                                                   needs_recompute=\
                                                   recompute_flag))
    
    def _build_mlp_block(self):
        recompute_flag = False
        if self.act_recompute == "full":
            recompute_flag = True
        if self.seq_par:
            self.megatron_block.append(LinearNorm("MlpBlock_LinearNorm",
                                                  self.seq_par_activation_size,
                                                  self.hidden,
                                                  needs_recompute=\
                                                  recompute_flag))
        else: 
            self.megatron_block.append(LinearNorm("MlpBlock_LinearNorm",
                                                  self.hidden,
                                                  self.activation_size,
                                                  needs_recompute=\
                                                  recompute_flag))
        self.megatron_block.append(Linear("MlpBlock_MLP1",
                                          self.batch_seq,
                                          self.hidden,
                                          self.hidden*4 / self.t,
                                          needs_recompute=recompute_flag))
        self.megatron_block.append(GeLU("MlpBlock_GeLU",
                                        self.activation_size / self.t,
                                        needs_recompute=recompute_flag))
        self.megatron_block.append(Linear("MlpBlock_MLP2",
                                          self.batch_seq,
                                          self.hidden*4 / self.t,
                                          self.hidden,
                                          needs_recompute=recompute_flag))
        if self.seq_par:
            self.megatron_block.append(DropOut("MlpBlock_DropOut",
                                               self.seq_par_activation_size,
                                               needs_recompute=recompute_flag))
            self.megatron_block.append(ElementWise("MlpBlock_Residual",
                                                   self.seq_par_activation_size,
                                                   self.seq_par_activation_size,
                                                   needs_recompute=\
                                                   recompute_flag))
        else: 
            self.megatron_block.append(DropOut("MlpBlock_DropOut",
                                               self.activation_size,
                                               needs_recompute=recompute_flag))
            self.megatron_block.append(ElementWise("MlpBlock_Residual",
                                                   self.activation_size,
                                                   self.activation_size,
                                                   needs_recompute=\
                                                   recompute_flag))

    # TODO move wherever appropriate, e.g. some config class
    types_size_dict = {
        'float8'    : 1,
        'float16'   : 2,
        'float32'   : 4,
        'bfloat16'  : 2
    }

    def compile(self, sw_config):
        self.sw_config = sw_config
        self.num_gpus = sw_config.get('n', 1)
        self.training = sw_config.get('training', True)
        self.p = sw_config.get('p', 1)
        self.t = sw_config.get('t', 1)
        self.d = sw_config.get('d', 1)
        assert self.p * self.t * self.d == self.num_gpus, "t*p*d != num_gpus"
        self.batch = sw_config.get('batch_size', 1)
        self.minibatch_size = sw_config.get('minibatch_size', 1)
        self.num_minibatches = self.batch / self.d / self.minibatch_size
        self.layers_per_gpu = self.num_layers / self.p
        self.datatype = sw_config.get('datatype', 'float16')
        self.bytes_per_element = self.types_size_dict[self.datatype]
        self.act_recompute = sw_config.get('act_recompute', 'full')
        self.optim_sharding = sw_config.get('optim_sharding', False)
        self.pipeline_interleaving = sw_config.get('pipeline_interleaving', 1)
        self.sharp = sw_config.get('sharp', False)
        self.seq_par = sw_config.get('seq_par', False)
        self.point_to_point_rs_ag = sw_config.get('point_to_point_rs_ag',
                                                   False)
        self.dp_overlap = sw_config.get('dp_overlap', False)
        # Build model during the compilation step
        self.batch_seq = self.minibatch_size * self.seq_size
        self.activation_size = self.batch_seq * self.hidden
        self.batch_seq_par = self.batch_seq / self.t
        self.seq_par_activation_size = self.batch_seq_par * self.hidden
        self._build_attn_block()
        self._build_mlp_block()
        # TODO add f/g functions to properly account for activation space?
        for layer in self.megatron_block:
            layer.set_bytes_per_element(self.bytes_per_element)
        self._compiled = True

    def _update_hw_throughput(self):
        self.vector_throughput = self.hw_config['vector_tflops'] * 1e12 * \
                                 self.hw_config['vector_flop_eff']
        self.matrix_throughput = self.hw_config['matrix_tflops'] * 1e12 * \
                                 self.hw_config['matrix_flop_eff']
        self.mem_throughput = self.hw_config['mem_tier1_bw'] * \
                              self.hw_config['mem_tier1_eff'] * 1024 ** 3
        self.offload_throughput = self.hw_config['mem_tier2_bw'] * \
                                  self.hw_config['mem_tier2_eff'] * 1024 ** 3
        assert (self.t <= self.hw_config['net_tier1_size'] or
                self.t <= self.hw_config['net_tier2_size']), \
                f"t={self.t} is larger than the network " \
                f"size {self.hw_config['net_tier1_size']} " \
                f"or {self.hw_config['net_tier2_size']}"
        self.tp_net_throughput = self.hw_config['net_tier2_bw'] * \
                                 self.hw_config['net_tier2_eff'] * 1024 ** 3
        if self.t <= self.hw_config['net_tier1_size']:
            self.tp_net_throughput = self.hw_config['net_tier1_bw'] * \
                                     self.hw_config['net_tier1_eff'] * 1024 ** 3
        assert (self.d * self.t <= self.hw_config['net_tier1_size'] or
                self.d * self.t <= self.hw_config['net_tier2_size']), \
                f"d={self.d} x t={self.t} is larger than the " \
                f"network size {self.hw_config['net_tier1_size']} " \
                f"or {self.hw_config['net_tier2_size']}"
        self.dp_net_throughput = self.hw_config['net_tier2_bw'] * \
                                 self.hw_config['net_tier2_eff'] * 1024 ** 3
        if self.d * self.t <= self.hw_config['net_tier1_size']:
            self.dp_net_throughput = self.hw_config['net_tier1_bw'] * \
                                     self.hw_config['net_tier1_eff'] * 1024 ** 3
        assert (self.p * self.d * self.t <= self.hw_config['net_tier1_size'] or
                self.p * self.d * self.t <= self.hw_config['net_tier2_size']), \
                f"p={self.p} x d={self.d} x t={self.t} is larger than the " \
                f"network size {self.hw_config['net_tier1_size']} " \
                f"or {self.hw_config['net_tier2_size']}"
        self.pp_net_throughput = self.hw_config['net_tier2_bw'] * \
                                 self.hw_config['net_tier2_eff'] * 1024 ** 3
        if self.p * self.d * self.t < self.hw_config['net_tier1_size']:
            self.pp_net_throughput = self.hw_config['net_tier1_bw'] * \
                                     self.hw_config['net_tier1_eff'] * 1024 ** 3

    def _compute_minibatch_stats(self):
        print("vector_throughput:", self._human_format(self.vector_throughput, 'throughput'))
        print("matrix_throughput:", self._human_format(self.matrix_throughput, 'throughput'))
        print("mem_throughput:", self._human_format(self.mem_throughput, 'bandwidth'))
        print("offload_throughput:", self._human_format(self.offload_throughput, 'bandwidth'))
        print("tp_net_throughput:", self._human_format(self.tp_net_throughput, 'bandwidth'))
        print("pp_net_throughput:", self._human_format(self.pp_net_throughput, 'bandwidth'))
        print("dp_net_throughput:", self._human_format(self.dp_net_throughput, 'bandwidth'))
        for layer in self.megatron_block:
            flops_throughput = self.vector_throughput
            if isinstance(layer, Linear):
                flops_throughput = self.matrix_throughput
            # Add flops/bytes/times per layer
            self.minibatch_fw_flops += layer.get_fw_flops()
            self.minibatch_fw_flops_time += \
                self.minibatch_fw_flops /flops_throughput
            self.minibatch_fw_mem_accessed = layer.get_fw_mem_accessed()
            self.minibatch_fw_mem_time = \
                self.minibatch_fw_mem_accessed / self.mem_throughput
            self.minibatch_bw_flops += layer.get_bw_flops()
            self.minibatch_bw_flops_time += \
                self.minibatch_bw_flops / flops_throughput
            self.minibatch_bw_mem_accessed = layer.get_bw_mem_accessed()
            self.minibatch_bw_mem_time = \
                self.minibatch_bw_mem_accessed / self.mem_throughput
            self.minibatch_recompute_time = layer.get_recompute_flag() * (
                self.minibatch_fw_flops_time + self.minibatch_fw_mem_time)
            self.minibatch_recompute_mem_saving = layer.get_recompute_flag() * (
                layer.get_activation())
            self.gpu_weight_space += layer.get_weight()
            self.gpu_act_space += layer.get_activation()
            self.gpu_weight_grad_space += layer.get_weight_grad()
            self.gpu_act_grad_space += layer.get_activation_grad()
            self.gpu_optim_space += layer.get_optim()
            print(layer.name, 'FW flops:', self._human_format(layer.get_fw_flops(), 'flops'))
            print(layer.name, 'FW flops time:', self.minibatch_fw_flops_time)
            print(layer.name, 'FW mem:', self._human_format(layer.get_fw_mem_accessed(), 'bytes'))
            print(layer.name, 'FW mem time:', self.minibatch_fw_mem_time)
            print(layer.name, 'BW flops:', self._human_format(layer.get_bw_flops(), 'flops'))
            print(layer.name, 'BW flops time:', self.minibatch_bw_flops_time)
            print(layer.name, 'BW mem:', self._human_format(layer.get_bw_mem_accessed(), 'bytes'))
            print(layer.name, 'BW mem time:', self.minibatch_bw_mem_time)
            print(layer.name, 'Recompute time:', self.minibatch_recompute_time)
            print(layer.name, 'Recompute mem saving:', self._human_format(self.minibatch_recompute_mem_saving, 'bytes'))
            print(layer.name, 'Weight:', self._human_format(layer.get_weight(), 'bytes'))
            print(layer.name, 'Act:', self._human_format(layer.get_activation(), 'bytes'))
            print(layer.name, 'Weight grad:', self._human_format(layer.get_weight_grad(), 'bytes'))
            print(layer.name, 'Act grad:', self._human_format(layer.get_activation_grad(), 'bytes'))
            print(layer.name, 'Optim:', self._human_format(layer.get_optim(), 'bytes'))
            print(layer.name, 'Incremental Weight:', self._human_format(self.gpu_weight_space, 'bytes'))
            print(layer.name, 'Incremental Act:', self._human_format(self.gpu_act_space, 'bytes'))
            print(layer.name, 'Incremental Weight grad:', self._human_format(self.gpu_weight_grad_space, 'bytes'))
            print(layer.name, 'Incremental Act grad:', self._human_format(self.gpu_act_grad_space, 'bytes'))
            print(layer.name, 'Incremental Optim:', self._human_format(self.gpu_optim_space, 'bytes'))
        if self.t > 1:
            if self.seq_par or self.point_to_point_rs_ag:
                self.minibatch_fw_tp_size = 2*2 * self.bytes_per_element * \
                    self.seq_par_activation_size
            else:
                self.minibatch_fw_tp_size = 2*2 * self.bytes_per_element * \
                    self.activation_size
                if self.sharp:
                    self.minibatch_fw_tp_size /= 2
        self.minibatch_fw_tp_time = \
            self.minibatch_fw_tp_size / self.tp_net_throughput
        if self.training:
            self.minibatch_bw_tp_size = self.minibatch_fw_tp_size
            self.minibatch_bw_tp_time = self.minibatch_fw_tp_time
        self.minibatch_fw_pp_size = self.pipeline_interleaving
        if self.point_to_point_rs_ag:
            self.minibatch_fw_pp_size *= \
                self.bytes_per_element * self.seq_par_activation_size
        else:
            self.minibatch_fw_pp_size *= \
                self.bytes_per_element * self.activation_size
        self.minibatch_fw_pp_time = \
            self.minibatch_fw_pp_size / self.pp_net_throughput
        if self.training:
            self.minibatch_bw_pp_size = self.minibatch_fw_pp_size
            self.minibatch_bw_pp_time = self.minibatch_fw_pp_time

    def _compute_batch_stats(self):
        # compute/memory stats
        self.gpu_fw_flops = self.layers_per_gpu * self.num_minibatches *\
                            self.minibatch_fw_flops
        self.gpu_fw_flops_time = self.layers_per_gpu * self.num_minibatches *\
                                 self.minibatch_fw_flops_time
        self.gpu_fw_mem_accessed = self.layers_per_gpu * self.num_minibatches *\
                                   self.minibatch_fw_mem_accessed
        self.gpu_fw_mem_time = self.layers_per_gpu * self.num_minibatches *\
                               self.minibatch_fw_mem_time
        self.gpu_bw_flops = self.layers_per_gpu * self.num_minibatches *\
                            self.minibatch_bw_flops
        self.gpu_bw_flops_time = self.layers_per_gpu * self.num_minibatches *\
                                 self.minibatch_bw_flops_time
        self.gpu_bw_mem_accessed = self.layers_per_gpu * self.num_minibatches *\
                                   self.minibatch_bw_mem_accessed
        self.gpu_bw_mem_time = self.layers_per_gpu * self.num_minibatches *\
                               self.minibatch_bw_mem_time
        self.gpu_recompute_time = self.layers_per_gpu * self.num_minibatches *\
                                  self.minibatch_recompute_time
        # network stats
        self.gpu_tp_comm_size = self.layers_per_gpu * self.num_minibatches * (
            self.minibatch_fw_tp_size + self.minibatch_bw_tp_size)
        self.gpu_tp_comm_time = self.layers_per_gpu * self.num_minibatches * (
            self.minibatch_fw_tp_time + self.minibatch_bw_tp_time)
        self.gpu_pp_comm_size = \
            self.num_minibatches * self.pipeline_interleaving * (
                self.minibatch_fw_pp_size + self.minibatch_bw_pp_size)
        self.gpu_pp_comm_time = self.num_minibatches * (
            self.minibatch_fw_pp_time + self.minibatch_bw_pp_time)
        self.gpu_bubble_time = (self.p - 1) * (
            self.layers_per_gpu / self.pipeline_interleaving * (
                self.minibatch_fw_flops_time + self.minibatch_fw_mem_time +
                self.minibatch_bw_flops_time + self.minibatch_bw_mem_time +
                self.minibatch_recompute_time +
                self.minibatch_fw_tp_time + self.minibatch_bw_tp_time) +
                self.minibatch_fw_pp_time + self.minibatch_bw_pp_time)
        self.gpu_dp_comm_size = 2 * self.gpu_weight_space
        self.gpu_dp_comm_time = self.gpu_dp_comm_size / self.dp_net_throughput
        if self.sharp and not self.optim_sharding:
            self.gpu_dp_comm_time /= 2
        if self.dp_overlap:
            exposed_time = (self.p - 1) * max(
                0, self.gpu_dp_comm_size / self.layers_per_gpu - (
                    self.minibatch_bw_flops_time + \
                    self.minibatch_bw_mem_time) * self.pipeline_interleaving)
            self.gpu_dp_comm_size = \
                self.gpu_dp_comm_size / self.layers_per_gpu + exposed_time
        # memory capacity stats
        self.gpu_weight_space *= self.layers_per_gpu
        # account for activation recomputation
        if self.act_recompute != "full":
            if self.act_recompute == "partial":
                self.gpu_act_space += self.layers_per_gpu *\
                                      self.minibatch_recompute_mem_saving
            else:
                self.gpu_act_space *= self.layers_per_gpu
        # Only need activation grads for a single layer
        self.gpu_act_grad_space = self.gpu_act_grad_space
        # Can utilize optimizer split optimization 
        self.gpu_weight_grad_space = self.gpu_weight_grad_space *\
                                     self.layers_per_gpu
        self.gpu_optim_space = self.gpu_optim_space * self.layers_per_gpu
        if self.optim_sharding:
            self.gpu_weight_grad_space /= self.d
            self.gpu_optim_space /= self.d

    def run(self, hw_config):
        assert self._compiled, "You should first call self.compile(sw_config)"
        # TODO - think about how to implement overlap
        self.hw_config = hw_config
        self._update_hw_throughput()
        self._compute_minibatch_stats()
        self._compute_batch_stats()
        # TODO def _compute_offload_requirements(self):
        # TODO incorporate 'weight_offload' and 'activations_offload'/'optimizer_offload'
        self._executed = True
        # or make a big ass dict, or csv, or pandas?

    def get_fw_time(self):
        return self.gpu_fw_flops_time + self.gpu_fw_mem_time

    def get_bw_time(self):
        if self.training:
            return self.gpu_bw_flops_time + self.gpu_bw_mem_time
        else:
            return 0

    def get_recompute_time(self):
        return self.gpu_recompute_time

    def get_bubble_time(self):
        return self.gpu_bubble_time

    def get_tp_comm_time(self):
        return self.gpu_tp_comm_time

    def get_pp_comm_time(self):
        return self.gpu_pp_comm_time

    def get_dp_comm_time(self):
        return self.gpu_dp_comm_time

    def get_total_time(self):
        time = self.get_fw_time()
        time += self.get_bw_time()
        time += self.get_recompute_time()
        time += self.get_bubble_time()
        time += self.get_tp_comm_time()
        time += self.get_pp_comm_time()
        time += self.get_dp_comm_time()
        return time

    def get_useful_flops(self):
        total_flops = sum(
            [layer.get_fw_flops() for layer in self.megatron_block])
        if self.training:
            total_flops += sum(
                [layer.get_bw_flops() for layer in self.megatron_block])
        return total_flops

    def get_compute_efficiency(self):
        total_flops = self.get_useful_flops()
        compute_time = self.get_fw_time() + self.get_bw_time()
        perfect_time = self.num_minibatches * total_flops / (
            self.hw_config['matrix_tflops'] * 1000000000)
        return perfect_time / compute_time

    def get_system_efficiency(self):
        return (self.get_bw_time() + self.get_fw_time()) / self.get_total_time()
        
    def get_total_efficiency(self):
        total_flops = self.get_useful_flops()
        perfect_time = self.num_minibatches * total_flops / (
            self.hw_config['matrix_tflops'] * 1000000000)
        return perfect_time / self.get_total_time()

    def get_gpu_weight_space(self):
        return self.gpu_weight_space

    def get_gpu_act_space(self):
        return self.gpu_act_space
    
    def get_gpu_act_checkpoint_size(self):
        return self.bytes_per_element * self.activation_size * \
            self.layers_per_gpu

    def get_gpu_weight_grad_space(self):
        return self.gpu_weight_grad_space

    def get_gpu_act_grad_space(self):
        return self.gpu_act_grad_space

    def get_gpu_optim_space(self):
        return self.gpu_optim_space
    
    def get_gpu_mem_requirements(self):
        mem = self.get_gpu_weight_space() + \
            self.get_gpu_act_space() + \
            self.get_gpu_act_checkpoint_size() + \
            self.get_gpu_weight_grad_space() + \
            self.get_gpu_act_grad_space() + \
            self.get_gpu_optim_space()
        return mem

    # TODO ===============================================================
    def get_gpu_mem_cap_req(self):
        return self.gpu_mem_cap_req

    def get_offload_mem_bw_req(self):
        return self.offload_mem_bw_req
    # ====================================================================

    def get_total_weight_space(self):
        return self.num_gpus * self.get_gpu_weight_space()

    def get_total_act_space(self):
        return self.num_gpus * self.get_gpu_act_space()
    
    def get_total_act_checkpoint_size(self):
        return self.num_gpus * self.get_gpu_act_checkpoint_size()

    def get_total_weight_grad_space(self):
        return self.num_gpus * self.get_gpu_weight_grad_space()

    def get_total_act_grad_space(self):
        return self.num_gpus * self.get_gpu_act_grad_space()

    def get_total_optim_space(self):
        return self.num_gpus * self.get_gpu_optim_space()

    @staticmethod
    def _human_format(value, v_type):
        step = 1
        suffix = ''
        if v_type == 'bytes':
            step = 1024
            suffix = 'B'
        elif v_type == 'bandwidth':
            step = 1024
            suffix = 'B/s'
        elif v_type == 'flops':
            step = 1000
            suffix = 'OP'
        elif v_type == 'throughput':
            step = 1000
            suffix = 'OP/s'
        else:
            raise ValueError(
                "Type value should be 'bytes' or 'flops' or 'bandwidth' or 'throughput', given {}".format(v_type))
        labels = ['', 'K', 'M', 'G', 'T', 'P', 'E']
        index = 0
        for l in labels:
            if value >= step:
                value /= step
                index += 1
            else:
                break
        return "{0:.2f} {1}{2}".format(value, labels[index], suffix)

    def display_stats(self):
        stats = f"Model {self.name}: {self.num_layers} layers, " \
                f"hidden={self.hidden}, num attn heads: {self.attn_heads}\n" \
                f"Run on {self.num_gpus} GPUs with TP={self.t}, PP={self.p}, " \
                f"DP={self.d}, {self.layers_per_gpu} layers per GPU\n" \
                f"SW config: {self.sw_config};\n" \
                f"HW config: {self.hw_config};\n" \
                f"Weights: {self._human_format(self.get_gpu_weight_space(), 'bytes')};\n" \
                f"Act: {self._human_format(self.get_gpu_act_space(), 'bytes')};\n" \
                f"Act CP: {self._human_format(self.get_gpu_act_checkpoint_size(), 'bytes')};\n" \
                f"Act grad: {self._human_format(self.get_gpu_act_grad_space(), 'bytes')};\n" \
                f"Weight grad: {self._human_format(self.get_gpu_weight_grad_space(), 'bytes')};\n" \
                f"Optim space: {self._human_format(self.get_gpu_optim_space(), 'bytes')};\n" \
                f"Total mem requirements: {self._human_format(self.get_gpu_mem_requirements(), 'bytes')};\n" \
                f"Batch FW time: {self.get_fw_time():.2f};\n" \
                f"Batch BW time: {self.get_bw_time():.2f};\n" \
                f"Batch recompuet time: {self.get_recompute_time():.2f};\n" \
                f"Batch bubble time: {self.get_bubble_time():.2f};\n" \
                f"Batch TP comm time: {self.get_tp_comm_time():.2f};\n" \
                f"Batch PP comm time: {self.get_pp_comm_time():.2f};\n" \
                f"Batch DP comm time: {self.get_dp_comm_time():.2f};\n" \
                f"Batch total time: {self.get_total_time():.2f};\n" \
                f"Total Flops: {self._human_format(self.get_useful_flops(), 'flops')};\n" \
                f"Compute efficiency: {self.get_compute_efficiency()*100:.2f}%;\n" \
                f"System eficiency: {self.get_system_efficiency()*100:.2f}%;\n" \
                f"Total efficiency: {self.get_total_efficiency()*100:.2f}%;\n"
        print(stats)