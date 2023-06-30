#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tol_colors as tc

########## Utils ##########
def transformer_attn_size(hidden, layers, attn_size_step=32):
  return step_rounder(hidden / layers, attn_size_step)

def transformer_num_parameters(hidden, layers, attn_size_step=32):
  attn_heads = layers
  attn_size = transformer_attn_size(hidden, layers, attn_size_step)
  mlp_params = 8 * layers * hidden **2
  attn_params = 4 * layers * hidden * attn_heads * attn_size
  return mlp_params + attn_params
  #return 12 * layers * hidden **2

def transformer_t_params(hidden, layers):
  return transformer_num_parameters(hidden, layers) / 10**12

def step_rounder(layer, step=1):
  return np.round(layer/step) * step

def model_ratio(hidden, layers):
  return hidden / layers

def human_format(value, v_type='base10', precision=3):
  step = 1
  suffix = ''
  if v_type == 'base10':
    step = 1000
    suffix = ''
  elif v_type == 'base2':
    step = 1024
    suffix = ''
  elif v_type == 'bytes':
    step = 1024
    suffix = 'iB'
  elif v_type == 'bandwidth':
    step = 1000
    suffix = 'B/s'
  elif v_type == 'flops':
    step = 1000
    suffix = 'Ops'
  elif v_type == 'throughput':
    step = 1000
    suffix = 'Op/s'
  else:
    raise ValueError(
      f"Type value should be 'base10', 'base2', 'bytes', 'flops', "
      f"'bandwidth', or 'throughput'. You gave {v_type}")
  labels = ['', 'k', 'M', 'G', 'T', 'P', 'E']
  index = 0
  if value != None:
    abs_value = abs(value)
    if value >= 0:
      sign = 1
    else:
      sign = -1
    for l in labels:
      if abs_value >= step:
        abs_value /= step
        index += 1
      else:
        break
    value = sign * abs_value
    return "{0:.{1}f}{2}{3}".format(value, precision, labels[index], suffix)
  else:
    return "n/a {1}{2}".format(value, labels[0], suffix)

########## Scale rules with ratio ##########
def ratio_layer_scale(hidden, ratio=128, step=4):
  return step_rounder(hidden/ratio, step=step)
def ratio_hidden_scale(layers, ratio=128, step=4096):
  return step_rounder(layers * ratio, step=step)
def ratio_param_layer_scale(layers, ratio=128, step=4096):
  return transformer_num_parameters(
    ratio_hidden_scale(layers, ratio=ratio, step=step), layers)
def ratio_param_hidden_scale(hidden, ratio=128, step=4):
  return transformer_num_parameters(
    hidden, ratio_layer_scale(hidden, ratio=ratio, step=step))



hidden_step = 1024
layer_step = 32
hiddens = [x for x in range(24*1024, 8192*24 + 1, hidden_step)]
layers = [x for x in range(128, 576 + 1, layer_step)]
slope = (320-192) / ((512-128)/layer_step)
y_intercept = 192
targets = [slope * x + y_intercept for x in range(len(layers))]
#targets = [200 for x in range(len(layers))]
hiddens = np.asarray(hiddens)
layers = np.asarray(layers)
params_grid = np.zeros((hiddens.shape[0], layers.shape[0]), dtype="float")
ratio_grid = np.zeros((hiddens.shape[0], layers.shape[0]), dtype="float")
target_ratio_grid = np.zeros((hiddens.shape[0], layers.shape[0]), dtype="float")
for row, h in enumerate(hiddens):
  for col, l in enumerate(layers):
    params_grid[row][col] = transformer_num_parameters(h, l)
    ratio = model_ratio(h, l)
    ratio_grid[row][col] = ratio
    target_ratio_grid[row][col] = ratio / targets[col]


fig = plt.figure(figsize=(16, 16), dpi=200)
ax = fig.add_subplot(1, 1, 1)

im = ax.imshow(target_ratio_grid, cmap=tc.tol_cmap('BuRd'),
               vmin=.5, vmax=1.5, origin='lower')#, aspect=0.8)
ax.set_xlabel('# of blocks')
ax.set_ylabel('Hidden size')

# Show all ticks and label them with the respective list entries
ax.set_yticks(np.arange(hiddens.shape[0]))
ax.set_xticks(np.arange(layers.shape[0]))
ax.set_yticklabels(hiddens)
ax.set_xticklabels(layers)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
print('name,hidden,feedforward,seq_size,attn_heads,attn_size,num_blocks,gbs,ratio')
for col, l in enumerate(layers):
  best_val = 9999
  best_row = None
  for row, h in enumerate(hiddens):
    val = abs(target_ratio_grid[row][col] - 1)
    if val < best_val:
      best_val = val
      best_row = row
  for row, h in enumerate(hiddens):
    result = human_format(params_grid[row][col], precision=0)
    result += "\n"
    result += human_format(ratio_grid[row][col], precision=0)
    weight = 'bold' if row == best_row else None
    text = ax.text(col, row, result, ha="center", va="center", color="k", size=8, weight=weight)
    if row == best_row:
      attn_size = int(step_rounder(hiddens[row] / layers[col]))
      params = human_format(transformer_num_parameters(hiddens[row], layers[col]), precision=0)
      ratio = hiddens[row] / layers[col]
      print(f'{params},{hiddens[row]},{hiddens[row]*4},8192,{layers[col]},{attn_size},{layers[col]},3072,{ratio}')

exit(0)
ax.spines[:].set_visible(False)
ax.set_xticks(np.arange(params_grid.shape[1]+1)-.5, minor=True)
ax.set_yticks(np.arange(params_grid.shape[0]+1)-.5, minor=True)
ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
ax.tick_params(which="minor", bottom=False, left=False)
ax.set_title("Number of parameters in trillions, and model ratio, colored by ratio")

fig.tight_layout()
fig.savefig('huge.png')
plt.close(fig)
