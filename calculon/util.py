"""
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  https://www.apache.org/licenses/LICENSE-2.0
 *
 * See the NOTICE file distributed with this work for additional information
 * regarding copyright ownership.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
"""

import argparse


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
    return "{0:.{1}f} {2}{3}".format(value, precision, labels[index], suffix)
  else:
    return "n/a {1}{2}".format(value, labels[0], suffix)


def pick(en, a, b):
  if en:
    return a
  return b


def arg_true_false_all(arg):
  trues = ['t', 'true', 'T', 'True', '1']
  falses = ['f', 'false', 'F', 'False', '0']
  alls = ['both', 'all', '*']
  if arg in trues:
    return [True]
  elif arg in falses:
    return [False]
  elif arg in alls:
    return [False, True]
  else:
    raise argparse.ArgumentTypeError(f'Invalid true/false/all: {arg}')
