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

import json
import logging
import math

import calculon
from calculon.util import pick
from calculon.megatron import *

kModels = ['22B', '175B', '530B', '1T']
kModes = ['full', 'seqsel']
# These profiled values are reported here:
# https://arxiv.org/pdf/2205.05198.pdf
kProfile = {
  '22B': {
    'full': 1.42,
    'seqsel': 1.10
  },
  '175B': {
    'full': 18.13,
    'seqsel': 13.75
  },
  '530B': {
    'full': 49.05,
    'seqsel': 37.83
  },
  '1T': {
    'full': 94.42,
    'seqsel': 71.49
  }
}

def get_files(model, mode):
  assert model in kModels
  assert mode in kModes
  app = f'examples/{model}.json'
  exe = f'validation/{model}_{mode}.json'
  return app, exe


def get_profile(model, mode):
  assert model in kModels
  assert mode in kModes
  return kProfile[model][mode]


class Validation(calculon.CommandLine):
  NAME = 'megatron-validation'
  ALIASES = ['mv']

  @staticmethod
  def create_parser(subparser):
    sp = subparser.add_parser(
      Validation.NAME, aliases=Validation.ALIASES,
      help='run a validation of megatron execution')
    sp.set_defaults(func=Validation.run_command)

  @staticmethod
  def run_command(logger, args):
    syst_file = f'examples/a100.json'
    with open(syst_file, 'r') as fd:
      syst = System(json.load(fd))
    data = {}
    for model in kModels:
      data[model] = {}
      for mode in kModes:
        data[model][mode] = {}
        app_file, exe_file = get_files(model, mode)
        with open(app_file, 'r') as fd:
          app = Megatron.Application(json.load(fd))
        with open(exe_file, 'r') as fd:
          exe = Megatron.Execution(json.load(fd))
        mt = Megatron(app, logger)
        mt.compile(exe)
        mt.run(syst)
        stats = mt.get_stats_json()
        data[model][mode]['profile_time'] = get_profile(model, mode)
        data[model][mode]['actual_time'] = stats["total_time"]
        data[model][mode]['memory_req'] = stats["proc_mem_tier1_cap_req"]

    print(',|,full,,,,|,seqsel,,,,')
    print('Model,|,Profile,Calc,Delta,GiB,|,Profile,Calc,Delta,GiB,')
    max_error = 0
    abs_error = 0
    for model in kModels:
      print(f'{model},', end='')
      for mode in kModes:
        p = data[model][mode]['profile_time']
        a = data[model][mode]['actual_time']
        d = 100*(1-a/p)
        if math.fabs(d) > max_error:
          max_error = math.fabs(d)
        abs_error += math.fabs(d)
        m = data[model][mode]['memory_req'] / (1024**3)
        print(f'|,{p},{a:.2f},{d:.2f}%,{m:.2f},', end='')
      print()
    print(',')
    ave_error = abs_error / (len(kModels) * len(kModes))
    print(f'Ave,,{ave_error:.2f}%')
    print(f'Max,,{max_error:.2f}%')

    return 0

calculon.CommandLine.register(Validation)
