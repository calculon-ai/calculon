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

import logging
import math
import os

import calculon
from calculon.util import pick
from calculon.llm import *


class Validation(calculon.CommandLine):
  NAME = 'llm-validation'
  ALIASES = ['lv']

  @staticmethod
  def create_parser(subparser):
    sp = subparser.add_parser(
      Validation.NAME, aliases=Validation.ALIASES,
      help='run a validation of llm execution')
    sp.set_defaults(func=Validation.run_command)
    sp.add_argument('-b', '--base_dir', default='.',
                    help='Base directory')
    sp.add_argument('-v', '--verbose', action='store_true',
                    help='Show verbose output while running')

  @staticmethod
  def run_command(logger, args):
    funcs = [
      Validation.seqsel_fig1,
      Validation.seqsel_fig7,
      Validation.seqsel_tab5
    ]
    for func in funcs:
      if args.verbose:
        print(f'\n\nNow running test: {func.__name__}')
      if func(logger, args) is not None:
        return -1

  @staticmethod
  def seqsel_fig1(logger, args):
    kModels = ['megatron-22B', 'gpt3-175B', 'turing-530B', 'megatron-1T']
    kModes = ['none', 'seqsel']
    # These profiled values are reported here:
    # https://arxiv.org/pdf/2205.05198.pdf
    # Figure 1
    kProfile = {
      'megatron-22B': {
        'none': {
          'par_opt': 45.5625,
          'act': 59.25
        },
        'seqsel': {
          'par_opt': 45.5625,
          'act': 9.5625
        }
      },
      'gpt3-175B': {
        'none': {
          'par_opt': 45.5625,
          'act': 66.84375
        },
        'seqsel': {
          'par_opt': 45.5625,
          'act': 12.3515625
        }
      },
      'turing-530B': {
        'none': {
          'par_opt': 31.640625,
          'act': 114.0234375
        },
        'seqsel': {
          'par_opt': 31.640625,
          'act': 23.076171875
        }
      },
      'megatron-1T': {
        'none': {
          'par_opt': 32.958984375,
          'act': 131.25
        },
        'seqsel': {
          'par_opt': 32.958984375,
          'act': 26.5625
        }
      }
    }

    def get_files(model, mode):
      assert model in kModels
      assert mode in kModes
      app = os.path.join(args.base_dir, 'models', f'{model}.json')
      exe = os.path.join(args.base_dir, 'validation', 'seqsel', 'fig1',
                         f'{model}_{mode}.json')
      return app, exe

    def get_profile(model, mode):
      assert model in kModels
      assert mode in kModes
      return kProfile[model][mode]

    syst_file = os.path.join(args.base_dir, 'systems', 'a100_80e.json')
    syst = System(calculon.io.read_json_file(syst_file))
    data = {}
    for model in kModels:
      data[model] = {}
      for mode in kModes:
        if args.verbose:
          print(f'Analyzing {model} {mode}')
        data[model][mode] = {}
        app_file, exe_file = get_files(model, mode)
        app = Llm.Application(calculon.read_json_file(app_file))
        exe = Llm.Execution.from_json(calculon.read_json_file(exe_file))
        mt = Llm(app, logger)
        mt.compile(syst, exe)
        mt.run(syst)
        stats = mt.get_stats_json(False)
        data[model][mode]['profile_gib'] = get_profile(model, mode)
        act_par_opt = (stats['weight_space'] + stats['weight_grad_space'] +
                       stats['optimizer_space']) / (1024**3)
        act_act = stats['act_space'] / (1024**3)
        data[model][mode]['actual_gib'] = {
          'par_opt': act_par_opt,
          'act': act_act
        }

    print('*Params & Opt,|,none,,,|,seqsel,,,')
    print('Model,|,Profile,Calc,Delta,|,Profile,Calc,Delta,')
    max_error = 0
    abs_error = 0
    for model in kModels:
      print(f'{model},', end='')
      for mode in kModes:
        p = data[model][mode]['profile_gib']['par_opt']
        a = data[model][mode]['actual_gib']['par_opt']
        d = 100*(1-a/p)
        if math.fabs(d) > max_error:
          max_error = math.fabs(d)
        abs_error += math.fabs(d)
        print(f'|,{p},{a:.2f},{d:.2f}%,', end='')
      print()
    ave_error = abs_error / (len(kModels) * len(kModes))
    print(f'Ave,,{ave_error:.2f}%')
    print(f'Max,,{max_error:.2f}%')
    print(',')

    print('*Activations,|,none,,,|,seqsel,,,')
    print('Model,|,Profile,Calc,Delta,|,Profile,Calc,Delta,')
    max_error = 0
    abs_error = 0
    for model in kModels:
      print(f'{model},', end='')
      for mode in kModes:
        p = data[model][mode]['profile_gib']['act']
        a = data[model][mode]['actual_gib']['act']
        d = 100*(1-a/p)
        if math.fabs(d) > max_error:
          max_error = math.fabs(d)
        abs_error += math.fabs(d)
        print(f'|,{p},{a:.2f},{d:.2f}%,', end='')
      print()
    ave_error = abs_error / (len(kModels) * len(kModes))
    print(f'Ave,,{ave_error:.2f}%')
    print(f'Max,,{max_error:.2f}%')
    print(',')

  @staticmethod
  def seqsel_fig7(logger, args):
    kModels = ['megatron-22B', 'gpt3-175B', 'turing-530B', 'megatron-1T']
    kModes = ['none', 'seq', 'sel', 'seqsel', 'full']
    # These profiled values are reported here:
    # https://arxiv.org/pdf/2205.05198.pdf
    # Figure 7
    kProfile = {
      'megatron-22B': {
        'none': 100.00,
        'seq': 66.84,
        'sel': 49.42,
        'seqsel': 16.18,
        'full': 7.64
      },
      'gpt3-175B': {
        'none': 100.00,
        'seq': 62.04,
        'sel': 56.53,
        'seqsel': 18.49,
        'full': 8.71
      },
      'turing-530B': {
        'none': 100.00,
        'seq': 58.31,
        'sel': 62.04,
        'seqsel': 20.27,
        'full': 9.42
      },
      'megatron-1T': {
        'none': 100.00,
        'seq': 58.31,
        'sel': 62.04,
        'seqsel': 20.27,
        'full': 9.42
      }
    }

    def get_files(model, mode):
      assert model in kModels
      assert mode in kModes
      app = os.path.join(args.base_dir, 'models', f'{model}.json')
      exe = os.path.join(args.base_dir, 'validation', 'seqsel', 'fig7',
                         f'{model}_{mode}.json')
      return app, exe

    def get_profile(model, mode):
      assert model in kModels
      assert mode in kModes
      return kProfile[model][mode]

    syst_file = os.path.join(args.base_dir, 'systems', 'a100_80e.json')
    syst = System(calculon.io.read_json_file(syst_file))
    raw = {}
    for model in kModels:
      raw[model] = {}
      for mode in kModes:
        if args.verbose:
          print(f'Analyzing {model} {mode}')
        raw[model][mode] = {}
        app_file, exe_file = get_files(model, mode)
        app = Llm.Application(calculon.read_json_file(app_file))
        exe = Llm.Execution.from_json(calculon.read_json_file(exe_file))
        mt = Llm(app, logger)
        mt.compile(syst, exe)
        mt.run(syst)
        stats = mt.get_stats_json(False)
        raw[model][mode] = stats['act_space'] + stats['act_checkpoint_size']

    rel = {}
    for model in kModels:
      rel[model] = {}
      for mode in kModes:
        rel[model][mode] = {}
        rel[model][mode] = raw[model][mode] / raw[model]['none'] * 100

    print('Activations,|,none,,,|,seq,,,|,sel,,,|,seqsel,,,|,full,,,')
    print('Model,|,Profile,Calc,Delta,|,Profile,Calc,Delta,|'
          ',Profile,Calc,Delta,|,Profile,Calc,Delta,|,Profile,Calc,Delta,')
    max_error = 0
    abs_error = 0
    for model in kModels:
      print(f'{model},', end='')
      for mode in kModes:
        p = get_profile(model, mode)
        a = rel[model][mode]
        d = 100*(1-a/p)
        if math.fabs(d) > max_error:
          max_error = math.fabs(d)
        abs_error += math.fabs(d)
        print(f'|,{p}%,{a:.2f}%,{d:.2f}%,', end='')
      print()
    ave_error = abs_error / (len(kModels) * len(kModes))
    print(f'Ave,,{ave_error:.2f}%')
    print(f'Max,,{max_error:.2f}%')
    print(',')

  @staticmethod
  def seqsel_tab5(logger, args):
    kModels = ['megatron-22B', 'gpt3-175B', 'turing-530B', 'megatron-1T']
    kModes = ['full', 'seqsel']
    # These profiled values are reported here:
    # https://arxiv.org/pdf/2205.05198.pdf
    # Table 5
    kProfile = {
      'megatron-22B': {
        'full': 1.42,
        'seqsel': 1.10
      },
      'gpt3-175B': {
        'full': 18.13,
        'seqsel': 13.75
      },
      'turing-530B': {
        'full': 49.05,
        'seqsel': 37.83
      },
      'megatron-1T': {
        'full': 94.42,
        'seqsel': 71.49
      }
    }

    def get_files(model, mode):
      assert model in kModels
      assert mode in kModes
      app = os.path.join(args.base_dir, 'models', f'{model}.json')
      exe = os.path.join(args.base_dir, 'validation', 'seqsel', 'tab5',
                         f'{model}_{mode}.json')
      return app, exe

    def get_profile(model, mode):
      assert model in kModels
      assert mode in kModes
      return kProfile[model][mode]

    syst_file = os.path.join(args.base_dir, 'systems', 'a100_80g.json')
    syst = System(calculon.io.read_json_file(syst_file))
    data = {}
    for model in kModels:
      data[model] = {}
      for mode in kModes:
        if args.verbose:
          print(f'Analyzing {model} {mode}')
        data[model][mode] = {}
        app_file, exe_file = get_files(model, mode)
        app = Llm.Application(calculon.read_json_file(app_file))
        exe = Llm.Execution.from_json(calculon.read_json_file(exe_file))
        mt = Llm(app, logger)
        mt.compile(syst, exe)
        mt.run(syst)
        stats = mt.get_stats_json(False)
        data[model][mode]['profile_time'] = get_profile(model, mode)
        data[model][mode]['actual_time'] = stats["total_time"]
        data[model][mode]['memory_req'] = stats["proc_mem_tier1_cap_req"]

    print('End-to-end,|,full,,,,|,seqsel,,,,')
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
    ave_error = abs_error / (len(kModels) * len(kModes))
    print(f'Ave,,{ave_error:.2f}%')
    print(f'Max,,{max_error:.2f}%')
    print(',')

calculon.CommandLine.register(Validation)
