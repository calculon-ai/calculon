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

import calculon
from calculon.megatron import *

class Runner(calculon.CommandLine):
  NAME = 'megatron'
  ALIASES = ['mt', 'm']

  @staticmethod
  def create_parser(subparser):
    sp = subparser.add_parser(Runner.NAME, aliases=Runner.ALIASES,
                              help='run a single megatron calculation')
    sp.set_defaults(func=Runner.run_command)
    sp.add_argument('application', type=str,
                    help='File path to application configuration')
    sp.add_argument('execution', type=str,
                    help='File path to execution configuration')
    sp.add_argument('system', type=str,
                    help='File path to system configuration')

  @staticmethod
  def run_command(args):
    with open(args.application, 'r') as fd:
      app_json = json.load(fd)
    with open(args.execution, 'r') as fd:
      exe_json = json.load(fd)
    with open(args.system, 'r') as fd:
      sys_json = json.load(fd)

    model = Megatron(Megatron.Application(app_json))
    model.compile(Megatron.Execution(exe_json))
    model.run(System(sys_json))
    model.display_stats()


calculon.CommandLine.register(Runner)
