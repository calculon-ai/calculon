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

import calculon
from calculon.llm import *

class Runner(calculon.CommandLine):
  NAME = 'llm'
  ALIASES = []

  @staticmethod
  def create_parser(subparser):
    sp = subparser.add_parser(Runner.NAME, aliases=Runner.ALIASES,
                              help='run a single llm calculation')
    sp.set_defaults(func=Runner.run_command)
    sp.add_argument('application', type=str,
                    help='File path to application configuration')
    sp.add_argument('execution', type=str,
                    help='File path to execution configuration')
    sp.add_argument('system', type=str,
                    help='File path to system configuration')
    sp.add_argument('stats', type=str,
                    help='File path to stats output ("-" for stdout")')
    sp.add_argument('-p', '--peers', type=str, default=None,
                    help='File path to write out peers file')
    sp.add_argument('-l', '--layers', action='store_true',
                    help='Include layers information in output stats file')

  @staticmethod
  def run_command(logger, args):
    app_json = calculon.io.read_json_file(args.application)
    exe_json = calculon.io.read_json_file(args.execution)
    sys_json = calculon.io.read_json_file(args.system)

    app = Llm.Application(app_json)
    exe = Llm.Execution.from_json(exe_json)
    syst = System(sys_json)

    try:
      model = Llm(app, logger)
      model.compile(syst, exe)
      model.run(syst)
    except Llm.Error as error:
      print(f'ERROR: {error}')
      return -1

    if args.stats == '-':
      model.display_stats()
    elif calculon.is_json_extension(args.stats):
      calculon.write_json_file(model.get_stats_json(args.layers), args.stats)
    else:
      assert False, f'unknown stats extension: {args.stats}'

    if args.peers:
      calculon.write_json_file(exe.get_peers_json(), args.peers)

    return 0


calculon.CommandLine.register(Runner)
