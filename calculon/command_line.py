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

import copy

class CommandLine:
  """Defines the abstract interface definition for a command line interface.
  Inspired from: https://github.com/ssnetsim/ssplot/
  """

  @staticmethod
  def create_parser(subparser):
    """
    This function adds a parser to the subparser object according to the
    specific command line interface implementation.
    """
    raise NotImplementedError('subclasses must override this')

  @staticmethod
  def run_command(logger, args):
    """
    This function is used to run the command if it is chosen at the command
    line. This function should be registered to the parser in create_parser().
    """
    raise NotImplementedError('subclasses must override this')

  # this is a mapping of all names (class->names)
  _names = {}

  @staticmethod
  def register(cls):
    # gather names
    primary_name = cls.NAME
    aliases = cls.ALIASES

    # create a set to hold all
    all_names = [primary_name] + aliases

    # check current names against all new names
    for new_name in all_names:
      for pname in CommandLine._names:
        assert new_name is not pname, f'{new_name} already exists'
        for alias in CommandLine._names[pname]:
          assert new_name is not alias, f'{new_name} already exists'

    # add to map
    CommandLine._names[cls] = all_names

  @staticmethod
  def command_lines():
    return set(CommandLine._names.keys())

  @staticmethod
  def all_names():
    return copy.copy(CommandLine._names)
