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

class Version(calculon.CommandLine):
  NAME = 'version'
  ALIASES = ['v']

  @staticmethod
  def create_parser(subparser):
    sp = subparser.add_parser(Version.NAME, aliases=Version.ALIASES,
                              help='show the version and exit')
    sp.set_defaults(func=Version.run_command)

  @staticmethod
  def run_command(logger, args):
    # version is specified in __init__.py
    logger.info(calculon.__version__)


calculon.CommandLine.register(Version)
