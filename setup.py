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

import codecs
import re
import os
import sys

try:
  from setuptools import setup
except:
  print('please install setuptools via pip:')
  print('  pip3 install setuptools')
  sys.exit(-1)

def find_version(*file_paths):
  version_file = codecs.open(os.path.join(os.path.abspath(
    os.path.dirname(__file__)), *file_paths), 'r').read()
  version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                            version_file, re.M)
  if version_match:
    return version_match.group(1)
  raise RuntimeError("Unable to find version string.")


setup(
  name='calculon',
  version=find_version('calculon', '__init__.py'),
  description='Co-design for large scale parallel applications',
  author='Michael Isaev',
  author_email='michael.v.isaev@gmail.com',
  license='Apache 2',
  url='http://github.com/calculon-ai/calculon',
  packages=['calculon', 'calculon.llm'],
  scripts=['bin/calculon'],
  install_requires=[],
)
