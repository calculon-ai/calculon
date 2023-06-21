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
import os
import tempfile
import unittest


class JsonWriteReadTestCase(unittest.TestCase):
  def test_json_read_write(self):
    jd = {
      'a': 1239,
      'hi': {
        '34': 'world',
        'ugh': 77,
        '1': 'hello world world world world world world world world world world'
      }
    }

    _, reg_file = tempfile.mkstemp(suffix='.json')
    _, gz_file = tempfile.mkstemp(suffix='.json.gz')
    _, foo_file = tempfile.mkstemp(suffix='.json.foo')
    _, bar_file = tempfile.mkstemp(suffix='.bar.gz')
    os.remove(reg_file)
    os.remove(gz_file)
    os.remove(foo_file)
    os.remove(bar_file)

    self.assertTrue(calculon.is_json_extension(reg_file))
    self.assertTrue(calculon.is_json_extension(gz_file))
    self.assertFalse(calculon.is_json_extension(foo_file))
    self.assertFalse(calculon.is_json_extension(bar_file))

    self.assertFalse(os.path.exists(reg_file))
    self.assertFalse(os.path.exists(gz_file))

    calculon.io.write_json_file(jd, reg_file)
    calculon.io.write_json_file(jd, gz_file)

    self.assertTrue(os.path.exists(reg_file))
    self.assertTrue(os.path.exists(gz_file))

    reg_size = os.path.getsize(reg_file)
    gz_size = os.path.getsize(gz_file)
    self.assertTrue(reg_size > 0)
    self.assertTrue(reg_size > gz_size)
    self.assertTrue(gz_size > 0)

    reg_jd = calculon.io.read_json_file(reg_file)
    gz_jd = calculon.io.read_json_file(gz_file)

    self.assertEqual(reg_jd, jd)
    self.assertEqual(gz_jd, jd)

    os.remove(reg_file)
    os.remove(gz_file)
