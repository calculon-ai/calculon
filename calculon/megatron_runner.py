#!/usr/bin/env python3

import argparse
import json
import sys

from layers import *
from megatron import *


def main(args):
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


if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('application', type=str,
                  help='File path to application configuration')
  ap.add_argument('execution', type=str,
                  help='File path to execution configuration')
  ap.add_argument('system', type=str,
                  help='File path to system configuration')
  args = ap.parse_args()
  sys.exit(main(args))
