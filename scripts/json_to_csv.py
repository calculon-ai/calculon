#!/usr/bin/env python3

import argparse
import calculon
import gzip
import json
import sys


def main(args):
  j = calculon.read_json_file(args.json_file)

  header_entries = []
  for category in j['0']:
    for key in j['0'][category]:
      header_entries.append((category, key))

  opener = gzip.open if args.csv_file.endswith('.gz') else open
  with opener(args.csv_file, 'wb') as fd:
    # Header
    fd.write(bytes(',', 'utf-8'))
    for _, key in header_entries:
      fd.write(bytes(f'{key},', 'utf-8'))
    fd.write(bytes(',\n', 'utf-8'))

    # Rows
    for entry in j.keys():
      fd.write(bytes(f'{entry},', 'utf-8'))
      for category, key in header_entries:
        v = j[entry][category][key]
        fd.write(bytes(f'{v},', 'utf-8'))
      fd.write(bytes(',\n', 'utf-8'))

if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('json_file', help='input JSON file')
  ap.add_argument('csv_file', help='output CSV file')
  sys.exit(main(ap.parse_args()))
