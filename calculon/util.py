def human_format(value, v_type):
  step = 1
  suffix = ''
  if v_type == 'bytes':
    step = 1024
    suffix = 'B'
  elif v_type == 'bandwidth':
    step = 1024
    suffix = 'B/s'
  elif v_type == 'flops':
    step = 1000
    suffix = 'Ops'
  elif v_type == 'throughput':
    step = 1000
    suffix = 'Op/s'
  else:
    raise ValueError(("Type value should be 'bytes' or 'flops' or 'bandwidth' "
                      "or 'throughput', given {}".format(v_type)))
  labels = ['', 'k', 'M', 'G', 'T', 'P', 'E']
  index = 0
  if value != None:
    for l in labels:
      if value >= step:
        value /= step
        index += 1
      else:
        break
    return "{0:.2f} {1}{2}".format(value, labels[index], suffix)
  else:
    return "n/a {1}{2}".format(value, labels[0], suffix)


def pick(en, a, b):
  if en:
    return a
  return b
