import re
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def read_probes():
    sel_line = ''
    with open('p', 'r') as f:
        for line in f:
            if line.strip().startswith('500'):
                sel_line = line.strip()
                break
    
    sel_line = sel_line.replace('-1e+300', '0')
    vals = re.findall(r'(\S+)',sel_line)
    vals.remove('500')

    return np.array([float(x) for x in vals])

def probes_to_array() -> np.ndarray:
    prev_match = None
    with open('p') as f:
        for line in f:
            match = re.search(r".*Probe (\d).*\(.*\).*", line)
            if match is None:
                break
            prev_match = match

    if prev_match is None:
        raise TypeError('Unable to count probes.')

    count = int(re.findall(r".*Probe (\d+).*", prev_match.string)[-1])+1
    print(count)

    a: np.ndarray = np.zeros((count,3))

    i = 0

    with open('p') as f:
        for line in f:
            if re.search(r'.*Probe\s\d+.*\(.*\).*', line):
                match = re.findall(r'.*\(([\-0-9\.]+) ([\-0-9\.]+) ([\-0-9\.]+)\).*', line)[0]
                a[i, 0], a[i, 1] = float(match[0]), float(match[2])
                i += 1
            else:
                break
    a[:, 2] = read_probes()

    return a
    

if __name__ == '__main__':
    probes_to_array()
