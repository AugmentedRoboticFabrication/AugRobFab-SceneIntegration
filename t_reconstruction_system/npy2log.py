import numpy as np
import re
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()

    Ts = np.load(args.path)
    with open('trajectory.log', 'w') as f:

        for i in range(len(Ts)):
            f.write('{} {} {}\n'.format(i-1, i, len(Ts)))

            T = Ts[i]
            T[:3, 3] *= 0.001

            s = np.array2string(T)
            s = re.sub('[\[\]]', '', s)
            f.write('{}\n'.format(s))
