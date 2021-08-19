#!/usr/bin/env python

import os
import sys
import argparse
import time

import numpy as np
import time
import scipy
import scipy.linalg

def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--emb_file', help="Input embedding file affix")
    parser.add_argument('--output_file', help="Output model file path",
                        default='output')
    parser.add_argument('--output_file_reverse', help="Output model file path",
                        default='output')
    parser.add_argument('--orthogonal', action='store_true', \
                        help='use orthogonal constrained mapping')

    args = parser.parse_args(arguments)
    
    # read aligned embeddings
    suffix = ['.src', '.trg']
    embeds = [None, None]

    for j in [0, 1]:
        embeds[j] = np.loadtxt(args.emb_file+suffix[j], delimiter=' ')
    
    x, z = embeds[0], embeds[1] # EN -> Non-EN
    print(x.shape, z.shape)

    xp = np

    # learn the mapping w
    # x.dot(w) \approx z
    if args.orthogonal:  # orthogonal mapping
        u, s, vt = xp.linalg.svd(z.T.dot(x))
        w = vt.T.dot(u.T)
    else:  # unconstrained mapping
        x_pseudoinv = xp.linalg.inv(x.T.dot(x)).dot(x.T)
        w = x_pseudoinv.dot(z)

    np.savetxt(args.output_file+'.BAM.map', w, delimiter=' ', fmt='%0.6f')
    
    u, sigma, v = xp.linalg.svd(x - z)  
    v_b = v[0]
    np.savetxt(args.output_file+'.GBDD.map', v_b, delimiter=' ', fmt='%0.6f')
    
    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
