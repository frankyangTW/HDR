import numpy as np
import time
from tqdm import tqdm

g1 = 71.15
g2 = 2590.34

h = 540
w = 960

def grey_world(rgb):
    r_avg = np.mean(rgb[:, :, 0])
    g_avg = np.mean(rgb[:, :, 1])
    b_avg = np.mean(rgb[:, :, 2])
    a = g_avg / r_avg
    b = g_avg / b_avg

    return rgb * [a, 1, b]


def merge_raw(filename, h=540, w=960):
    sp1h = np.fromfile(filename + ".sp1h", dtype='uint16').reshape([h, w])
    sp1l = np.fromfile(filename + ".sp1l", dtype='uint16').reshape([h, w])
    sp2 = np.fromfile(filename + ".sp2", dtype='uint16').reshape([h, w])

    h_over = sp1h >= 4000
    l_over = sp1l >= 4000

    s1 = (sp2) * g2 * h_over * l_over + sp1l * g1 * h_over * (1-l_over) + sp1h * (1-h_over) * (1-l_over)

    return s1

def debayer(A):
    start = time.time()
    h, w = A.shape
    R = np.zeros([h, w])
    G = np.zeros([h, w])
    B = np.zeros([h, w])


    R[::2, ::2] += A[::2, ::2]
    R[::2, 1::2] += 0.5 * A[::2, ::2] + 0.5 * np.hstack([A[::2, 2::2], A[::2, -2].reshape(-1, 1)])
    R[1::2, ::2] += 0.5 * A[::2, ::2] + 0.5 * np.vstack([A[2::2, ::2], A[-2, ::2]])

    BR = np.hstack([A[2::2, 2::2], A[2::2, -2].reshape(-1, 1)])
    R[1::2, 1::2] += 0.25 * A[::2, ::2] + 0.25 * np.hstack([A[::2, 2::2], A[::2, -2].reshape(-1, 1)]) \
                    + 0.25 * np.vstack([A[2::2, ::2], A[-2, ::2]]) + 0.25 * np.vstack([BR, BR[-1, :]])

    
    B[1::2, 1::2] += A[1::2, 1::2]
    B[::2, 1::2] += 0.5 * A[1::2, 1::2] + 0.5 * np.vstack([A[1, 1::2], A[1:-1:2, 1::2]])
    B[1::2, ::2] += 0.5 * A[1::2, 1::2] + 0.5 * np.hstack([A[1::2, 1].reshape(-1, 1), A[1::2, 1:-1:2]])

    UL = np.hstack([A[1:-1:2, 1].reshape(-1, 1), A[1:-1:2, 1:-1:2]])
    B[::2, ::2] += 0.25 * A[1::2, 1::2] + 0.25 * np.vstack([A[1, 1::2], A[1:-1:2, 1::2]]) \
                    + 0.25 * np.hstack([A[1::2, 1].reshape(-1, 1), A[1::2, 1:-1:2]]) + 0.25 * np.vstack([UL[0, :], UL])

    G[1::2, ::2] += A[1::2, ::2]
    G[::2, 1::2] += A[::2, 1::2]
    G[1:-1:2, 1:-1:2] += 0.25 * A[:-2:2, 1:-1:2] + 0.25 * A[2::2, 1:-1:2] + 0.25 * A[1:-1:2, :-2:2] + 0.25 * A[1:-1:2, 2::2]
    G[2::2, 2::2] += 0.25 * A[1:-1:2, 2:-1:2] + 0.25 * A[3::2, 2:-1:2] + 0.25 * A[2::2, 1:-1:2] + 0.25 * A[2::2, 3::2]
    
    ## four edges
    G[0, ::2] = A[0, 1::2]
    G[-1, 1::2] = A[-1, ::2]
    G[2::2, 0] = A[1:-1:2, 0]
    G[1::2, -1] = A[:-1:2, -1]

    # print(time.time() - start)
    rgb = np.dstack((R, G, B))
    return rgb








