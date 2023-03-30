# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# credits: https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/metrics/rank_cylib/rank_cy.pyx

import cython
import numpy as np
cimport numpy as np
from collections import defaultdict


"""
Compiler directives:
https://github.com/cython/cython/wiki/enhancements-compilerdirectives
Cython tutorial:
https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html
Credit to https://github.com/luzai
"""


# Main interface
cpdef evaluate_cy(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    distmat = np.asarray(distmat, dtype=np.float32)
    q_pids = np.asarray(q_pids, dtype=np.int64)
    g_pids = np.asarray(g_pids, dtype=np.int64)
    q_camids = np.asarray(q_camids, dtype=np.int64)
    g_camids = np.asarray(g_camids, dtype=np.int64)
    return eval_market1501_cy(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)


cpdef eval_market1501_cy(float[:,:] distmat, long[:] q_pids, long[:]g_pids,
                         long[:]q_camids, long[:]g_camids, long max_rank):

    cdef long num_q = distmat.shape[0]
    cdef long num_g = distmat.shape[1]

    if num_g < max_rank:
        max_rank = num_g

    cdef:
        long[:,:] indices = np.argsort(distmat, axis=1)
        long[:] matches

        float[:,:] all_cmc = np.zeros((num_q, max_rank), dtype=np.float32)
        float[:] all_AP = np.zeros(num_q, dtype=np.float32)
        float[:] all_INP = np.zeros(num_q, dtype=np.float32)
        float num_valid_q = 0. # number of valid query
        long valid_index = 0

        long q_idx, q_pid, q_camid, g_idx
        long[:] order = np.zeros(num_g, dtype=np.int64)
        long keep

        float[:] raw_cmc = np.zeros(num_g, dtype=np.float32) # binary vector, positions with value 1 are correct matches
        float[:] cmc = np.zeros(num_g, dtype=np.float32)
        long max_pos_idx = 0
        float inp
        long num_g_real, rank_idx
        unsigned long meet_condition

        float num_rel
        float[:] tmp_cmc = np.zeros(num_g, dtype=np.float32)
        float tmp_cmc_sum


    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        for g_idx in range(num_g):
            order[g_idx] = indices[q_idx, g_idx]
        num_g_real = 0
        meet_condition = 0
        matches = (np.asarray(g_pids)[np.asarray(order)] == q_pid).astype(np.int64)

        # remove gallery samples that have the same pid and camid with query
        for g_idx in range(num_g):
            cam_cond = ((g_pids[order[g_idx]] != q_pid) or (g_camids[order[g_idx]] != q_camid)) if q_camid != -1 else True
            if cam_cond:
                raw_cmc[num_g_real] = matches[g_idx]
                num_g_real += 1
                # this condition is true if query appear in gallery
                if matches[g_idx] > 1e-31:
                    meet_condition = 1

        if not meet_condition:
            # this condition is true when query identity does not appear in gallery
            continue

        # compute cmc
        function_cumsum(raw_cmc, cmc, num_g_real)
        # compute mean inverse negative penalty
        # reference : https://github.com/mangye16/ReID-Survey/blob/master/utils/reid_metric.py
        max_pos_idx = 0
        for g_idx in range(num_g_real):
            if (raw_cmc[g_idx] == 1) and (g_idx > max_pos_idx):
                max_pos_idx = g_idx
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP[valid_index] = inp

        for g_idx in range(num_g_real):
            if cmc[g_idx] > 1:
                cmc[g_idx] = 1

        for rank_idx in range(max_rank):
            all_cmc[q_idx, rank_idx] = cmc[rank_idx]
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        function_cumsum(raw_cmc, tmp_cmc, num_g_real)
        num_rel = 0
        tmp_cmc_sum = 0
        for g_idx in range(num_g_real):
            tmp_cmc_sum += (tmp_cmc[g_idx] / (g_idx + 1.)) * raw_cmc[g_idx]
            num_rel += raw_cmc[g_idx]
        all_AP[valid_index] = tmp_cmc_sum / num_rel
        valid_index += 1

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    # compute averaged cmc
    cdef float[:] avg_cmc = np.zeros(max_rank, dtype=np.float32)
    for rank_idx in range(max_rank):
        for q_idx in range(num_q):
            avg_cmc[rank_idx] += all_cmc[q_idx, rank_idx]
        avg_cmc[rank_idx] /= num_valid_q

    return np.asarray(avg_cmc).astype(np.float32), np.asarray(all_AP[:valid_index]).mean()


# Compute the cumulative sum
cdef void function_cumsum(cython.numeric[:] src, cython.numeric[:] dst, long n):
    cdef long i
    dst[0] = src[0]
    for i in range(1, n):
        dst[i] = src[i] + dst[i - 1]