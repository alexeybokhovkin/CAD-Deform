#!/usr/bin/env python3

########################################################################
# Alignment input folder structure
########################################################################
#
# ./Validation/Network-output/kernel-{3|5|7|9}/{scan_id}/{shape_category_id}_{shape_object_id}/
# 	/input.json
# 		[
# 			{
# 				'p_scan': [0.3, 1.21, 3.2],     # keypoints (from network input)
# 				'heatmap': 'heatmaps/XXX.vox',  # location of heatmap from network output
# 				'scale': [1.2, 1.1, 0.9] 		# scale from network output
# 				'match': 0.2 					# match from network output
# 			},
# 			...
# 		]
# 	/heatmaps/
# 		XXX.vox
# 		YYY.vox
# 		ZZZ.vox
# 		...
#
########################################################################
# Alignment output folder structure
########################################################################
#
# ./Validation/Align-output-XXX/
# 	{scan_id}.csv
# 		columns: objectCategory,alignedModelId,tx,ty,tz,qw,qx,qy,qz,sx,sy,sz
# 	{scan_id}.log
#
########################################################################

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
import torch

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..')
)

sys.path[1:1] = [__dir__]

from shapefit.utils.utils import rts_to_matrix, make_M_from_tqs, get_theta, get_validation_appearance
from shapefit.utils.utils import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis, angle_axis_to_quaternion
from shapefit.utils.vox_loader import load_sample, vox2obj


def mean_heatmap_coord(hm):
    # take all where |sdf| < 1
    idx = np.argwhere(np.abs(hm.sdf[0]) < hm.res)

    # take all where |pdf| > 0
    idx = np.array([k for k in idx if hm.pdf[0][k[0], k[1], k[2]] > 1e-8])

    # normalize pdf
    pdf = np.array([hm.pdf[0][k[0], k[1], k[2]] for k in idx])
    pdf /= sum(pdf)

    # take it coords
    idx = idx[:, ::-1]
    mean_coord = list((idx * pdf.reshape(-1, 1)).sum(0)) + [1]
    mean_coord = (hm.grid2world @ mean_coord)[:3]

    return mean_coord


def align(DIR, options):
    input_name = os.path.join(DIR, 'input.json')
    output_log = os.path.join(options.path_to_alignments, options.scan_id + '.log')

    # LOAD INPUT DATA
    with open(input_name, 'r') as f:
        input_json = json.load(f)

    if len(input_json) == 0:
        raise Exception('No correspondences')

    heatmaps = [load_sample(os.path.join(DIR, p['heatmap'])) for p in input_json]
    keypoints = np.array([p['p_scan'] for p in input_json])
    scales = np.array([p['scale'] for p in input_json])

    # POINTS INITIALIZATION
    # initialize keypoints & heatmaps
    kps = torch.tensor(keypoints)
    n_keypoints = len(kps)
    heatmap_values = torch.DoubleTensor([heatmaps[i].pdf[0] for i in range(len(kps))]).reshape(n_keypoints, -1)
    results = []

    # initial position grid on scan
    idx_shape = np.array(np.meshgrid(range(32), range(32), range(32))).T
    idx_shape = np.swapaxes(idx_shape, 1, 2)
    idx_shape = idx_shape @ heatmaps[0].grid2world[:3, :3] + heatmaps[0].grid2world[:3, -1]
    idx_shape = np.hstack((idx_shape.reshape(-1, 3), np.ones((32 ** 3, 1))))
    idx_shape = torch.tensor(idx_shape)

    # per keypoint stuff
    idx_shape = idx_shape.reshape(1, -1, 4).repeat(n_keypoints, 1, 1)

    idx_sdf = np.argwhere(np.abs(heatmaps[0].sdf[0].flatten()) < heatmaps[0].res).flatten()
    heatmap_values = heatmap_values[:, idx_sdf]
    idx_shape = idx_shape[:, idx_sdf]

    # ANGLES INITIALIZATION
    random_angle = torch.DoubleTensor([[0, 0, x] for x in (np.random.rand(len(keypoints)) * 2 - 1) * np.pi])
    random_angle = angle_axis_to_rotation_matrix(random_angle)
    gen_rotation = angle_axis_to_rotation_matrix(torch.DoubleTensor([[options.angle_range, 0, 0]]))
    rotations = rotation_matrix_to_angle_axis((random_angle @ gen_rotation)[:, :3]).numpy()

    #######
    # Graph
    #######

    # elementwise distance between keypoints and heatmaps
    keypoints_cdist = cdist(keypoints, keypoints)
    heatmap_cdist = 1e+10 * np.ones_like(keypoints_cdist)

    heatmap_coords = np.array([mean_heatmap_coord(hm) for hm in heatmaps])

    for i in range(len(keypoints)):
        for j in range(len(keypoints)):
            scale_diff = np.array([scales[i], scales[j]])
            diff = min(scale_diff.min(0) / scale_diff.max(0))
            if diff < options.clustering_difference:
                continue

            scale_mean = scale_diff.mean(0)
            heatmap_diff = heatmap_coords[[i, j]] @ np.diag(scale_mean)
            heatmap_cdist[i, j] = np.linalg.norm(heatmap_diff[0] - heatmap_diff[1])

    # proporsion between these two matrices
    main_cdist = np.minimum(heatmap_cdist, keypoints_cdist) / np.maximum(heatmap_cdist, keypoints_cdist)
    main_cdist = np.nan_to_num(main_cdist)
    main_cdist += np.eye(len(main_cdist))
    main_cdist = 1 - main_cdist

    # clustering
    if options.clustering_distance_threshold == 0:
        n = 13 # TODO: we know the number of clusters. Load it
        sc = AgglomerativeClustering(n_clusters=n, affinity='precomputed', linkage='average')
    else:
        sc = AgglomerativeClustering(n_clusters=None, distance_threshold=options.clustering_distance_threshold,
                                     affinity='precomputed', linkage='average')

    sc.fit(main_cdist)

    # batches for per-cluster optimization
    batch = []
    for k in range(sc.n_clusters_):
        idx = sc.labels_ == k
        nums = np.argwhere(sc.labels_ == k).flatten()
        batch.append((
            nums,
            rotations[idx], kps[nums], scales[idx],
            heatmap_values[nums]
        ))

    costs = np.empty(len(sc.labels_))
    result_T = torch.empty(len(sc.labels_), 9).double()

    # Hyperparams
    lr = options.learning_rate
    max_iter = options.max_iter

    if options.kernel_theta_type == 'const':
        theta = options.kernel_theta
    else:
        theta = 0

    # OPTIMIZATION LOOP
    for num, rot, keypts, scs, htmaps in batch:
        init_T = [list(r) + list(k[:3]) + list(s) for r, k, s in zip(rot, keypts, scs)]
        n_keypoints = len(init_T)

        T = torch.tensor(init_T, dtype=torch.double, requires_grad=True)
        T_matrix = rts_to_matrix(T)

        optimizer = torch.optim.Adam([T], lr=lr)
        learning_store = [T_matrix.detach().numpy().copy()]

        with open(output_log, 'a+') as f:
            print_string = '#\tMinVal\t\tMeanVal\t\tMaxVal\t\tGrad Norm\n'
            f.write(print_string)

            if options.verbose:
                print(print_string[:-1])

        for it in range(max_iter):
            # from 9DoF to matrix
            T_matrix = rts_to_matrix(T)

            # Translate shapenet cube 32x32x32x3
            idx_shape_translated = torch.einsum('nij,nkj->nik', idx_shape[list(num)], T_matrix)

            if options.kernel_theta_type == 'adaptive' and theta == 0:
                left = idx_shape_translated.max(1)[0]
                right = idx_shape_translated.min(1)[0]

                delta = (left - right).mean(0).detach().numpy()
                theta = get_theta(delta)

            # shape = n_restarts x n_keypoints x n_points x n_dim(3)
            left = idx_shape_translated[:, :, :3].reshape(n_keypoints, 1, -1, 3)
            right = keypts.reshape(1, n_keypoints, 1, 3)

            # kernel between cubes and keypoints
            if options.kernel_type == 'gaussian':
                inner_mean = torch.mean((left - right) ** 2, -1)
                kernel = torch.exp(- theta * inner_mean)
            elif options.kernel_type == 'exp':
                inner_mean = torch.mean(torch.abs(left - right), -1)
                kernel = torch.exp(- theta * inner_mean)
            else:
                #TODO: epanechnikov kernel
                raise NotImplementedError('Epanechnikov kernel not implemented')

            # masking according to heatmap
            masked_kernel = kernel * htmaps.reshape(1, n_keypoints, -1)

            # value over restarts (BIGGER -> BETTER)
            val = masked_kernel.mean(-1).mean(-1)

            # TODO: add regularization on scale

            # STEP
            optimizer.zero_grad()
            (-val.mean()).backward()
            optimizer.step()

            grad_norm = np.linalg.norm(T.grad.data.numpy())

            if it % options.save_every == 0:
                learning_store.append(T_matrix.detach().numpy().copy())

                buf = val.detach().numpy()
                with open(output_log, 'a+') as f:
                    print_string = '%d:\t%e\t%e\t%e\t%e\n' % (it, buf.min(), buf.mean(), buf.max(), grad_norm)
                    f.write(print_string)

                if options.verbose:
                    print(print_string[:-1])


            if grad_norm < options.float_tolerance:
                with open(output_log, 'a+') as f:
                    print_string = 'Terminated. Torelance\n'
                    f.write(print_string)

                if options.verbose:
                    print(print_string[:-1])

                break

        # REARRANGE according to cost function
        costs[num] = -val.detach().numpy()
        result_T[num] = T

    rearranged_idx = np.argsort(costs)
    Ts = result_T[rearranged_idx]

    quaternion = angle_axis_to_quaternion(Ts[:, :3]).detach().numpy()
    vector = Ts[:, 3:6].detach().numpy()
    scale = Ts[:, 6:].detach().numpy()

    return np.hstack((vector, quaternion, scale)), learning_store


def pruning(options, shape_vis, T):
    output_log = os.path.join(options.path_to_alignments, options.scan_id + '.log')
    shape_vis = shape_vis.reshape(1, -1, 4).repeat(len(T), 1, 1)

    # calculating shapes position
    Ts = torch.tensor([make_M_from_tqs(x[:3], x[3:7], x[7:]) for x in T])
    shapes = torch.einsum('nij,nkj->nik', shape_vis, Ts).detach().numpy()

    # SORT
    with open(output_log, 'a+') as f:
        print_string = 'Init: %d alignments\n' % len(T)
        f.write(print_string)

        if options.verbose:
            print(print_string[:-1])

    ## PRUNNING 1 -> with scale < 0
    if options.pruning_scale:
        idx = list(T[:, 7:].min(-1) > 0)
        if sum(idx) > 0.:
            T, Ts, shapes = T[idx], Ts[idx], shapes[idx]

        with open(output_log, 'a+') as f:
            print_string = 'After negative pruning: %d alignments\n' % len(T)
            f.write(print_string)

            if options.verbose:
                print(print_string[:-1])

    ## PRUNNING 2 -> intersection
    bbox = np.array([shapes.max(1), shapes.min(1)]).swapaxes(0, 1)[:, :, :3]
    intersect = np.array([[np.all([
        max(b1[1, i] - b2[0, i], b2[1, i] - b1[0, i]) < -options.pruning_intersection_tolerance
    for i in range(3)]) for b2 in bbox] for b1 in bbox])

    idx = []
    for i in range(len(shapes)):
        if len(idx) == 0:
            idx = [i]
            continue

        if not np.any(intersect[i, idx]):
            idx.append(i)

    T, Ts, shapes = T[idx], Ts[idx], shapes[idx]
    with open(output_log, 'a+') as f:
        print_string = 'FINAL After intersections: %d alignments\n' % len(T)
        f.write(print_string)

        if options.verbose:
            print(print_string[:-1])

    return T


def save(k, options, Ts):
    scan_id = options.scan_id
    RESULTS_DIR = options.path_to_alignments
    category_id, shapenet_id = k.split('_')

    cols = [
        'objectCategory', 'alignedModelId',
        'tx', 'ty', 'tz',
        'qw', 'qx', 'qy', 'qz',
        'sx', 'sy', 'sz'
    ]

    df_result = pd.DataFrame([[category_id, shapenet_id] + list(x) for x in Ts])
    df_result.columns = cols

    # SAVING
    file_align = os.path.join(RESULTS_DIR, scan_id + '.csv')
    if os.path.exists(file_align):
        df_prev = pd.read_csv(file_align, dtype={'objectCategory': str})
        if sum(df_prev[cols[1]] == shapenet_id) != 0:
            df_prev = df_prev[df_prev[cols[1]] != shapenet_id]
        df_result = pd.concat([df_prev, df_result], sort=False).reset_index()[cols]

    df_result.to_csv(file_align)



def main(options):
    scan_id = options.scan_id
    output_log = os.path.join(options.path_to_alignments, options.scan_id + '.log')

    appearance = get_validation_appearance('2k')
    for k in appearance[scan_id]:
        # e.g. /data/network-output/kernel-5/scene0535_00/04379243_aa3a0c759428a9eaa5199c5eb7fa3865/
        DIR = os.path.join(
            options.input_dir,
            'kernel-{}'.format(options.sampling_kernel),
            scan_id, k
        )

        if not os.path.exists(DIR):
            print_string = '##' * 40 + '\n### ALARM: ' + k + ' does not exist\n'
            with open(output_log, 'a+') as f:
                f.write(print_string)

            if options.verbose:
                print(print_string[:-1])

            continue

        print_string = '##' * 40 + '\n### Object: ' + k + '\n'
        with open(output_log, 'a+') as f:
            f.write(print_string)

        if options.verbose:
            print(print_string[:-1])

        with open(os.path.join(DIR, 'input.json'), 'rb') as f:
            input_json = json.load(f)

        # load shape vox
        vox_filename = os.path.join(DIR, input_json[0]['heatmap'])
        vox = load_sample(vox_filename)
        obj = torch.tensor(vox2obj(vox))

        # ALIGNMENT
        result, store = align(DIR, options)

        # PRUNING
        Ts = pruning(options, obj, result)

        # SAVING
        save(k, options, Ts)


def parse_args():
    parser = argparse.ArgumentParser()

    # Location parameters
    parser.add_argument('-i', '--input-dir', dest='input_dir', default='./Validation/Network-output/',
                        help='Directory with Scan2CAD output.')
    parser.add_argument('-o', '--path-to-alignments', dest='path_to_alignments', default='./Validation/Align-output/',
                        help='output directory used to save CSV files with alignments.')

    # Sampling param
    parser.add_argument('-k', '--sampling-kernel', dest='sampling_kernel', choices=[3, 5, 7, 9], default=5,
                        help='Sampling kernel width.')

    # Scan2CAD param
    parser.add_argument('-m', '--match-threshold', dest='match_threshold', type=float, default=0.05,
                        help='Alignment considers correspondences only with 0 <= *match_threshold* <= *match* <= 1.')

    # Init params
    parser.add_argument('-c', '--scan-id', dest='scan_id', default='scene0535_00', help='Scene ID from Scannet.')
    parser.add_argument('-a', '--angle-range', dest='angle_range', default=np.pi/2, type=float,
                        help='Angle range for init random object rotation [0, *a*]')

    # Clustering params
    parser.add_argument('-d', '--clustering-difference', dest='clustering_difference', default=0.7, type=float, help='''
        Group point if their distance in scan space not far away from then their distance in heatmap space: 
        d * scan_dist < heatmap_dist < scan_dist / d,
        0 <= d <= 1
    ''')
    parser.add_argument('--clustering-distance-threshold', dest='clustering_distance_threshold', default=0.2,
                        help='Init parameter for Agglomerative clustering')

    # Alignment params
    parser.add_argument('--learning-rate', dest='learning_rate', default=1e-1, type=float,
                        help='Learning rate for alignment optimization')
    parser.add_argument('--max-iter', dest='max_iter', default=100,
                        help='Maximum number of iterations of alignment optimization')
    parser.add_argument('--kernel-type', dest='kernel_type', choices=['gaussian', 'exp', 'epanechnikov'],
                        default='gaussian', help='Type of kernel for alignment')
    parser.add_argument('--kernel-theta-type', dest='kernel_theta_type', choices=['adaptive', 'const'],
                        default='adaptive', help='Type of kernel theta for alignment')
    parser.add_argument('--kernel-theta', dest='kernel_theta', default=1e-1, type=float,
                        help='Kernel theta for alignment. Used only if kernel_theta_type = const')

    # Pruning params
    parser.add_argument('--pruning-scale', dest='pruning_scale', default=True,
                        help='If True then remove all results with negative scale')
    parser.add_argument('--pruning-intersection-tolerance', dest='pruning_intersection_tolerance', default=0.2,
                        type=float,
                        help='''
        Results have intersections between each other. 
        Delete all result with intersections >= *pruning_intersection_tolerance*
    ''')

    # Preparation for deformation
    parser.add_argument('--around-area', dest='around_area', default=0.1, type=float,
                        help='Area of taking voxels from scan for the deformation')
    parser.add_argument('--bbox-scale', dest='bbox_scale', default=0.9, type=float,
                        help='Scale obtained bbox of the aligned object')

    # Other params
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Print verbose output.')
    parser.add_argument('-t', '--float-tolerance', dest='float_tolerance', default=1e-10,
                        help='Tolerance for optimization termination')
    parser.add_argument('-s', '--save-every', dest='save_every', default=5, help='Saving gap in optimization process')

    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
