#!/usr/bin/env python3

import argparse
import glob
import numpy as np
import os
import pandas as pd
import quaternion
import sys
import trimesh
import json

from tqdm import tqdm
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings("ignore")

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..')
)

sys.path[1:1] = [__dir__]

top_classes = {
    "03211117": "display", "04379243": "table",
    "02747177": "trashbin", "03001627": "chair",
    # "04256520": "sofa", "02808440": "bathtub",
    "02933112": "cabinet", "02871439": "bookshelf"
}

from shapefit.utils.utils import get_validation_appearance, get_symmetries, get_gt_dir, \
    get_scannet, get_shapenet, make_M_from_tqs, make_tqs_from_M


# helper function to calculate difference between two quaternions
def calc_rotation_diff(q, q00):
    rotation_dot = np.dot(quaternion.as_float_array(q00), quaternion.as_float_array(q))
    rotation_dot_abs = np.abs(rotation_dot)
    try:
        error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    except:
        return 0.0

    error_rotation = np.rad2deg(error_rotation_rad)
    return error_rotation


def rotation_error(row):
    q = quaternion.quaternion(*row[:4])
    q_gt = quaternion.quaternion(*row[4:8])
    sym = row[-1]

    if sym == "__SYM_ROTATE_UP_2":
        m = 2
        tmp = [
            calc_rotation_diff(q, q_gt * quaternion.from_rotation_vector([0, (i * 2.0 / m) * np.pi, 0]))
            for i in range(m)]
        return np.min(tmp)
    elif sym == "__SYM_ROTATE_UP_4":
        m = 4
        tmp = [
            calc_rotation_diff(q, q_gt * quaternion.from_rotation_vector([0, (i * 2.0 / m) * np.pi, 0]))
            for i in range(m)]
        return np.min(tmp)
    elif sym == "__SYM_ROTATE_UP_INF":
        m = 36
        tmp = [
            calc_rotation_diff(q, q_gt * quaternion.from_rotation_vector([0, (i * 2.0 / m) * np.pi, 0]))
            for i in range(m)]
        return np.min(tmp)
    else:
        return calc_rotation_diff(q, q_gt)


def print_to_(verbose, log_file, string):
    if verbose:
        print(string)
        sys.stdout.flush()

    with open(log_file, 'a+') as f:
        f.write(string + '\n')


def get_init_mesh(scan_id, key):
    path = glob.glob(os.path.join(
        '/home/ishvlad/workspace/Scan2CAD/MeshDeformation/ARAP/',
        'arap_output_GT', scan_id, key + '*', 'init.obj'
    ))

    if len(path) == 0:
        return None

    return trimesh.load_mesh(path[0], process=False)


def DAME(mesh_1, mesh_2, k=0.59213):
    def dihedral(mesh):
        unique_faces, _ = np.unique(np.sort(mesh.faces, axis=1), axis=0, return_index=True)

        parts_bitriangles_map = []
        bitriangles = {}
        for face in unique_faces:
            edge_1 = tuple(sorted([face[0], face[1]]))
            if edge_1 not in bitriangles:
                bitriangles[edge_1] = set([face[0], face[1], face[2]])
            else:
                bitriangles[edge_1].add(face[2])
            edge_2 = tuple(sorted([face[1], face[2]]))
            if edge_2 not in bitriangles:
                bitriangles[edge_2] = set([face[0], face[1], face[2]])
            else:
                bitriangles[edge_2].add(face[0])
            edge_3 = tuple(sorted([face[0], face[2]]))
            if edge_3 not in bitriangles:
                bitriangles[edge_3] = set([face[0], face[1], face[2]])
            else:
                bitriangles[edge_3].add(face[1])
        bitriangles_aligned = np.empty((len(mesh.edges_unique), 4), dtype=int)
        for j, edge in enumerate(mesh.edges_unique):
            bitriangle = [*sorted(edge)]
            bitriangle += [x for x in list(bitriangles[tuple(sorted(edge))]) if x not in bitriangle]
            bitriangles_aligned[j] = bitriangle

        vertices_bitriangles_aligned = mesh.vertices[bitriangles_aligned]

        normals_1 = np.cross((vertices_bitriangles_aligned[:, 2] - vertices_bitriangles_aligned[:, 0]),
                             (vertices_bitriangles_aligned[:, 2] - vertices_bitriangles_aligned[:, 1]))
        normals_1 = normals_1 / np.sqrt(np.sum(normals_1 ** 2, axis=1)[:, None])

        normals_2 = np.cross((vertices_bitriangles_aligned[:, 3] - vertices_bitriangles_aligned[:, 0]),
                             (vertices_bitriangles_aligned[:, 3] - vertices_bitriangles_aligned[:, 1]))
        normals_2 = normals_2 / np.sqrt(np.sum(normals_2 ** 2, axis=1)[:, None])

        n1_n2_arccos = np.arccos(np.sum(normals_1 * normals_2, axis=1).clip(-1, 1))
        n1_n2_signs = np.sign(
            np.sum(normals_1 * (vertices_bitriangles_aligned[:, 3] - vertices_bitriangles_aligned[:, 1]), axis=1))
        D_n1_n2 = n1_n2_arccos * n1_n2_signs

        return D_n1_n2

    D_mesh_1 = dihedral(mesh_1)
    D_mesh_2 = dihedral(mesh_2)

    mask_1 = np.exp((k * D_mesh_1) ** 2)

    per_edge = np.abs(D_mesh_1 - D_mesh_2) * mask_1
    result = np.sum(per_edge) / len(mesh_1.edges_unique)

    return result, per_edge


def calc_ATSD(dists, border):
    return np.minimum(dists.min(1), border).mean()


def calc_F1(dists, border):
    return np.sum(dists.min(1) < border) / len(dists)


def calc_CD(dists, border):
    return max(np.minimum(dists.min(1), border).mean(), np.minimum(dists.min(0), border).mean())


def calc_metric(scan_mesh, shape_mesh, method='all', border=0.1):
    area = border * 2
    # get scan bbox
    bbox = np.array([shape_mesh.vertices.min(0), shape_mesh.vertices.max(0)])
    bbox += [[-area], [area]]

    batch = np.array([np.diag(bbox[0]), np.diag(bbox[1]), np.eye(3), -np.eye(3)])
    slice_mesh = scan_mesh.copy()

    # xyz
    for i in range(3):
        slice_mesh = slice_mesh.slice_plane(batch[0, i], batch[2, i])
        slice_mesh = slice_mesh.slice_plane(batch[1, i], batch[3, i])

    if len(slice_mesh.vertices) == 0:
        if method == 'all':
            return {'ATSD': border, 'CD': border, 'F1': 0.0}
        else:
            return border

    scan_vertices = np.array(slice_mesh.vertices)
    if len(scan_vertices) > 20000:
        scan_vertices = scan_vertices[::len(scan_vertices) // 20000]

    dists = cdist(np.array(shape_mesh.vertices), scan_vertices, metric='minkowski', p=1)

    if method == 'ATSD':
        return calc_ATSD(dists, border)
    elif method == 'CD':
        return calc_CD(dists, border)
    elif method == 'F1':
        return calc_F1(dists, border)
    else:
        return {
            'ATSD': calc_ATSD(dists, border),
            'CD': calc_CD(dists, border),
            'F1': calc_F1(dists, border),
        }


def metric_on_deformation(options):
    output_name = options.output_name + '_' + str(options.border) + \
                  '_' + str(options.val_set) + '_' + str(options.metric_type)

    if options.output_type == 'align':
        # load needed models
        appearance = get_validation_appearance(options.val_set)

        # LOAD list of all aligned scenes
        csv_files = glob.glob(os.path.join(options.input_dir, '*.csv'))
        scenes = [x.split('/')[-1][:-4] for x in csv_files]

        # Which scenes do we want to calculate?
        scenes = np.intersect1d(scenes, list(appearance.keys()))

        batch = []
        for s in scenes:
            df_scan = pd.read_csv(
                os.path.join(options.input_dir, s + '.csv'),
                index_col=0, dtype={'objectCategory': str}
            )

            # Filter: take only objects from appearance
            df_scan['key'] = df_scan.objectCategory + '_' + df_scan.alignedModelId
            df_scan = df_scan[np.in1d(df_scan['key'].values, list(appearance[s].keys()))]

            batch.extend([{
                'scan_id': s,
                'key': row['key'],
                'objectCategory': row['objectCategory'],
                'alignedModelId': row['alignedModelId'],
                'path': 'path to origin ShapeNet mesh',
                'object_num': i,
                'T': [row['tx'], row['ty'], row['tz']],
                'Q': [row['qw'], row['qx'], row['qy'], row['qz']],
                'S': [row['sx'], row['sy'], row['sz']]
            } for i, row in df_scan.iterrows()])

        df = pd.DataFrame(batch)
    else:
        # LOAD list of all aligned scenes
        in_files = glob.glob(os.path.join(options.input_dir, 'scene*/*/approx.obj'))
        if len(in_files) == 0:
            in_files = glob.glob(os.path.join(options.input_dir, '*/scene*/*/approx.obj'))

        info = []
        for x in in_files:
            parts = x.split('/')[-3:-1]

            if len(parts[1].split('_')) == 3:
                category_id, shape_id, object_num = parts[1].split('_')
            else:
                category_id, shape_id = parts[1].split('_')
                object_num = -1

            row = [
                parts[0],  # scan_id
                category_id + '_' + shape_id,  # key
                category_id,
                shape_id,
                object_num,
                x,  # path
            ]
            info.append(row)

        df = pd.DataFrame(info, columns=['scan_id', 'key', 'objectCategory', 'alignedModelId', 'object_num', 'path'])

        transform_files = ['/'.join(x.split('/')[:-1]) + '/transform.json' for x in in_files]

        Ts, Qs, Ss = [], [], []
        for f in transform_files:
            if os.path.exists(f):
                matrix = np.array(json.load(open(f, 'rb'))['transform'])
            else:
                Ts.append(None)
                Qs.append(None)
                Ss.append(None)
                continue

            t, q, s = make_tqs_from_M(matrix)
            q = quaternion.as_float_array(q)

            Ts.append(t)
            Qs.append(q)
            Ss.append(s)

        df['T'] = Ts
        df['Q'] = Qs
        df['S'] = Ss

    metrics = {}

    batch = df.groupby('scan_id')
    if options.verbose:
        batch = tqdm(batch, desc='Scenes')

    # CALCULATE METRICS
    for scan_id, df_scan in batch:
        scan_mesh = get_scannet(scan_id, 'mesh')

        scan_batch = df_scan.iterrows()
        if options.verbose:
            scan_batch = tqdm(scan_batch, total=len(df_scan), desc='Shapes', leave=False)

        for i, row in scan_batch:
            if options.output_type == 'align':
                shape_mesh = get_shapenet(row['objectCategory'], row['alignedModelId'], 'mesh')
            else:
                try:
                    shape_mesh = trimesh.load_mesh(row['path'])
                except Exception:
                    metrics[i] = {'ATSD': np.nan, 'CD': np.nan, 'F1': np.nan}
                    continue

            if row['T'] is None:
                metrics[i] = {'ATSD': np.nan, 'CD': np.nan, 'F1': np.nan}
                continue

            T = make_M_from_tqs(row['T'], row['Q'], row['S'])
            shape_mesh.apply_transform(T)

            metrics[i] = calc_metric(scan_mesh, shape_mesh, border=options.border)

    df_final = df.merge(pd.DataFrame(metrics).T, left_index=True, right_index=True)
    df_final.to_csv(output_name + '.csv')

    if len(df_final) == 0:
        print_to_(options.verbose, output_name + '.log', 'No aligned shapes')
        return

    df_final = df_final[~pd.isna(df_final['ATSD'])]

    # Calculate INSTANCE accuracy
    acc = df_final[['ATSD', 'CD', 'F1']].mean().values
    acc[-1] *= 100
    print_string = '#' * 57 + '\nINSTANCE MEAN. ATSD: {:>4.2f}, CD: {:>4.2f}, F1: {:6.2f}\n'.format(
        *acc) + '#' * 57
    print_to_(options.verbose, output_name + '.log', print_string)

    df_final['name'] = [top_classes.get(x, 'zother') for x in df_final.objectCategory]
    df_class = df_final.groupby('name').mean()[['ATSD', 'CD', 'F1']]

    print_string = '###' + ' ' * 7 + 'CLASS' + ' ' * 4 + '# ATSD   #   CD    #    F1    ###'
    print_to_(options.verbose, output_name + '.log', print_string)

    for name, row in df_class.iterrows():
        print_string = '###\t{:10} # {:>4.2f}   #  {:>4.2f}   #  {:6.2f}  ###'.format(
            name, row['ATSD'], row['CD'], row['F1']*100
        )
        print_to_(options.verbose, output_name + '.log', print_string)

    acc = df_class.mean().values
    acc[-1] *= 100
    print_string = '#' * 57 + '\n   CLASS MEAN. ATSD: {:>4.2f}, CD: {:>4.2f}, F1: {:6.2f}\n'.format(
        *acc) + '#' * 57
    print_to_(options.verbose, output_name + '.log', print_string)


def metric_on_alignment(options):
    output_name = options.output_name + '_' + str(options.border) + \
                  '_' + str(options.val_set) + '_' + str(options.metric_type)

    if options.output_type == 'deform':
        raise Exception

    # LOAD list of all aligned scenes
    csv_files = glob.glob(os.path.join(options.input_dir, '*.csv'))
    scenes = [x.split('/')[-1][:-4] for x in csv_files]

    # Which scenes do we want to calculate?
    appearances_cad = get_validation_appearance(options.val_set)
    df_appearance = pd.DataFrame(np.concatenate([
        [(k, kk, appearances_cad[k][kk]) for kk in appearances_cad[k]] for k in appearances_cad
    ]), columns=['scan_id', 'key', 'count'])

    scenes = np.intersect1d(scenes, list(set(df_appearance.scan_id)))

    # LOAD GT and target alignments
    gt_dir = get_gt_dir('align')
    batch = []
    batch_gt = []
    for s in scenes:
        df = pd.read_csv(os.path.join(gt_dir, s + '.csv'), index_col=0, dtype={'objectCategory': str})
        df['scan_id'] = s
        # Filter: take only objects from appearance
        df['key'] = df.objectCategory + '_' + df.alignedModelId
        df = df[np.in1d(df['key'].values, list(appearances_cad[s].keys()))]
        batch_gt.append(df)

        df = pd.read_csv(os.path.join(options.input_dir, s + '.csv'), index_col=0, dtype={'objectCategory': str})
        df['scan_id'] = s
        # Filter: take only objects from appearance
        df['key'] = df.objectCategory + '_' + df.alignedModelId
        df = df[np.in1d(df['key'].values, list(appearances_cad[s].keys()))]
        batch.append(df)

    df_alignment = pd.concat(batch)
    df_alignment.reset_index(drop=True, inplace=True)

    df_alignment_gt = pd.concat(batch_gt)
    df_alignment_gt.reset_index(drop=True, inplace=True)

    # Create index for each GT object
    df_alignment_gt.reset_index(inplace=True)

    # LOAD Symmetry info
    df_symmetry = get_symmetries()
    df_alignment = df_alignment.merge(df_symmetry, how='left', on='key')
    df_alignment_gt = df_alignment_gt.merge(df_symmetry, how='left', on='key')

    # Make pairs for difference calculation
    df_mutual = df_alignment_gt.merge(df_alignment, how='left', on=['scan_id', 'objectCategory'], suffixes=('_gt', ''))

    # Calculate the difference
    t_dist = df_mutual[['tx_gt', 'ty_gt', 'tz_gt']].values - df_mutual[['tx', 'ty', 'tz']].values
    df_mutual['t_dist'] = np.linalg.norm(t_dist, ord=2, axis=-1)

    s_diff = df_mutual[['sx', 'sy', 'sz']].values / df_mutual[['sx_gt', 'sy_gt', 'sz_gt']].values
    df_mutual['s_dist'] = 100 * np.abs(s_diff.mean(-1) - 1)

    cols = ['qw', 'qx', 'qy', 'qz', 'qw_gt', 'qx_gt', 'qy_gt', 'qz_gt', 'symmetry']
    df_mutual['q_dist'] = [rotation_error(row) for row in df_mutual[cols].values]
    df_mutual.q_dist.fillna(0, inplace=True)

    # does the aligned shape is near to GT shape
    df_mutual['is_fitted'] = (df_mutual.t_dist <= 0.2) & (df_mutual.q_dist <= 20) & (df_mutual.s_dist <= 20)

    # GET and SAVE the result
    df_fit = df_mutual[['index', 'is_fitted']].groupby('index').max().reset_index()
    df_fit = df_alignment_gt.merge(df_fit, on='index')

    df_final = df_fit[['scan_id', 'objectCategory', 'alignedModelId', 'is_fitted']]
    df_final.to_csv(output_name + '.csv')

    # Calculate INSTANCE accuracy
    df_scan = df_final.groupby('scan_id').agg({'is_fitted': ('sum', 'count')})
    df_scan.columns = ['sum', 'count']
    total = df_scan.sum()
    acc = total['sum'] / total['count'] * 100

    print_string = '#' * 80 + '\n# scenes: {}, # fitted: {}, # total: {}, INSTANCE ACCURACY: {:>4.2f}\n'.format(
        len(df_scan), int(total['sum']), int(total['count']), acc
    ) + '#' * 80
    print_to_(options.verbose, output_name + '.log', print_string)

    # Calculate CLASS accuracy
    df_final['name'] = [top_classes.get(x, 'zother') for x in df_final.objectCategory]

    df_class = df_final.groupby('name').agg({'is_fitted': ('sum', 'count')})
    df_class.columns = ['sum', 'count']
    df_class.sort_index()

    for name, row in df_class.iterrows():
        print_string = '\t{:10} # fitted: {:4}, # total: {:4}, CLASS ACCURACY: {:6.2f}'.format(
            name, int(row['sum']), int(row['count']), row['sum'] / row['count'] * 100
        )
        print_to_(options.verbose, output_name + '.log', print_string)

    print_string = '#' * 80 + '\n CLASS MEAN ACCURACY: {:>4.2f}'.format(
        (df_class['sum'] / df_class['count']).mean() * 100
    )
    print_to_(options.verbose, output_name + '.log', print_string)


def metric_on_perceptual(options):

    output_name = options.output_name + '_' + str(options.border) + \
                  '_' + str(options.val_set) + '_' + str(options.metric_type)

    if options.output_type == 'align':
        raise Exception('Perceptual is zero!')

    # LOAD list of all aligned scenes
    is_zhores = False
    paths = glob.glob(os.path.join(options.input_dir, 'scene*/*'))
    if len(paths) == 0:
        paths = glob.glob(os.path.join(options.input_dir, '*/scene*/*'))
        is_zhores = True

    if options.verbose:
        paths = tqdm(paths)

    rows = []
    for p in paths:
        scan_id = p.split('/')[-2]
        if is_zhores:
            align, method = p.split('_output_')[-1].split('/')[0].split('_50_')
        else:
            method, align = p.split('/')[-3].split('_output_')

        if len(p.split('/')[-1].split('_')) == 3:
            category_id, shape_id, object_num = p.split('/')[-1].split('_')
        else:
            category_id, shape_id = p.split('/')[-1].split('_')
            object_num = -1

        init = os.path.join(p, 'init.obj')
        if not os.path.exists(init):
            continue

        init_mesh = get_init_mesh(scan_id, category_id + '_' + shape_id)
        if init_mesh is None:
            continue
        approx_mesh = trimesh.load_mesh(os.path.join(p, 'approx.obj'), process=False)

        dame = np.array(DAME(init_mesh, approx_mesh, 0.59213)[1])
        dame = dame[~pd.isna(dame)].mean()

        rows.append([
            scan_id,
            category_id + '_' + shape_id,  # key
            category_id,
            shape_id,
            object_num,
            p,  # path
            dame
        ])

    cols = ['scan_id', 'key', 'objectCategory', 'alignedModelId', 'object_num', 'path', 'DAME']
    df_final = pd.DataFrame(rows, columns=cols)
    df_final.to_csv(output_name + '.csv')

    if len(df_final) == 0:
        print_to_(options.verbose, output_name + '.log', 'No aligned shapes')
        return

    df_final = df_final[~pd.isna(df_final['DAME'])]

    # Calculate INSTANCE accuracy
    dame = df_final['DAME'].mean()

    print_string = '#' * 57 + '\nINSTANCE MEAN. DAME: {:>4.2f}\n'.format(dame) + '#' * 57
    print_to_(options.verbose, output_name + '.log', print_string)

    df_final['name'] = [top_classes.get(x, 'zother') for x in df_final.objectCategory]
    df_class = df_final.groupby('name').mean()[['DAME']]

    print_string = '###' + ' ' * 7 + 'CLASS' + ' ' * 4 + '#   DAME   ###'
    print_to_(options.verbose, output_name + '.log', print_string)

    for name, row in df_class.iterrows():
        print_string = '###\t{:10} # {:>4.2f}   ###'.format(name, row['DAME'])
        print_to_(options.verbose, output_name + '.log', print_string)

    dame_class = df_class['DAME'].mean()
    print_string = '#' * 57 + '\n   CLASS MEAN. DAME: {:>4.2f}\n'.format(dame_class) + '#' * 57
    print_to_(options.verbose, output_name + '.log', print_string)


def main(options):
    if options.metric_type == 'align':
        metric_on_alignment(options)
    elif options.metric_type == 'deform':
        metric_on_deformation(options)
    else:
        metric_on_perceptual(options)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input-dir', dest='input_dir', default='./Validation/Align-output/',
                        help='Directory with results.')
    parser.add_argument('-o', '--output-name', dest='output_name', default='./Validation/Metric/align_metric',
                        help='Output file prefix with metric calculations.')

    parser.add_argument('-m', '--metric-type', dest='metric_type',
                        choices=['align', 'deform', 'perceptual'], default='deform', help='Metric type.')
    parser.add_argument('-n', '--val-set', dest='val_set',
                        choices=['50', '2k', 'all'], default='2k', help='set of scenes to calculate.')
    parser.add_argument('-t', '--output-type', dest='output_type',
                        choices=['align', 'deform'], default='deform', help='Algorithm output type.')
    parser.add_argument('-b', '--border', dest='border', type=float, default=0.0,
                        help='Border for max distance in deformation metrics.')

    parser.add_argument('-v', '--verbose', action='store_true',
                        default=False, help='Print verbose output.')

    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)

