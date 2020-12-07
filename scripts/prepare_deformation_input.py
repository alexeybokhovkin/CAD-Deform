#!/usr/bin/env python3

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
# Deformation input structure
########################################################################
#
# ./Validation/Deformation-input-XXX/
# 	{scan_id}.pkl: array of parts of aligned objects. For each part we have a dictionary:
#       'scan_id':                              ID of a Scannet scene
#       'shape_id':                             id of Shapenet instance on the scene
#       'global_id':                            ID of a PartNet object
#       'path2vox', 'path2mesh':                paths to voxels and mesh of th part
#       'transformation':                       Parts alignments
#       'scan_keypoints', 'part_modes':         Correspondences that are used for alignment.
#                                               For part shapes we have a mode of a distribution.
#       'scan_voxels', 'scan_mesh_vertices':    Voxels and vertices near the aligned object
#       'unlabelled_scan_vertices_via_shape':   Scan vertices inside shape's oriented bbox
#                                               that are not included into 'scan_mesh_vertices'
#
########################################################################

import argparse
import json
import numpy as np
import os
import pandas as pd
import sys
import trimesh
import quaternion
import pickle

from scipy.spatial.distance import cdist

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..')
)

sys.path[1:1] = [__dir__]

from shapefit.utils.utils import make_M_from_tqs, make_tqs_from_M, get_shapes_in_scan2cad_parts, get_part_ids
from shapefit.utils.vox_loader import load_sample, vox2obj


def get_mode(heatmap_dir):
    heatmap = load_sample(heatmap_dir)
    idx = np.argmax(heatmap.pdf)
    idx_x = idx // (heatmap.dimx * heatmap.dimy)
    idx -= idx_x * heatmap.dimx * heatmap.dimy
    min_idx = idx % heatmap.dimx, idx // heatmap.dimx, idx_x

    return min_idx


def main(options):
    scan_id = options.scan_id
    df_label = get_part_ids()

    # LOAD scene
    scan_vox_path = os.path.join(options.data_dir, 'scannet-voxelized-sdf', scan_id, scan_id + '.vox')
    scan_vox = load_sample(scan_vox_path)
    scan_obj = vox2obj(scan_vox)[:, :3]

    scan_mesh_path = os.path.join(options.data_dir, 'scannet', scan_id, scan_id + '_vh_clean_2.ply')
    scan_mesh = trimesh.load_mesh(scan_mesh_path)
    scan_mesh = np.array(scan_mesh.vertices)

    if options.is_gt:
        full_anno = json.load(open(os.path.join(options.full_anno_dir, 'full_annotations.json'), 'rb'))
        scan_anno = [x for x in full_anno if x['id_scan'] == scan_id][0]

        scan_trs = make_M_from_tqs(
            scan_anno['trs']['translation'],
            scan_anno['trs']['rotation'],
            scan_anno['trs']['scale']
        )

        partnet_dict = get_shapes_in_scan2cad_parts()
        batch = enumerate([x for x in scan_anno['aligned_models'] if x['id_cad'] in partnet_dict[:, :2]])
    else:
        scan_path = os.path.join(options.input_dir, scan_id + '.csv')
        df = pd.read_csv(scan_path, index_col=0, dtype={'objectCategory': str})

        batch = df.iterrows()

    json_output = []
    for i, obj_anno in batch:
        print("Processing object #{}".format(i))
        result = {
            'scan_id': scan_id,
            'shape_id': i
        }

        if not options.is_gt:
            category_id, shape_id = obj_anno['objectCategory'], obj_anno['alignedModelId']

            M_total = make_M_from_tqs(
                obj_anno[['tx', 'ty', 'tz']],
                obj_anno[['qw', 'qx', 'qy', 'qz']],
                obj_anno[['sx', 'sy', 'sz']]
            )

            transformation_cols = ['tx', 'ty', 'tz', 'qw', 'qx', 'qy', 'qz', 'sx', 'sy', 'sz']
            result['transformation'] = obj_anno[transformation_cols].values.astype(float)
        else:
            category_id, shape_id = obj_anno['catid_cad'], obj_anno['id_cad']

            shape_trs = make_M_from_tqs(
                obj_anno['trs']['translation'],
                obj_anno['trs']['rotation'],
                obj_anno['trs']['scale']
            )
            M_total = np.linalg.inv(scan_trs) @ shape_trs
            t, q, s = make_tqs_from_M(M_total)
            q = quaternion.as_float_array(q)

            result['transformation'] = np.concatenate((t, q, s))

        # keypoint <-> heatmap pairs
        pairs = json.load(open(os.path.join(
            options.align_input_dir, scan_id, category_id + '_' + shape_id, 'input.json'
        ), 'rb'))

        # get keypoints
        keypoints = np.array([p['p_scan'] for p in pairs])

        # get heatmaps' mode
        heatmap_dirs = [os.path.join(
            options.align_input_dir, scan_id, category_id + '_' + shape_id, p['heatmap']
        ) for p in pairs]
        modes = np.array([get_mode(x) for x in heatmap_dirs])
        part_names = ['-'.join(x.split('-')[-4:-2]) for x in heatmap_dirs]

        df_shape = df_label[
            (df_label.category_id == category_id) &
            (df_label.object_id == shape_id)
        ]
        json_of_shapes = []

        for global_id, part_info in df_shape.iterrows():
            result_part = result.copy()
            result_part['global_id'] = global_id

            result_part['path2vox'] = os.path.join(
                options.data_dir, 'partnet-voxelized-df', category_id,
                shape_id + '__' + part_info['part_dir_name'] + '__0__.df'
            )
            result_part['path2mesh'] = os.path.join(
                options.data_dir, 'partnet', part_info['object_partnet_id'],
                'objs-normalized', part_info['part_dir_name'] + '.obj'
            )

            part_vox = load_sample(result_part['path2vox'])

            # Keypoints and Modes
            kps = np.array([
                k for k, hm, name in zip(keypoints, modes, part_names) if name == part_info['part_dir_name']
            ])
            mds = np.array([
                hm for k, hm, name in zip(keypoints, modes, part_names) if name == part_info['part_dir_name']
            ])

            if len(mds) == 0:
                result_part['scan_keypoints'] = []
                result_part['part_modes'] = []
            else:
                mds = np.hstack((mds, np.ones((len(mds), 1))))
                mds = mds @ part_vox.grid2world.T

                dist_from_modes_to_kps = np.mean(((mds @ M_total.T)[:, :3] - kps) ** 2, axis=1)
                result_part['scan_keypoints'] = [
                    list(k) for k, m, d in zip(kps, mds, dist_from_modes_to_kps) if d < options.around_area
                ]
                result_part['part_modes'] = [
                    list(m) for k, m, d in zip(kps, mds, dist_from_modes_to_kps) if d < options.around_area
                ]

            part_obj = vox2obj(part_vox) @ M_total.T
            part_obj = part_obj[:, :3]

            dist = cdist(scan_obj, part_obj)
            nearest_voxels = scan_obj[dist.min(-1) < options.around_area / 3]
            result_part['scan_voxels'] = nearest_voxels

            # get only near scan vertices
            min_, max_ = part_obj.min(0) - options.around_area / 3, part_obj.max(0) + options.around_area / 3
            scan_sub_mesh = scan_mesh[(scan_mesh[:, 0] >= min_[0]) & (scan_mesh[:, 0] <= max_[0]) &
                                      (scan_mesh[:, 1] >= min_[1]) & (scan_mesh[:, 1] <= max_[1]) &
                                      (scan_mesh[:, 2] >= min_[2]) & (scan_mesh[:, 2] <= max_[2])]

            dist = cdist(scan_sub_mesh, part_obj)
            nearest_vertex = scan_sub_mesh[dist.min(-1) < options.around_area / 3]
            result_part['scan_mesh_vertices'] = nearest_vertex

            json_of_shapes.append(result_part)

        # load object's mesh
        obj_path = os.path.join(options.data_dir, 'shapenet', category_id, shape_id,
                                'models/model_normalized_remeshed_4.obj')
        obj_mesh = trimesh.load_mesh(obj_path)
        if isinstance(obj_mesh, trimesh.Scene) and len(obj_mesh.geometry) == 0:
            obj_mesh = trimesh.load_mesh(obj_path[:-6] + '_3.obj')
        obj_mesh.apply_transform(M_total)

        obj_mesh_vertices = np.array(obj_mesh.vertices)

        # get only near scan vertices
        min_, max_ = obj_mesh_vertices.min(0) - options.around_area/3, obj_mesh_vertices.max(0) + options.around_area/3
        scan_sub_mesh = scan_mesh[(scan_mesh[:, 0] >= min_[0]) & (scan_mesh[:, 0] <= max_[0]) &
                                  (scan_mesh[:, 1] >= min_[1]) & (scan_mesh[:, 1] <= max_[1]) &
                                  (scan_mesh[:, 2] >= min_[2]) & (scan_mesh[:, 2] <= max_[2])]

        if len(scan_sub_mesh) != 0:
            # define oriented bbox and scale it by 90%
            bbox_grid = obj_mesh.bounding_box_oriented.sample_grid(step=options.around_area / 2)
            bbox_grid = (bbox_grid - bbox_grid.mean(0)) * options.bbox_scale + bbox_grid.mean(0)

            # only in the OBB area
            dist = cdist(scan_sub_mesh, bbox_grid).min(-1)
            scan_sub_mesh = scan_sub_mesh[dist <= options.around_area / 3]

            if len(scan_sub_mesh) != 0:
                # take only non-assigned vertices
                assigned_scan_vertices = np.concatenate([x['scan_mesh_vertices'] for x in json_of_shapes])
                if len(assigned_scan_vertices) != 0:
                    not_assigned = cdist(scan_sub_mesh, assigned_scan_vertices).min(-1) > options.around_area * 1e-3
                    scan_sub_mesh = scan_sub_mesh[not_assigned]

        # write to each part
        for j in json_of_shapes:
            j['unlabelled_scan_vertices_via_shape'] = scan_sub_mesh

        json_output.extend(json_of_shapes)

    pickle.dump(json_output, open(os.path.join(options.output_dir, scan_id + '.pkl'), 'wb+'))


def parse_args():
    parser = argparse.ArgumentParser()

    # Location parameters
    parser.add_argument('-i', '--input-dir', dest='input_dir', default='./Validation/Align-output/',
                        help='Directory with CSV after alignment. Only used when *is_gt*=True')
    parser.add_argument('-p', '--align-input-dir', dest='align_input_dir', default='./Validation/Align-input/',
                        help='Directory with input data for alignment.')
    parser.add_argument('-o', '--output-dir', dest='output_dir', default='./Validation/Deformation-input/',
                        help='directory used to save files for deformation.')
    parser.add_argument('-f', '--full-anno-dir', dest='full_anno_dir', default='./',
                        help='Directory with file full_annotation.json and validation set. Only used when *is_gt*=True')
    parser.add_argument('-d', '--data-dir', dest='data_dir', default='../../Assets/full/',
                        help='Directory with shapenet, scannet and partnet')

    # Init params
    parser.add_argument('-s', '--scan-id', dest='scan_id', default='scene0535_00', help='Scene ID from Scannet.')
    parser.add_argument('-g', '--is-gt', dest='is_gt', action='store_true', default=False, help='Use GT alignments.')

    # Preparation for deformation
    parser.add_argument('-a', '--around-area', dest='around_area', default=0.1, type=float,
                        help='Area of taking voxels from scan for the deformation')
    parser.add_argument('-b', '--bbox-scale', dest='bbox_scale', default=0.9, type=float,
                        help='Scale obtained bbox of the aligned object')

    # Other params
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Print verbose output.')

    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
