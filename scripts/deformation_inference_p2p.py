#!/usr/bin/env python3

import argparse
import json
import os
import pickle
import sys

from joblib import Parallel, delayed
import numpy as np
import torch
import trimesh

__dir__ = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..')
)

sys.path[1:1] = [__dir__]

from shapefit.deform.src.deformation.deformation import Deformer
from shapefit.deform.src.data_proc.voxels import read_shapenet_voxels
from shapefit.deform.src.deformation.utils import (
    set_new_mesh_vertices,
    add_noise_to_mesh,
    mesh_pcloud
)
from shapefit.deform.src.deformation.preproc import find_sharp_edges, split_vertices_by_parts, transform_voxels_to_origin, \
    filter_voxels_by_clustering, neighboring_scene_voxel_parts, filter_scene_voxel_parts_with_obb


def process_mesh(data_object, shapenet_path, scene_id, sharp_edges,
                 sharp_vertices, alpha_0, alpha_reg, alpha_sharp, alpha_data_nn,
                 alpha_data_p2p, iterations, output_dir):
    '''
    open remeshed mesh
    '''
    MESH_REMESHED = os.path.join(shapenet_path, data_object['shapenet_id'], data_object['object_id'],
                                 'models/model_normalized_remeshed_5.obj')
    mesh_remeshed = trimesh.load(MESH_REMESHED)
    transform = data_object['transform']
    '''
    open parts labeling
    '''
    with open(os.path.join(shapenet_path,
                           data_object['shapenet_id'],
                           data_object['object_id'],
                           'models/partnet_map_5.pkl'), 'rb') as read_file:
        partnet_map = pickle.load(read_file)
    all_parts = sorted(list(set([partnet_map[k][0] for k in partnet_map])))
    cmap = {}
    for part in all_parts:
        cmap[part] = np.zeros(4).astype(int)
        cmap[part][:3] = np.random.randint(0, 255, (3,))
    for k in partnet_map:
        mesh_remeshed.visual.vertex_colors[k] = cmap[partnet_map[k][0]]
    '''
    find sharp edges
    '''
    part_sharp_edges_ids = find_sharp_edges(mesh_remeshed, data_object, sharp_edges, partnet_map)
    '''
    split all vertices by parts
    '''
    parts_idx = split_vertices_by_parts(all_parts, partnet_map)
    '''
    collect scan mesh eps-neighborhood vertices
    '''
    voxel_centers_p2p, surface_samples_p2p = transform_voxels_to_origin(data_object, all_parts, transform)
    '''
    make voxel clouds filtering
    '''
    voxel_centers_p2p = filter_voxels_by_clustering(voxel_centers_p2p)
    '''
    define neighboring point clouds
    '''
    neighboring_voxel_centers_ids = neighboring_scene_voxel_parts(voxel_centers_p2p)
    '''
    filter voxel centers by neighboring bounding boxes
    '''
    voxel_centers_p2p = filter_scene_voxel_parts_with_obb(voxel_centers_p2p, neighboring_voxel_centers_ids)
    '''
    visualisation
    '''
    part_clouds = []
    for i, part in enumerate(all_parts):
        pcloud_color = cmap[part]
        if len(voxel_centers_p2p[i]) > 0:
            pcloud = mesh_pcloud(voxel_centers_p2p[i], size=0.005, color=pcloud_color)
            part_clouds += [pcloud]
    '''
    prepare input for deformation
    '''
    parts = [mesh_remeshed]
    starting_vertices_transforms = [torch.Tensor(np.repeat(np.eye(4)[None, :, :], len(part.vertices), axis=0)) for part
                                    in parts]
    starting_vertices_transforms = [torch.cat(starting_vertices_transforms, dim=0)]
    initial_vertices_transforms = [torch.Tensor(np.repeat(np.eye(4)[None, :, :], len(parts[0].vertices), axis=0)) for
                                   part in parts]
    initial_vertices_transforms = [torch.cat(initial_vertices_transforms, dim=0)]

    noise = np.random.normal(size=(len(mesh_remeshed.vertices), 3), scale=0.00001)
    noisy_part = add_noise_to_mesh(mesh_remeshed, noise)
    noisy_parts = [noisy_part]

    torch.set_num_threads(4)
    if noisy_parts[0].is_watertight and sum([len(x) for x in voxel_centers_p2p]) > 0:
        deformer = Deformer(noisy_parts,
                            initial_vertices_transforms,
                            starting_vertices_transforms,
                            sigma=0.07,
                            method='bitriangles',
                            part_vertex_indices=None,
                            voxel_centers_nn=None,
                            surface_samples_nn=None,
                            voxel_centers_p2p=voxel_centers_p2p,
                            surface_samples_p2p=surface_samples_p2p,
                            part_sharp_edges_ids=part_sharp_edges_ids,
                            kernel='ep_kernel',
                            mapping='p2p',
                            cuda=None,
                            device_mode='cpu',
                            deg_thr=1e-10,
                            ampl_factor=10)
        '''
        deform
        '''
        deformer.solve_data(alpha_0=alpha_0,
                            alpha_reg=alpha_reg,
                            alpha_sharp=alpha_sharp,
                            alpha_data_nn=alpha_data_nn,
                            alpha_data_p2p=alpha_data_p2p,
                            alpha_quad=1,
                            iterations=iterations,
                            lr=1e-0,
                            print_freq=5,
                            use_precond=True,
                            hessian_cpu=True,
                            load_hessian_path=None,
                            plot=False)
        '''
        prepare data for saving
        '''
        first_approximation = deformer.get_first_approximation()
        initial_vertices = deformer.get_initial_parts_vertices()
        initial_shape = []
        for i in range(len(initial_vertices)):
            initial_mesh = set_new_mesh_vertices(noisy_parts[i], initial_vertices[i])
            initial_shape += [initial_mesh]
        initial_shape = trimesh.util.concatenate(initial_shape)
        approximation_shape = []
        for i in range(len(first_approximation)):
            approximation_mesh = set_new_mesh_vertices(noisy_parts[i], first_approximation[i])
            approximation_shape += [approximation_mesh]
        approximation_shape = trimesh.util.concatenate(approximation_shape)
        '''
        save
        '''
        save_params = {}
        save_params["alpha_0"] = alpha_0
        save_params["alpha_reg"] = alpha_reg
        save_params["alpha_sharp"] = alpha_sharp
        save_params["alpha_data_nn"] = alpha_data_nn
        save_params["alpha_data_p2p"] = alpha_data_p2p
        save_params["iterations"] = iterations
        save_params["bbox_transform"] = False
        SAVE_DIR = os.path.join(output_dir, scene_id,
                                '{}_{}_{}'.format(data_object['shapenet_id'], data_object['object_id'], data_object['shape_id']))
        os.makedirs(SAVE_DIR, exist_ok=True)
        approximation_shape.export(os.path.join(SAVE_DIR, 'approx.obj'))
        initial_shape.export(os.path.join(SAVE_DIR, 'init.obj'))
        trimesh.util.concatenate(part_clouds).export(os.path.join(SAVE_DIR, 'cloud.obj'))
        with open(os.path.join(output_dir, 'params.json'), 'w') as writefile:
            json.dump(save_params, writefile)

        transform_param = {}
        transform_param['transform'] = data_object['transform'].tolist()
        with open(os.path.join(SAVE_DIR, 'transform.json'), 'w') as writefile:
            json.dump(transform_param, writefile)

        print('Processed mesh, results at {}'.format(SAVE_DIR))


def main(options):

    with open(options.sharp_edges_path, 'r') as readfile:
        sharp_edges = json.load(readfile)
    with open(options.sharp_vertices_path, 'r') as readfile:
        sharp_vertices = json.load(readfile)

    scene_voxels = read_shapenet_voxels(options.input_dir, options.scene_id)

    params = (options.shapenet_path, options.scene_id, sharp_edges,
              sharp_vertices, options.alpha_0, options.alpha_reg, options.alpha_sharp, options.alpha_data_nn,
              options.alpha_data_p2p, options.iterations, options.output_dir)

    parallel = Parallel(n_jobs=options.n_jobs, backend='multiprocessing')
    delayed_iterable = (delayed(process_mesh)(data_object, *params) for data_object in scene_voxels)
    parallel(delayed_iterable)


def parse_args():
    parser = argparse.ArgumentParser()
    
    # alphas
    parser.add_argument('-a', '--alpha-0', dest='alpha_0', default='1.0', type=float, required=True, help='Alpha_0 for deformer')
    parser.add_argument('-b', '--alpha-reg', dest='alpha_reg', default='1.0', type=float, required=True, help='Alpha_reg for deformer')
    parser.add_argument('-c', '--alpha-sharp', dest='alpha_sharp', default='1.0', type=float, required=True, help='Alpha_sharp for deformer')
    parser.add_argument('-d', '--alpha-data-nn', dest='alpha_data_nn', default='1.0', type=float, required=True, help='Alpha_data_nn for deformer')
    parser.add_argument('-e', '--alpha-data-p2p', dest='alpha_data_p2p', default='1.0', type=float, required=True, help='Alpha_data_p2p for deformer')
    
    # other params
    parser.add_argument('-r', '--iterations', dest='iterations', default=50, type=int, help='Number of iterations')
    parser.add_argument('-t', '--bbox-transform', dest='make_bbox_transform', default=False, type=bool, help='Flag for applying bbox transform')
    parser.add_argument('-g', '--cuda-device', dest='cuda_device', default=None, type=int, help='Cuda device # or None')
    
    # locations
    parser.add_argument('-i', '--input-dir', dest='input_dir', required=True, help='Path to data dir')
    parser.add_argument('-z', '--scene-id', dest='scene_id', required=True, help='Path to scene id')
    parser.add_argument('-o', '--output-dir', dest='output_dir', required=True, help='Save dir for deformations')
    parser.add_argument('-w', '--shapenet', dest='shapenet_path', required=True, help='Path to ShapeNet')
    parser.add_argument('-k', '--sharp-edges', dest='sharp_edges_path', required=True, help='Path to json with sharp edges')
    parser.add_argument('-v', '--sharp-vertices', dest='sharp_vertices_path', required=True, help='Path to json with sharp vertices')

    parser.add_argument('-j', '--jobs', dest='n_jobs',
                        type=int, default=4, help='CPU jobs to use in parallel [default: 4].')

    parser.add_argument('--verbose', dest='verbose', default=False, action='store_true', help='be verbose')

    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)