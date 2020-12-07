import numpy as np
import os
import pickle
from shapefit.utils.utils import get_part_ids
from shapefit.deform.src.deformation.utils import make_M_from_tqs


def read_shapenet_voxels(data_path, scene_id, merge_part_voxels=True):
    
    parts = pickle.load(open(os.path.join(data_path, scene_id + '.pkl'), 'rb'))
    global_part_ids = get_part_ids()

    shapenet_instances = {}

    for part in parts:
        global_id = part['global_id']
        part_info = global_part_ids.loc[global_id]
        transformation = part['transformation']
        shape_id = part['shape_id']
        object_id = part_info['object_id']

        if shape_id not in shapenet_instances:
            shapenet_instances[shape_id] = {}
            shapenet_instances[shape_id]['parts_voxels'] = {}
            shapenet_instances[shape_id]['parts_voxels_o2o'] = {}
            shapenet_instances[shape_id]['parts_vertices_p2p'] = {}
            shapenet_instances[shape_id]['shapenet_name'] = part_info['category_name_from_shapenet']
            shapenet_instances[shape_id]['scan_id'] = part['scan_id']
            shapenet_instances[shape_id]['partnet_id'] = part_info['object_partnet_id']
            shapenet_instances[shape_id]['shapenet_id'] = part_info['category_id']
            shapenet_instances[shape_id]['voxels'] = []
            shapenet_instances[shape_id]['part_dir_names'] = []
            shapenet_instances[shape_id]['mesh_paths'] = []
            shapenet_instances[shape_id]['object_id'] = object_id
        shapenet_instances[shape_id]['part_dir_names'].append(part_info['part_dir_name'])
        shapenet_instances[shape_id]['voxels'].append(np.array(part['scan_voxels']))
        shapenet_instances[shape_id]['parts_voxels'][part_info['part_dir_name']] = np.array(part['scan_voxels'])
        shapenet_instances[shape_id]['parts_vertices_p2p'][part_info['part_dir_name']] = np.array(
            part['scan_mesh_vertices'])
        shapenet_instances[shape_id]['mesh_paths'].append(part['path2mesh'])

        shapenet_instances[shape_id]['transform'] = make_M_from_tqs(transformation[:3], transformation[3:7],
                                                                    transformation[7:])

    if merge_part_voxels:
        for shape_id in shapenet_instances:
            shapenet_instances[shape_id]['voxels'] = np.vstack(shapenet_instances[shape_id]['voxels'])

    shape_list = []
    num_shape_instances = {}
    for shape_id in shapenet_instances:
        object_id = shapenet_instances[shape_id]['object_id']
        if object_id not in num_shape_instances:
            num_shape_instances[object_id] = 1
        else:
            num_shape_instances[object_id] += 1
        shape_list += [{'shape_id': shape_id, **shapenet_instances[shape_id]}]
    for shape in shape_list:
        object_id = shape['object_id']
        if num_shape_instances[object_id] > 1:
            shape['is_multiple'] = 1
        else:
            shape['is_multiple'] = 0
            
    return shape_list