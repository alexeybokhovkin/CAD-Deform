import numpy as np
import trimesh
from trimesh.grouping import clusters

from .utils import filter_edges_by_parts


def compute_bitriangles(mesh_unique_faces, mesh_unique_edges):
    bitriangles = {}
    for face in mesh_unique_faces:
        edge_1 = tuple(sorted([face[0], face[1]]))
        if edge_1 not in bitriangles:
            bitriangles[edge_1] = {face[0], face[1], face[2]}
        else:
            bitriangles[edge_1].add(face[2])
        edge_2 = tuple(sorted([face[1], face[2]]))
        if edge_2 not in bitriangles:
            bitriangles[edge_2] = {face[0], face[1], face[2]}
        else:
            bitriangles[edge_2].add(face[0])
        edge_3 = tuple(sorted([face[0], face[2]]))
        if edge_3 not in bitriangles:
            bitriangles[edge_3] = {face[0], face[1], face[2]}
        else:
            bitriangles[edge_3].add(face[1])
    bitriangles_aligned = np.empty((len(mesh_unique_edges), 4), dtype=int)
    for j, edge in enumerate(mesh_unique_edges):
        bitriangle = [*sorted(edge)]
        bitriangle += [x for x in list(bitriangles[tuple(sorted(edge))]) if x not in bitriangle]
        bitriangles_aligned[j] = bitriangle

    return bitriangles_aligned

def level_merger(data_object, partnet_map, part_levels, global_part_ids, level='map_0k_8'):
    data_object_merged = data_object.copy()
    partnet_map_merged = partnet_map.copy()
    object_part_ids = global_part_ids[global_part_ids['object_id'] == data_object_merged['object_id']]
    part_to_global_id = {}
    for index, row in object_part_ids.iterrows():
        part_to_global_id[index] = row['part_dir_name']
    global_id_to_merge = {}
    for global_id in part_to_global_id.keys():
        global_id_to_merge[global_id] = part_levels.iloc[global_id][level]
    part_name_to_merge = {}
    for global_id in global_id_to_merge.keys():
        if global_id_to_merge[global_id] not in part_name_to_merge:
            part_name_to_merge[global_id_to_merge[global_id]] = []
        part_name_to_merge[global_id_to_merge[global_id]] += [part_to_global_id[global_id]]
    new_parts_list = list(part_name_to_merge.values())
    new_parts_dict = {}
    old_parts_to_new = {}
    for i, new_part in enumerate(new_parts_list):
        new_part_points = []
        old_parts = []
        for old_part in new_part:
            old_parts += [old_part]
            if old_part in data_object_merged['parts_vertices_p2p']:
                new_part_points += [data_object_merged['parts_vertices_p2p'][old_part]]
        if len(new_part_points) > 0:
            new_part_points = np.vstack(new_part_points)
        new_parts_dict['merged-new-{}'.format(i)] = new_part_points
        for old_part in old_parts:
            old_parts_to_new[old_part] = 'merged-new-{}'.format(i)
    data_object_merged['parts_vertices_p2p'] = new_parts_dict

    for key in partnet_map_merged:
        old_part = partnet_map_merged[key][0].split('.')[0]
        partnet_map_merged[key] = (old_parts_to_new[old_part] + '.obj',
                                   partnet_map_merged[key][1],
                                   partnet_map_merged[key][2])
    return data_object_merged, partnet_map_merged


def find_sharp_edges(mesh, data_object, sharp_edges, partnet_map):
    part_sharp_edges_ids = None
    if data_object['shapenet_id'] in sharp_edges:
        if data_object['object_id'] in sharp_edges[data_object['shapenet_id']]:
            sharp_edges_for_mesh = sharp_edges[data_object['shapenet_id']][data_object['object_id']]

            non_conflict_edges, conflict_edges = filter_edges_by_parts(sharp_edges_for_mesh, partnet_map)

            unique_edges_to_vertices = {i: list(x) for i, x in enumerate(mesh.edges_unique)}
            vertices_to_unique_edges = {tuple(unique_edges_to_vertices[i]): i for i in unique_edges_to_vertices}

            part_sharp_edges_ids = []
            for part_id in non_conflict_edges:
                part_edges = non_conflict_edges[part_id]
                edges_ids = []
                for edge in part_edges:
                    try:
                        if tuple(edge) in vertices_to_unique_edges:
                            edges_ids += [vertices_to_unique_edges[tuple(edge)]]
                        else:
                            edges_ids += [vertices_to_unique_edges[tuple(edge[::-1])]]
                    except:
                        continue
                part_sharp_edges_ids += [edges_ids]
    return part_sharp_edges_ids


def split_vertices_by_parts(part_names, partnet_map):
    parts_idx = [[] for _ in range(len(part_names))]
    for k in partnet_map:
        for i, part in enumerate(part_names):
            if partnet_map[k][0] == part:
                parts_idx[i] += [k]
    return parts_idx


def transform_voxels_to_origin(data_object, all_parts, transform, parts_idx):
    voxel_centers_p2p = []
    surface_samples_p2p = []
    for i, part in enumerate(all_parts):
        part_samples_ids = parts_idx[i]
        surface_samples_p2p += [part_samples_ids]

        points = data_object['parts_vertices_p2p'][part.split('.')[0]]
        points = np.hstack([points, np.ones(len(points))[:, None]])
        points = (points @ np.linalg.inv(transform).T)[:, :3]
        voxel_centers_p2p += [points]
    return voxel_centers_p2p, surface_samples_p2p


def filter_voxels_by_clustering(voxel_centers):
    voxel_centers_new = []
    for points in voxel_centers:
        if len(points) > 0:
            if len(points) != 1:
                groups = clusters(points, 0.1)
            else:
                groups = np.array([[0]])
            groups_lens = [len(group) for group in groups]
            if len(groups_lens) == 0:
                voxel_centers_new += [[]]
            else:
                max_group_id = np.argmax(groups_lens)
                new_group = groups[max_group_id]
                voxel_centers_new += [points[new_group]]
        else:
            voxel_centers_new += [[]]
    return voxel_centers_new


def neighboring_scene_voxel_parts(voxel_centers):
    neighboring_voxel_centers_ids = []
    for i, points_i in enumerate(voxel_centers):
        for j, points_j in enumerate(voxel_centers):
            if j > i and len(points_i) > 0 and len(points_j) > 0:
                min_dist = np.min(np.sum((points_i[None, ...] - points_j[:, None, :]) ** 2, axis=2))
                if min_dist < 0.05:
                    neighboring_voxel_centers_ids += [(i, j)]
    return neighboring_voxel_centers_ids


def filter_scene_voxel_parts_with_obb(voxel_centers, neighboring_voxel_centers_ids):
    point_clouds = []
    for i, points in enumerate(voxel_centers):
        if len(points) > 0:
            point_clouds += [trimesh.points.PointCloud(points)]
        else:
            point_clouds += [[]]
    neighbors_to_merge = []
    for neighbors in neighboring_voxel_centers_ids:
        try:
            if point_clouds[neighbors[0]] != [] and point_clouds[neighbors[1]] != []:
                bbox_1 = point_clouds[neighbors[0]].bounding_box_oriented
                bbox_2 = point_clouds[neighbors[1]].bounding_box_oriented
                point_cloud_merge = trimesh.points.PointCloud(np.vstack([voxel_centers[neighbors[0]],
                                                                         voxel_centers[neighbors[1]]]))
                bbox_merge = point_cloud_merge.bounding_box_oriented
                volume_1 = bbox_1.volume
                volume_2 = bbox_2.volume
                volume_merge = bbox_merge.volume

                max_volume = max(volume_1, volume_2)
                if volume_merge / max_volume < 1.3:
                    if volume_1 > volume_2:
                        major = neighbors[0]
                        minor = neighbors[1]
                    else:
                        major = neighbors[1]
                        minor = neighbors[0]
                    neighbors_to_merge += [(major, minor)]
        except:
            continue
    for neighbors in neighbors_to_merge:
        if len(voxel_centers[neighbors[1]]) > 0 and len(voxel_centers[neighbors[0]]) > 0:
            voxel_centers[neighbors[0]] = np.vstack([voxel_centers[neighbors[0]],
                                                         voxel_centers[neighbors[1]]])
            voxel_centers[neighbors[1]] = []
    return voxel_centers


def find_corresondences_with_obb(voxel_centers, mesh, parts_idx, make_bbox_transform=False):
    bbox_transforms = []
    bboxes_vertices = []
    bboxes_voxels = []
    if make_bbox_transform:
        for i, points in enumerate(voxel_centers):
            try:
                vertices = mesh.vertices[parts_idx[i]]
                if len(points) != 0:
                    bbox_vertices = trimesh.points.PointCloud(vertices).bounding_box_oriented
                    bbox_voxels = trimesh.points.PointCloud(points).bounding_box_oriented
                    bboxes_vertices += [bbox_vertices.vertices]
                    bboxes_voxels += [bbox_voxels.vertices]

                    vertices_box_vicinities = [[i] for i in range(len(bbox_vertices.vertices))]
                    noncorrect_edges = []
                    for facet in bbox_vertices.facets:
                        face_1 = bbox_vertices.faces[facet[0]]
                        face_2 = bbox_vertices.faces[facet[1]]
                        intersection = list(set(face_1).intersection(set(face_2)))
                        noncorrect_edges += [intersection]
                    noncorrect_edges = np.sort(np.array(noncorrect_edges), axis=1)
                    noncorrect_edges = [tuple(x) for x in noncorrect_edges]
                    all_edges = np.sort(bbox_vertices.edges_unique, axis=1)
                    all_edges = [tuple(x) for x in all_edges]
                    correct_edges = [x for x in all_edges if x not in noncorrect_edges]
                    for edge in correct_edges:
                        vertices_box_vicinities[edge[0]] += [edge[1]]
                        vertices_box_vicinities[edge[1]] += [edge[0]]
                    anchor_vertices_vicinity = vertices_box_vicinities[0]

                    voxels_box_vicinities = [[i] for i in range(len(bbox_voxels.vertices))]
                    noncorrect_edges = []
                    for facet in bbox_voxels.facets:
                        face_1 = bbox_voxels.faces[facet[0]]
                        face_2 = bbox_voxels.faces[facet[1]]
                        intersection = list(set(face_1).intersection(set(face_2)))
                        noncorrect_edges += [intersection]
                    noncorrect_edges = np.sort(np.array(noncorrect_edges), axis=1)
                    noncorrect_edges = [tuple(x) for x in noncorrect_edges]
                    all_edges = np.sort(bbox_voxels.edges_unique, axis=1)
                    all_edges = [tuple(x) for x in all_edges]
                    correct_edges = [x for x in all_edges if x not in noncorrect_edges]
                    for edge in correct_edges:
                        voxels_box_vicinities[edge[0]] += [edge[1]]
                        voxels_box_vicinities[edge[1]] += [edge[0]]

                    voxels_full_vicinities = []
                    for vicinity in voxels_box_vicinities:
                        three_other_vertices = vicinity[1:]
                        voxels_full_vicinities += [vicinity]
                        voxels_full_vicinities += [
                            [vicinity[0], three_other_vertices[1], three_other_vertices[2], three_other_vertices[0]]]
                        voxels_full_vicinities += [
                            [vicinity[0], three_other_vertices[2], three_other_vertices[0], three_other_vertices[1]]]

                    best_dist = 100
                    vertices_target = np.array(bbox_vertices.vertices[anchor_vertices_vicinity])
                    vertices_target = np.hstack([vertices_target, np.ones(len(vertices_target))[:, None]])
                    for vicinity in voxels_full_vicinities:
                        voxels_source = np.array(bbox_voxels.vertices[vicinity])
                        voxels_source = np.hstack([voxels_source, np.ones(len(voxels_source))[:, None]])
                        transform = np.linalg.inv(voxels_source) @ vertices_target
                        if transform[0, 0] > 0 and transform[1, 1] > 0 and transform[2, 2] > 0:
                            dist = np.sum((transform[:3, :3] - np.eye(3)) ** 2)
                            if dist < best_dist:
                                best_dist = dist
                    if best_dist < 0:
                        # choose transform here or np.eye(4)
                        bbox_transforms += [transform]
                    else:
                        bbox_transforms += [np.eye(4)]
                else:
                    bbox_transforms += [np.eye(4)]
            except:
                bbox_transforms += [np.eye(4)]
    else:
        bbox_transforms = [np.eye(4) for _ in voxel_centers]

    min_init_dists = []
    min_transformed_dists = []
    mesh_vertices_nn = []
    voxel_centers_nn = []
    parts_vertices = []
    parts_voxels = []
    parts_voxels_transformed = []
    for i, points in enumerate(voxel_centers):
        if len(points) != 0:
            vertices = mesh.vertices[parts_idx[i]]
            parts_vertices += [vertices]

            voxels = np.hstack([points, np.ones(len(points))[:, None]])
            parts_voxels += [voxels[:, :3]]

            voxels_transformed = (voxels @ bbox_transforms[i])[:, :3]
            parts_voxels_transformed += [voxels_transformed]
            vertices_idx = []
            for j, point in enumerate(voxels_transformed):
                dists = np.sum((vertices - point) ** 2, axis=1)
                min_vertex_id = np.argmin(dists)
                min_init_dist = np.sum((mesh.vertices[parts_idx[i]][min_vertex_id] - voxels[j][:3]) ** 2)
                if min_init_dist < 0.01:
                    min_init_dists += [min_init_dist]
                    min_transformed_dists += [min(dists)]
                    vertices_idx += [parts_idx[i][min_vertex_id]]
                    voxel_centers_nn += [voxels[j][:3]]
            mesh_vertices_nn += vertices_idx
    return voxel_centers_nn, mesh_vertices_nn
