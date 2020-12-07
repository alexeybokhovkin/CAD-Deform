import trimesh
from trimesh.base import Trimesh
import trimesh.creation
from trimesh.remesh import subdivide_to_size
import matplotlib.tri as mtri
import numpy as np
import torch
import quaternion
import os
from tqdm import tqdm_notebook as tqdm
from matplotlib import pyplot as plt


def plot_info(history_grad_norm, history_quad_loss,
              history_smooth_loss, history_loss,
              history_p_deviation, history_p_deviation_target,
              history_p_deviation_mean, history_p_deviation_target_mean):

    plt.figure(figsize=(10, 8))
    plt.semilogy(history_grad_norm)
    plt.title('Grad norm')
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.semilogy(np.array(history_quad_loss))
    plt.title('Quad energy')
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.plot(np.array(history_smooth_loss))
    plt.title('Smooth energy')
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.semilogy(np.array(history_loss))
    plt.title('Data energy')
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.semilogy(np.array(history_p_deviation), c='b', label='from origin')
    plt.semilogy(np.array(history_p_deviation_target), c='r', label='from target')
    plt.legend()
    plt.title('Deviation')
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.semilogy(np.array(history_p_deviation_mean), c='b', label='from origin')
    plt.semilogy(np.array(history_p_deviation_target_mean), c='r', label='from target')
    plt.legend()
    plt.title('Mean deviation')
    plt.show()

def make_M_from_tqs(t, q, s):
    q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    M = T.dot(R).dot(S)
    return M 


def two_tetrahedrons():
    
    vertices_1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    vertices_2 = np.array([[2, 0, 0], [3, 0, 0], [2, 1, 0], [2, 0, 2]])
    
    faces_1 = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 3], [1, 2, 3],
                        [0, 2, 1], [0, 3, 2], [0, 3, 1], [1, 3, 2]])
    faces_2 = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 3], [1, 2, 3],
                        [0, 2, 1], [0, 3, 2], [0, 3, 1], [1, 3, 2]])
    
    mesh_1 = Trimesh(vertices_1, faces_1)
    mesh_2 = Trimesh(vertices_2, faces_2)
    
    return mesh_1, mesh_2


def sphere(subdivisions=3, radius=1.0):
    
    mesh = trimesh.primitives.Sphere(subdivisions=subdivisions, radius=radius)
    
    return mesh


def plane(width=2, length=2, num_points=2500):
    
    x = np.linspace(0, length, num=int(np.sqrt(num_points)))
    y = np.linspace(0, width, num=int(np.sqrt(num_points)))
    x, y = np.meshgrid(x, y)
    z = np.zeros_like(x)
    tri = mtri.Triangulation(x.flatten(), y.flatten())
    
    faces = tri.triangles
    faces_dual = faces[: ,[0, 2, 1]]
    faces = np.vstack([faces, faces_dual])
    vertices = np.array([x.flatten(), y.flatten(), z.flatten()]).T
    
    plane = Trimesh(vertices=vertices, faces=faces)
    
    return plane


def saddle(num_points=2500):
    
    def f(x, y):
        return x ** 2 - y ** 2
    
    x = np.linspace(-1, 1, num=int(np.sqrt(num_points)))
    y = np.linspace(-1, 1, num=int(np.sqrt(num_points)))
    x, y = np.meshgrid(x, y)
    z = f(x, y)
    tri = mtri.Triangulation(x.flatten(), y.flatten())
    
    faces = tri.triangles
    faces_dual = faces[: ,[0, 2, 1]]
    faces = np.vstack([faces, faces_dual])
    vertices = np.array([x.flatten(), y.flatten(), z.flatten()]).T
    
    saddle = Trimesh(vertices=vertices, faces=faces)
    
    return saddle


def monkey_saddle(num_points=2500):
    
    def f(x, y):
        return x ** 3 - 3 * x * y ** 2
    
    x = np.linspace(-1, 1, num=int(np.sqrt(num_points)))
    y = np.linspace(-1, 1, num=int(np.sqrt(num_points)))
    x, y = np.meshgrid(x, y)
    z = f(x, y)
    tri = mtri.Triangulation(x.flatten(), y.flatten())
    
    faces = tri.triangles
    faces_dual = faces[: ,[0, 2, 1]]
    faces = np.vstack([faces, faces_dual])
    vertices = np.array([x.flatten(), y.flatten(), z.flatten()]).T
    
    saddle = Trimesh(vertices=vertices, faces=faces)
    
    return saddle


def box(size=(1, 1, 1), max_edge=0.1):
    
    box = trimesh.creation.box(extents=size)    
    vertices, faces = subdivide_to_size(box.vertices, box.faces, max_edge)
    mesh = Trimesh(vertices, faces)
    
    return mesh


def mesh_pcloud(points, size=0.1, color=None):
    
    boxes = []
    for point in points:
        box = trimesh.creation.box(extents=(size, size, size)) 
        box.apply_transform(translate([point - np.array([size/2, size/2, size/2])]))
        if color is not None:
            for facet in box.facets:
                box.visual.face_colors[facet] = color
        boxes += [box]
        
    boxes = trimesh.util.concatenate(boxes)
    
    return boxes
        
        
def set_new_mesh_vertices(mesh, vertices):
    
    mesh_new = Trimesh(vertices=vertices.copy(), faces=mesh.faces, process=False)
    
    return mesh_new


def affine_transform(mesh, transform):
    
    mesh_new = mesh.copy()
    mesh_new.apply_transform(transform)
    
    return mesh_new


def rot_x(angle=0):
    
    angle = angle * np.pi / 180
    return np.array([[1, 0, 0, 0],
                     [0, np.cos(angle), -np.sin(angle), 0],
                     [0, np.sin(angle), np.cos(angle), 0],
                     [0, 0, 0, 1]])


def rot_y(angle=0):
    
    angle = angle * np.pi / 180
    return np.array([[np.cos(angle), 0, -np.sin(angle), 0],
                     [0, 1, 0, 0],
                     [np.sin(angle), 0, np.cos(angle), 0],
                     [0, 0, 0, 1]])


def rot_z(angle=0):
    
    angle = angle * np.pi / 180
    return np.array([[np.cos(angle), -np.sin(angle), 0, 0],
                     [np.sin(angle), np.cos(angle), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def translate(vector=(0, 0, 0)):

    transform = np.eye(4, dtype=float)
    transform[:3, 3] = np.array(vector)
    return transform


def scale(scale=(1, 1, 1)):
    
    transform = np.eye(4, dtype=float)
    transform[0, 0] = np.array([scale[0]])
    transform[1, 1] = np.array([scale[1]])
    transform[2, 2] = np.array([scale[2]])
    return transform


def compose(transforms):
    
    identity = np.eye(4, dtype=float)
    for transform in transforms:
        identity = transform @ identity
        
    return identity 


def remove_duplicate_vertices(mesh):
    
    unique_vertices, unique_inverse = np.unique(mesh.vertices, axis=0, return_inverse=True)
    old2new_vertices = dict(zip(np.arange(len(mesh.vertices)), unique_inverse))
    new_faces = np.copy(mesh.faces)
    for i, face in enumerate(new_faces):
        new_face = np.array([old2new_vertices[face[0]], old2new_vertices[face[1]], old2new_vertices[face[2]]])
        new_faces[i] = new_face  
    new_mesh = Trimesh(unique_vertices, new_faces, process=False)
    
    return new_mesh, old2new_vertices


def mesh_subsample(mesh, vertices):
    
    num_vertices = len(vertices)
    new_faces = []
    for face in mesh.faces:
        if (face[0] < num_vertices) and (face[1] < num_vertices) and (face[2] < num_vertices):
            new_faces += [face]
    new_mesh = Trimesh(vertices, new_faces)
    
    return new_mesh


def pcloud_subsample(points, num=100):
    
    if num > len(points):
        num = len(points)
    points = np.array(points)
    new_points = [points[0]]
    new_points_numpy = np.array(new_points)[None, :]
    for i in range(num-1):
        dists = np.sum(np.sqrt(np.sum((points[:, None, ...] - new_points_numpy) ** 2, axis=2)), axis=1)
        max_dist_id = np.argmax(dists)
        new_points += [points[max_dist_id]]
        new_points_numpy = np.array(new_points)
    
    return new_points_numpy


def concatenate_with_transforms(parts, transforms):
    
    concatenated_mesh = trimesh.util.concatenate(parts)
    concatenated_transforms = torch.cat(transforms, dim=0)
    
    unique_vertices, unique_inverse = np.unique(concatenated_mesh.vertices, axis=0, return_inverse=True)
    old2new_vertices = dict(zip(np.arange(len(concatenated_mesh.vertices)), unique_inverse))
    new_faces = np.copy(concatenated_mesh.faces)
    for i, face in enumerate(new_faces):
        new_face = np.array([old2new_vertices[face[0]], old2new_vertices[face[1]], old2new_vertices[face[2]]])
        new_faces[i] = new_face 
    
    new_transforms = torch.zeros((len(unique_vertices), 4, 4))
    for i in range(len(concatenated_transforms)):
        new_transforms[old2new_vertices[i]] = concatenated_transforms[i]
    new_mesh = Trimesh(unique_vertices, new_faces, process=False)
    
    return new_mesh, new_transforms


def add_noise_to_mesh(mesh, noise):
    
    new_vertices = mesh.vertices.copy()
    new_vertices = new_vertices + noise
    
    new_mesh = Trimesh(new_vertices, mesh.faces, process=False)
    
    return new_mesh


def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def apply_transform(mesh, transforms):
    
    new_vertices = np.array(mesh.vertices.copy())
    new_vertices = np.stack([new_vertices, np.ones(len(new_vertices))[:, None]], axis=1)
    new_vertices = np.matmul(transforms, new_vertices)
    
    new_mesh = Trimesh(new_vertices, mesh.faces, process=False)
    
    return new_mesh


def sample_points_around(mesh, num_points, shift=0, offset=0, scale=1):
    
    if shift==None:
        center = np.mean(mesh.vertices, axis=0)
    else:
        center = np.mean(mesh.vertices, axis=0) + shift
    samples_id = np.random.choice(np.arange(len(mesh.vertices)), size=num_points, replace=False)
    samples = mesh.vertices[samples_id]
    samples_directions = samples - center
    
    samples_around = center + samples_directions * np.abs(np.random.normal(loc=offset, scale=scale, size=(len(samples_directions), 1)))
    
    return (samples, samples_id, samples_around)


def filter_edges_by_parts(sharp_edges_for_mesh, partnet_map):
    non_conflict_edges = {}
    conflict_edges = []
    for edge in sharp_edges_for_mesh:
        v0, v1 = edge
        if v0 != v1:
            if partnet_map[v0][0] == partnet_map[v1][0]:
                if partnet_map[v0][0] in non_conflict_edges:
                    non_conflict_edges[partnet_map[v0][0]] += [edge]
                else:
                    non_conflict_edges[partnet_map[v0][0]] = [edge]
            else:
                conflict_edges += [edge]
    return non_conflict_edges, conflict_edges


def remove_degeneracies(undeformed_vertices, target_vertices, unique_edges, bitriangles_map, deg_thr=1e-3, ampl_factor=1):

    n_vertices_old = len(undeformed_vertices)
    n_edges_old = len(unique_edges)
    original_vertices = torch.empty_like(undeformed_vertices).copy_(undeformed_vertices)
    original_bitriangles_map = torch.empty_like(bitriangles_map).copy_(bitriangles_map)
    original_target_vertices = torch.empty_like(target_vertices).copy_(target_vertices)

    #########################
    # Undeformed processing #
    #########################

    # torch.Tensor(n_edges, 4, 3)
    bitriangles_explicit_undeformed = (undeformed_vertices[bitriangles_map.view(-1).long(), :].view(n_edges_old, 4, 4)[:, :, :3]).double()
    # torch.Tensor(n_edges, 4, 3)
    bitriangles_explicit_target = (target_vertices[bitriangles_map.view(-1).long(), :].view(n_edges_old, 4, 4)[:, :, :3]).double()
    # torch.Tensor(n_edges, 3, 3)
    v_0 = torch.stack([bitriangles_explicit_undeformed[:, 0, :] - bitriangles_explicit_undeformed[:, 1, :],
                       bitriangles_explicit_undeformed[:, 3, :] - bitriangles_explicit_undeformed[:, 1, :],
                       bitriangles_explicit_undeformed[:, 2, :] - bitriangles_explicit_undeformed[:, 1, :]], dim=1).double()
    # torch.Tensor(n_edges, 3, 3)
    vol_120 = torch.sum(torch.cross(v_0[:, 1, :], v_0[:, 2, :]) * v_0[:, 0, :], dim=1)[:, None]
    # np.array(n_edges)
    zero_volumes = np.where(torch.abs(vol_120).numpy() < deg_thr)[0]

    print('[Num flat edges (before):', len(zero_volumes), ']')

    #############################
    # Adding auxiliary vertices #
    #############################

    if len(zero_volumes) > 0:
        cross = torch.cross(v_0[zero_volumes, 0], v_0[zero_volumes, 2])
        cross *= ampl_factor

        fifth_vertices = cross + bitriangles_explicit_undeformed[zero_volumes, 1, :]

        cross_target = torch.cross(bitriangles_explicit_target[zero_volumes, 0, :] - bitriangles_explicit_target[zero_volumes, 1, :],
                                   bitriangles_explicit_target[zero_volumes, 2, :] - bitriangles_explicit_target[zero_volumes, 1, :])
        cross_target *= ampl_factor
        fifth_vertices_target = cross_target + bitriangles_explicit_target[zero_volumes, 1, :]
        fifth_vertices_idx = torch.arange(len(zero_volumes)) + n_vertices_old

        original_vertices = torch.cat([original_vertices[:, :3].double(), fifth_vertices], dim=0)
        original_vertices = torch.cat([original_vertices, torch.ones(len(original_vertices))[..., None].double()], dim=1)

        original_target_vertices = torch.cat([original_target_vertices[:, :3].double(), fifth_vertices_target], dim=0)
        original_target_vertices = torch.cat([original_target_vertices, torch.ones(len(original_target_vertices))[..., None].double()], dim=1)

        # torch.Tensor(n_degenerate_bitriangles, 4)
        old_bitriangles = bitriangles_map[zero_volumes]
        new_bitriangles_1 = torch.empty_like(old_bitriangles).copy_(old_bitriangles)
        new_bitriangles_1[:, 3] = fifth_vertices_idx
        new_bitriangles_2 = torch.empty_like(old_bitriangles).copy_(old_bitriangles[:, [0, 1, 3, 2]])
        new_bitriangles_2[:, 3] = fifth_vertices_idx
        original_bitriangles_map[zero_volumes] = new_bitriangles_1

    n_vertices = len(original_vertices)
    n_edges = len(original_bitriangles_map)
    updated_vertices = original_vertices
    updated_target_vertices = original_target_vertices
    updated_edges = original_bitriangles_map[:, :2]
    updated_bitriangles_map = original_bitriangles_map

    ####################
    # Make final check #
    ####################

    bitriangles_explicit_undeformed = (updated_vertices[updated_bitriangles_map.view(-1).long(), :].view(n_edges, 4, 4)[:, :, :3]).double()
    v_0 = torch.stack([bitriangles_explicit_undeformed[:, 0, :] - bitriangles_explicit_undeformed[:, 1, :],
                       bitriangles_explicit_undeformed[:, 3, :] - bitriangles_explicit_undeformed[:, 1, :],
                       bitriangles_explicit_undeformed[:, 2, :] - bitriangles_explicit_undeformed[:, 1, :]], dim=1).double()
    vol_120 = torch.sum(torch.cross(v_0[:, 1, :], v_0[:, 2, :]) * v_0[:, 0, :], dim=1)[:, None]
    zero_volumes = np.where(torch.abs(vol_120).numpy() < deg_thr)[0]
    print('[Num flat edges (after):', len(zero_volumes), ']')
    print('[Number of vertices (before):', n_vertices_old, ']')
    print('[Number of vertices (after):', n_vertices, ']')

    return updated_bitriangles_map, updated_vertices, updated_target_vertices, updated_edges, \
           n_vertices, n_vertices_old, n_edges, n_edges_old
            
    
def make_frames(deformer, parts, alpha_reg_bounds=(-3, 3), alphas_per_scale=10, save_dir=''):
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'initials'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'approximations'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'compares'), exist_ok=True)
    
    scales = list(range(alpha_reg_bounds[0], alpha_reg_bounds[1], 1))
    alphas = [list(np.linspace(10 ** scales[i], 10 ** scales[i+1], num=alphas_per_scale, endpoint=False)) for i in range(len(scales)-1)]
    k = 0
    
    for scale_array in tqdm(alphas):
        for alpha in scale_array:
            deformer.solve(alpha_reg=alpha)
            
            first_approximation = deformer.get_first_approximation()
            initial_vertices = deformer.get_initial_parts_vertices()
            
            initial_shape = []
            for i in range(len(initial_vertices)):
                initial_mesh = set_new_mesh_vertices(parts[i], initial_vertices[i])
                color = np.array([255, 0, 0, 0])
                for facet in initial_mesh.facets:
                    initial_mesh.visual.face_colors[facet] = color
                initial_shape += [initial_mesh]
            initial_shape = trimesh.util.concatenate(initial_shape)
            
            approximation_shape = []
            for i in range(len(first_approximation)):
                approximation_mesh = set_new_mesh_vertices(parts[i], first_approximation[i])
                color = np.array([0, 255, 0, 0])
                for facet in approximation_mesh.facets:
                    approximation_mesh.visual.face_colors[facet] = color
                approximation_shape += [approximation_mesh]
            approximation_shape = trimesh.util.concatenate(approximation_shape)
            
            shape_to_compare = trimesh.util.concatenate([approximation_shape, initial_shape])
            
            initial_shape.export(os.path.join(save_dir, 'initials', '{}_{}_{}.obj'.format(k, 'initial', alpha)))
            approximation_shape.export(os.path.join(save_dir, 'approximations', '{}_{}_{}.obj'.format(k, 'approximation', alpha)))
            shape_to_compare.export(os.path.join(save_dir, 'compares', '{}_{}_{}.obj'.format(k, 'compare', alpha)))
            
            k += 1
