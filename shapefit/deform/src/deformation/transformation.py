import torch
from scipy.sparse import coo_matrix
import scipy
import numpy as np
import time


def edges_deformation_from_vertices(torch_vertices_4d,
                                    unique_edges,
                                    target_vertices,
                                    bitriangles_map=None):
    """ Returns list of edges deformations form list of vertices transformation 

    Parameters:
        parts_vertices (list<torch.Tensor>): list of tensors with vertices [n_parts, (n_vertices, 4)]
        parts_edges (list<torch.IntTensor>): list of tensors with edges indices [n_parts, (n_edges, 2)]
        parts_vertices_target (list<torch.Tensor): list of tensors with target vertices [n_parts, (n_vertices, 4)]
        parts_bitriangles_map (list<torch.IntTensor>): list of tensors with sets of vertex indices for each bitriangle 
                                                                                            [n_parts, (n_edges, 4)]
        parts_normals_map (list<torch.IntTensor>): list of tensors with the third vertex in face for each edge 
                                                                                            [n_parts, (n_edges, 1)]
        method: method for computing edge deformation, one of
            'bitriangles': for each edge find 2 adjacent triangles
    Returns:
        edges_deformations (list<torch.Tensor>): list of deformation matrices on edges [n_parts, (n_edges, 4, 4)]

    """
    edges_deformations = []
    if bitriangles_map is None:
        raise ValueError('parts_bitriangles_map can not be None for bitriangles method')

    # torch.Tensor(n_edges, 4, 4)
    bitriangles_explicit_orig = torch_vertices_4d[bitriangles_map.view(-1).long(), :].view(len(unique_edges), 4, 4).double()
    # torch.Tensor(n_edges, 4, 4)
    bitriangles_explicit_pred = target_vertices[bitriangles_map.view(-1).long(), :].view(len(unique_edges), 4, 4).double()
    edges_transforms = torch.matmul(bitriangles_explicit_orig.inverse(), bitriangles_explicit_pred)
    edges_deformations += [edges_transforms]

    return edges_deformations


def deformation_matrix(parts_vertices,
                       undeformed_vertices, 
                       parts_edges_indices, 
                       parts_bitriangles_map, 
                       edges_deformations,
                       parts_faces_to_edges_map, 
                       parts_faces,
                       part_sharp_edges_ids,
                       alpha_0,
                       alpha_reg,
                       alpha_sharp):
    
    # torch.Tensor(3, 4)
    D = torch.Tensor(np.array([[1, 0, 0, -1],
                               [0, 1, 0, -1],
                               [0, 0, 1, -1]])).double()
    
    D_1 = torch.zeros((3, 12)).double(); D_1[0, 0] = 1; D_1[1, 4] = 1; D_1[2, 8] = 1
    D_2 = torch.zeros((3, 12)).double(); D_2[0, 1] = 1; D_2[1, 5] = 1; D_2[2, 9] = 1
    D_3 = torch.zeros((3, 12)).double(); D_3[0, 2] = 1; D_3[1, 6] = 1; D_3[2, 10] = 1
    D_4 = torch.zeros((3, 12)).double(); D_4[0, 3] = 1; D_4[1, 7] = 1; D_4[2, 11] = 1
    
    A_matrices = []
    b_vectors = []
    
    b_vectors_normed = []

    # print(parts_vertices.shape)
    # print(parts_edges_indices.shape)
    # print(parts_faces.shape)

    n_edges = len(parts_edges_indices)
    n_vertices = len(parts_vertices)
    n_faces = len(parts_faces)
    # torch.Tensor(n_edges, 4, 3)
    bitriangles_explicit_undeformed = (undeformed_vertices[parts_bitriangles_map.view(-1).long(), :].view(n_edges, 4, 4)[:, :, :3]).double()
    # torch.Tensor(n_edges, 3, 3)
    v_0 = torch.stack([bitriangles_explicit_undeformed[:, 0, :] - bitriangles_explicit_undeformed[:, 3, :],
                       bitriangles_explicit_undeformed[:, 1, :] - bitriangles_explicit_undeformed[:, 3, :],
                       bitriangles_explicit_undeformed[:, 2, :] - bitriangles_explicit_undeformed[:, 3, :]], dim=1).double()
        
    # torch.Tensor(n_edges, 3, 3)
    vol_120 = torch.sum(torch.cross(v_0[:, 1, :], v_0[:, 2, :]) * v_0[:, 0, :], dim=1)[:, None]
    vol_201 = torch.sum(torch.cross(v_0[:, 2, :], v_0[:, 0, :]) * v_0[:, 1, :], dim=1)[:, None]
    vol_012 = torch.sum(torch.cross(v_0[:, 0, :], v_0[:, 1, :]) * v_0[:, 2, :], dim=1)[:, None]
    # np.array(n_edges)
    zero_volumes = np.where(torch.abs(vol_120).numpy() < 0)[0]
        
    flag_degenerate = False
    if len(zero_volumes > 0):
        flag_degenerate = True
        for j, bitriangle_idx in enumerate(zero_volumes):
            fifth_vertex = torch.cross(bitriangles_explicit_undeformed[bitriangle_idx, 0, :] - bitriangles_explicit_undeformed[bitriangle_idx, 1, :],
                                       bitriangles_explicit_undeformed[bitriangle_idx, 0, :] - bitriangles_explicit_undeformed[bitriangle_idx, 2, :]) + bitriangles_explicit_undeformed[bitriangle_idx, 0, :]
            fifth_vertex_idx = n_vertices + j
            undeformed_vertices = torch.cat([undeformed_vertices[:, :3].double(), fifth_vertex[None, :]], dim=0)
            old_bitriangle = parts_bitriangles_map[bitriangle_idx]
            new_bitriangle_1 = torch.IntTensor([old_bitriangle[0], old_bitriangle[1], old_bitriangle[2], fifth_vertex_idx])
            new_bitriangle_2 = torch.IntTensor([old_bitriangle[0], old_bitriangle[1], old_bitriangle[3], fifth_vertex_idx])

            parts_bitriangles_map = torch.cat([parts_bitriangles_map,
                                                  new_bitriangle_1[None, :],
                                                  new_bitriangle_2[None, :]], dim=0)
        parts_bitriangles_map = torch.IntTensor(np.delete(parts_bitriangles_map.numpy(), zero_volumes, axis=0))
        bitriangles_explicit_undeformed = (undeformed_vertices[parts_bitriangles_map.view(-1).long(), :].view(-1, 4, 3)).double()

        # torch.Tensor(n_edges, 3, 3)
        v_0 = torch.stack([bitriangles_explicit_undeformed[:, 0, :] - bitriangles_explicit_undeformed[:, 3, :],
                           bitriangles_explicit_undeformed[:, 1, :] - bitriangles_explicit_undeformed[:, 3, :],
                           bitriangles_explicit_undeformed[:, 2, :] - bitriangles_explicit_undeformed[:, 3, :]], dim=1).double()
        n_edges = len(parts_bitriangles_map)
        n_vertices = len(undeformed_vertices)

        # torch.Tensor(n_edges, 3, 3)
        vol_120 = torch.sum(torch.cross(v_0[:, 1, :], v_0[:, 2, :]) * v_0[:, 0, :], dim=1)[:, None]
        vol_201 = torch.sum(torch.cross(v_0[:, 2, :], v_0[:, 0, :]) * v_0[:, 1, :], dim=1)[:, None]
        vol_012 = torch.sum(torch.cross(v_0[:, 0, :], v_0[:, 1, :]) * v_0[:, 2, :], dim=1)[:, None]

    # torch.Tensor(n_edges, 3, 3)
    dual_w = torch.stack([torch.cross(v_0[:, 1, :], v_0[:, 2, :]) / vol_120,
                          torch.cross(v_0[:, 2, :], v_0[:, 0, :]) / vol_201,
                          torch.cross(v_0[:, 0, :], v_0[:, 1, :]) / vol_012], dim=1)
    dual_w = torch.transpose(dual_w, 1, 2)

    # torch.Tensor(n_edges, 3, 3)
    v_sqr = torch.stack([bitriangles_explicit_undeformed[:, 0, :] - bitriangles_explicit_undeformed[:, 1, :],
                       bitriangles_explicit_undeformed[:, 0, :] - bitriangles_explicit_undeformed[:, 2, :],
                       bitriangles_explicit_undeformed[:, 0, :] - bitriangles_explicit_undeformed[:, 3, :]], dim=1).double()
    sqr = torch.abs(torch.cross(v_sqr[:, 0, :], v_sqr[:, 1, :])) + torch.abs(torch.cross(v_sqr[:, 0, :], v_sqr[:, 2, :]))
    sqr = torch.sqrt(torch.sum(sqr ** 2, dim=1))

    ########################
    # Computing A_0 matrix #
    ########################

    # torch.Tensor(n_edges, 3, 4)
    WD = torch.bmm(dual_w, D.repeat(n_edges, 1, 1))
    # torch.Tensor(n_edges, 4, 4)
    G = torch.bmm(WD.transpose(1, 2), WD)

    G_normed = sqr[:, None, None] * G

    # torch.Tensor(n_edges, 48)
    G_flattened = torch.flatten(G, 1, 2).repeat(1, 3)

    G_flattened_normed = torch.flatten(G_normed, 1, 2).repeat(1, 3)
    # torch.Tensor(n_edges, 48)
    C = parts_bitriangles_map.repeat(1, 4)
    C_cat = torch.zeros(n_edges, 48)
    C_cat[:, :16] = C
    C_cat[:, 16:32] = C + n_vertices
    C_cat[:, 32:48] = C + 2*n_vertices
    # torch.Tensor(n_edges, 48)
    R = torch.transpose(parts_bitriangles_map.repeat(1, 4).view(n_edges, 4, 4), 1, 2).flatten(1, 2)
    R_cat = torch.zeros(n_edges, 48)
    R_cat[:, :16] = R
    R_cat[:, 16:32] = R + n_vertices
    R_cat[:, 32:48] = R + 2*n_vertices

    # scipy.sparse(3 * n_vertices, 3 * n_vertices)
    A_0_sparse = coo_matrix((G_flattened.numpy().flatten(), (R_cat.numpy().flatten(), C_cat.numpy().flatten())),
                            shape=(3*n_vertices, 3*n_vertices))
    # print(scipy.sparse.linalg.norm(A_0_sparse, 'fro'))
    # print(A_0_sparse.todense()[:10, :10])

    A_0_sparse_normed = coo_matrix((G_flattened_normed.numpy().flatten(), (R_cat.numpy().flatten(), C_cat.numpy().flatten())),
                                   shape=(3*n_vertices, 3*n_vertices))

    # torch.Tensor(n_edges, 36)
    WD_flattened = torch.flatten(WD, 1, 2).repeat(1, 3)
    # torch.Tensor(n_edges, 36)
    C = parts_bitriangles_map.repeat(1, 3)
    C_cat = torch.zeros(n_edges, 36)
    C_cat[:, :12] = C
    C_cat[:, 12:24] = C + n_vertices
    C_cat[:, 24:36] = C + 2 * n_vertices
    # torch.Tensor(n_edges, 36)
    R = torch.arange(3)[None, ...].repeat(4, 1).T.flatten().repeat(3)
    R[12:24] += 3 * n_edges
    R[24:36] += 6 * n_edges
    R_cat = torch.zeros(n_edges, 36)
    for i in range(n_edges):
        R_cat[i] = R + i * 3

    # scipy.sparse(36, 3 * n_vertices)
    A_0_sparse_sqrt = coo_matrix((WD_flattened.numpy().flatten(), (R_cat.numpy().flatten(), C_cat.numpy().flatten())),
                                 shape=(9*n_edges, 3*n_vertices))

    A_0_sparse_safe = A_0_sparse_sqrt.T @ A_0_sparse_sqrt
    # print(A_0_sparse_safe.todense()[:10, :10])
    # print('A_0_safe norm:', scipy.sparse.linalg.norm(A_0_sparse_safe, 'fro'))
    # print('A_0 norm:', scipy.sparse.linalg.norm(A_0_sparse, 'fro'))
    # print('A_0_safe - A_0 norm:', scipy.sparse.linalg.norm(A_0_sparse_safe - A_0_sparse, 'fro'))

    ###########################
    # Computing (D^T)D matrix #
    ###########################

    # torch.Tensor(n_edges, 4)
    s = torch.bmm(torch.transpose(WD, 1, 2), bitriangles_explicit_undeformed[:, 3][:, :, None])[:, :, 0]
    sD1 = s[:, 0][:, None, None] * D_1[None, :, :].repeat(n_edges, 1, 1)
    sD2 = s[:, 1][:, None, None] * D_2[None, :, :].repeat(n_edges, 1, 1)
    sD3 = s[:, 2][:, None, None] * D_3[None, :, :].repeat(n_edges, 1, 1)
    sD4 = s[:, 3][:, None, None] * D_4[None, :, :].repeat(n_edges, 1, 1)
    D_4_repeat = D_4[None, :, :].repeat(n_edges, 1, 1)
    # torch.Tensor(n_edges, 3, 12)
    D_e = D_4_repeat - sD1 - sD2 - sD3 - sD4
    # torch.Tensor(n_edges, 12, 12)
    DTD = torch.bmm(torch.transpose(D_e, 1, 2), D_e)

    # torch.Tensor(n_edges, 144)
    DTD_flattened = torch.flatten(DTD, 1, 2)
    # torch.Tensor(n_edges, 12, 12)
    C = torch.transpose(parts_bitriangles_map.repeat(12, 1, 1), 0, 1)
    C_cat = torch.zeros(n_edges, 12, 12)
    C_cat[:, :, :4] = C
    C_cat[:, :, 4:8] = C + n_vertices
    C_cat[:, :, 8:] = C + 2*n_vertices
    # torch.Tensor(n_edges, 144)
    C_flattened = torch.flatten(C_cat, 1, 2)
    # torch.Tensor(n_edges, 12, 12)
    R = torch.transpose(C_cat, 1, 2)
    # torch.Tensor(n_edges, 144)
    R_flattened = torch.flatten(R, 1, 2)

    # scipy.sparse(3 * n_vertices, 3 * n_vertices)
    DTD_sparse = coo_matrix((DTD_flattened.numpy().flatten(), (R_flattened.numpy().flatten(), C_flattened.numpy().flatten())),
                            shape=(3*n_vertices, 3*n_vertices))

    # torch.Tensor(n_edges, 36)
    D_e_flattened = torch.flatten(D_e, 1, 2)
    # torch.Tensor(n_edges, 3, 12)
    C = torch.transpose(parts_bitriangles_map.repeat(3, 1, 1), 0, 1)
    C_cat = torch.zeros(n_edges, 3, 12)
    C_cat[:, :, :4] = C
    C_cat[:, :, 4:8] = C + n_vertices
    C_cat[:, :, 8:] = C + 2 * n_vertices
    # torch.Tensor(n_edges, 36)
    C_cat = torch.flatten(C_cat, 1, 2)
    # torch.Tensor(n_edges, 36)
    R = torch.arange(3)[None, ...].repeat(12, 1).T.flatten()
    R_cat = torch.zeros(n_edges, 36)
    for i in range(n_edges):
        R_cat[i] = R + i * 3

    D_e_sparse = coo_matrix((D_e_flattened.numpy().flatten(), (R_cat.numpy().flatten(), C_cat.numpy().flatten())),
                            shape=(9*n_edges, 3*n_vertices))
    DTD_sparse_safe = D_e_sparse.T @ D_e_sparse
    # print('DTD_safe norm:', scipy.sparse.linalg.norm(DTD_sparse_safe, 'fro'))
    # print('DTD norm:', scipy.sparse.linalg.norm(DTD_sparse, 'fro'))
    # print('DTD_safe - DTD norm:', scipy.sparse.linalg.norm(DTD_sparse_safe - DTD_sparse, 'fro'))

    # scipy.sparse(3 * n_vertices, 3 * n_vertices)
    A_0_sparse = A_0_sparse + DTD_sparse

    # A_0_sparse_safe = A_0_sparse_sqrt_T @ A_0_sparse_sqrt
    # A_0_sparse_safe = A_0_sparse_safe + DTD_sparse

    ########################
    # Computing A_2 matrix #
    ########################

    M = torch.zeros(n_edges, 12, 12).double()
    M[:, :3, :4] = WD
    M[:, 3:6, 4:8] = WD
    M[:, 6:9, 8:] = WD
    M[:, 9:, :] = D_e

    # torch.Tensor(n_edges, 144)
    M_flattened = torch.flatten(M, 1, 2)

    # torch.Tensor(n_edges, 12, 12)
    C = torch.transpose(parts_bitriangles_map.repeat(12, 1, 1), 0, 1)
    C_cat = torch.zeros((n_edges, 12, 12), dtype=torch.int)
    C_cat[:, :, :4] = C
    C_cat[:, :, 4:8] = C + n_vertices
    C_cat[:, :, 8:] = C + 2*n_vertices
    C_flattened = torch.flatten(C_cat, 1, 2)

    # torch.Tensor(n_edges)
    edges_indices = torch.arange(n_edges)[:, None, None]
    # torch.Tensor(n_edges, 12, 12)
    R = edges_indices.repeat(1, 12, 12) * 12 + torch.arange(12)[:, None].repeat(1, 12)
    # torch.Tensor(n_edges, 144)
    R_flattened = torch.flatten(R, 1, 2)
    T_g_sparse = coo_matrix((M_flattened.numpy().flatten(), (R_flattened.numpy().flatten(), C_flattened.numpy().flatten())),
                            shape=(12*n_edges, 3*n_vertices))

    T_g_sparse_normed = coo_matrix(((torch.sqrt(sqr[:, None]) * M_flattened).numpy().flatten(), (R_flattened.numpy().flatten(), C_flattened.numpy().flatten())),
                                   shape=(12*n_edges, 3*n_vertices))

    faces_array = np.arange(len(parts_faces_to_edges_map))
    R = np.hstack([3*faces_array,
                   3*faces_array+1,
                   3*faces_array+2])
    C1 = np.hstack([parts_faces_to_edges_map[:, 1],
                    parts_faces_to_edges_map[:, 2],
                    parts_faces_to_edges_map[:, 0]])
    C2 = np.hstack([parts_faces_to_edges_map[:, 0],
                    parts_faces_to_edges_map[:, 1],
                    parts_faces_to_edges_map[:, 2]])
    Fs1 = coo_matrix((np.ones(len(R)), (R, C1)),
                     shape=(3*n_faces, n_edges))
    Fs2 = coo_matrix((-np.ones(len(R)), (R, C2)),
                     shape=(3*n_faces, n_edges))
    Fs = Fs1 + Fs2
    Fs = Fs.astype('int8')
    F_sparse = scipy.sparse.block_diag((Fs, Fs, Fs, Fs, Fs, Fs, Fs, Fs, Fs, Fs, Fs, Fs))

    edges_array = np.arange(n_edges)

    R = np.hstack([edges_array,
                   edges_array+n_edges,
                   edges_array+2*n_edges,
                   edges_array+3*n_edges,
                   edges_array+4*n_edges,
                   edges_array+5*n_edges,
                   edges_array+6*n_edges,
                   edges_array+7*n_edges,
                   edges_array+8*n_edges,
                   edges_array+9*n_edges,
                   edges_array+10*n_edges,
                   edges_array+11*n_edges])
    C = np.hstack([12*edges_array,
                   12*edges_array+1,
                   12*edges_array+2,
                   12*edges_array+3,
                   12*edges_array+4,
                   12*edges_array+5,
                   12*edges_array+6,
                   12*edges_array+7,
                   12*edges_array+8,
                   12*edges_array+9,
                   12*edges_array+10,
                   12*edges_array+11])
    P = coo_matrix((np.ones(len(R)), (R, C)),
                   shape=(12*n_edges, 12*n_edges))

    P_sparse = P.astype('int8')

    A_2_sparse = F_sparse @ P_sparse @ T_g_sparse
    A_2_sqrt = A_2_sparse
    A_2_sparse = A_2_sparse.T @ A_2_sparse

    A_2_sparse_normed = F_sparse @ P_sparse @ T_g_sparse_normed
    A_2_sparse_normed = A_2_sparse_normed.T @ A_2_sparse_normed
    A_sparse = alpha_0 * A_0_sparse + alpha_reg * A_2_sparse

    ########################
    # Computing A_3 matrix #
    ########################

    A_3_sparse = 0
    if part_sharp_edges_ids is not None:
        for sharp_edges_ids in part_sharp_edges_ids:
            if len(sharp_edges_ids) >= 2:
                first_set = sharp_edges_ids[1:]
                second_set = sharp_edges_ids[:-1]
                R = np.arange(len(sharp_edges_ids)-1)
                Fs1 = coo_matrix((np.ones(len(R)), (R, first_set)),
                                 shape=(len(sharp_edges_ids)-1, n_edges))
                Fs2 = coo_matrix((-np.ones(len(R)), (R, second_set)),
                                 shape=(len(sharp_edges_ids)-1, n_edges))
                Fs = Fs1 + Fs2
                Fs = Fs.astype('int8')
                F_sparse = scipy.sparse.block_diag((Fs, Fs, Fs, Fs, Fs, Fs, Fs, Fs, Fs, Fs, Fs, Fs))

                A_3_sparse_part = F_sparse @ P_sparse @ T_g_sparse
                A_3_sparse_part = A_3_sparse_part.T @ A_3_sparse_part
                A_3_sparse += A_3_sparse_part
    A_sparse += alpha_sharp * A_3_sparse
    A_matrices += [A_sparse]

    ###########################
    # Computing (d^T)D vector #
    ###########################

    # torch.Tensor(n_edges, 4, 4)
    deformations = edges_deformations[0].transpose(1, 2).double()
    # torch.Tensor(n_edges, 3, 3)
    affine_components = deformations[:, :3, :3]
    # torch.Tensor(n_edges, 1, 3)
    translation_components = deformations[:, :3, 3][:, None, :]

    # torch.Tensor(n_edges, 12)
    dTD = torch.matmul(translation_components, D_e)

    dTD_normed = sqr[:, None, None] * dTD
    # torch.Tensor(n_edges, 12)
    R = parts_bitriangles_map
    R_cat = torch.zeros(n_edges, 12)
    R_cat[:, :4] = R
    R_cat[:, 4:8] = R + n_vertices
    R_cat[:, 8:] = R + 2*n_vertices

    # scipy.sparse(3 * n_vertices, 1)
    dTD_sparse = coo_matrix((dTD.numpy().flatten(), (R_cat.numpy().flatten(), np.zeros(12*n_edges))),
                            shape=(3*n_vertices, 1))

    dTD_sparse_normed = coo_matrix((dTD_normed.numpy().flatten(), (R_cat.numpy().flatten(), np.zeros(12*n_edges))),
                                   shape=(3*n_vertices, 1))

    ############################
    # Computing (c^T)WD vector #
    ############################

    # torch.Tensor(n_edges, 1, 4)
    c1TWD = torch.bmm(affine_components[:, 0, :][:, None, :], WD)
    c2TWD = torch.bmm(affine_components[:, 1, :][:, None, :], WD)
    c3TWD = torch.bmm(affine_components[:, 2, :][:, None, :], WD)

    c1TWD_normed = sqr[:, None, None] * c1TWD
    c2TWD_normed = sqr[:, None, None] * c2TWD
    c3TWD_normed = sqr[:, None, None] * c3TWD
    # torch.Tensor(n_edges, 12)
    cTWD = torch.zeros(n_edges, 12).double()
    cTWD[:, :4] = c1TWD[:, 0, :]
    cTWD[:, 4:8] = c2TWD[:, 0, :]
    cTWD[:, 8:] = c3TWD[:, 0, :]

    cTWD_normed = torch.zeros(n_edges, 12).double()
    cTWD_normed[:, :4] = c1TWD_normed[:, 0, :]
    cTWD_normed[:, 4:8] = c2TWD_normed[:, 0, :]
    cTWD_normed[:, 8:] = c3TWD_normed[:, 0, :]
    # torch.Tensor(n_edges, 12)
    R = parts_bitriangles_map
    R_cat = torch.zeros(n_edges, 12)
    R_cat[:, :4] = R
    R_cat[:, 4:8] = R + n_vertices
    R_cat[:, 8:] = R + 2*n_vertices

    # scipy.sparse(3 * n_vertices, 1)
    cTWD_sparse = coo_matrix((cTWD.numpy().flatten(), (R_cat.numpy().flatten(), np.zeros(12*n_edges))),
                             shape=(3*n_vertices, 1))

    cTWD_sparse_normed = coo_matrix((cTWD_normed.numpy().flatten(), (R_cat.numpy().flatten(), np.zeros(12*n_edges))),
                                    shape=(3*n_vertices, 1))

    a = ((sqr[:, None] * cTWD) / cTWD).numpy()

    b_sparse = cTWD_sparse + dTD_sparse

    b_sparse_normed = cTWD_sparse_normed + dTD_sparse_normed

    p_e = undeformed_vertices[:, :3].numpy().reshape((-1), order='F')
    E_0 = p_e.T @ A_0_sparse @ p_e - (2 * b_sparse.T @ p_e)[0] + torch.sum(affine_components ** 2).numpy() + torch.sum(translation_components ** 2).numpy()
    E_0_safe = (A_0_sparse_sqrt @ p_e).T @ (A_0_sparse_sqrt @ p_e) \
               + (D_e_sparse @ p_e).T @ (D_e_sparse @ p_e) \
               - (2 * b_sparse.T @ p_e)[0] \
               + torch.sum(affine_components ** 2).numpy() \
               + torch.sum(translation_components ** 2).numpy()
    print('E_0 energy:', E_0)
    print('E_0 energy (safe):', E_0_safe)
    print('E_0 quadratic term:', p_e.T @ A_0_sparse @ p_e)
    print('E_0 quadratic term (safe):', (A_0_sparse_sqrt @ p_e).T @ (A_0_sparse_sqrt @ p_e))
    print('DTD term:', p_e.T @ DTD_sparse @ p_e)
    print('DTD term (safe):', (D_e_sparse @ p_e).T @ (D_e_sparse @ p_e))
    print('E_0 linear term:', - (2 * b_sparse.T @ p_e)[0])
    print('E_0 constant term:', torch.sum(affine_components ** 2).numpy() + torch.sum(translation_components ** 2).numpy())
    print('Smoothness energy:', p_e.T @ (A_2_sqrt.T @ A_2_sqrt) @ p_e)
    print('Smoothness energy (safe):', ((A_2_sqrt @ p_e).T @ (A_2_sqrt @ p_e)))
    print('A_0_sqrt range:', np.max(A_0_sparse_sqrt), np.min(A_0_sparse_sqrt))
    print('A_2_sqrt range:', np.max(A_2_sqrt), np.min(A_2_sqrt))

    max_values = []
    for transform in edges_deformations:
        max_value = torch.max(transform)
        max_values += [max_value]
    print('Max:', max(max_values))
    print()

    b_vectors += [b_sparse]

    b_vectors_normed += [b_sparse_normed]

    output = {'A_matrices': A_matrices,
              'b_vectors': b_vectors,
              'A_0_sparse': A_0_sparse,
              'A_2_sparse': A_2_sparse,
              'A_3_sparse': A_3_sparse,
              'affine_components': affine_components,
              'translation_components': translation_components,
              'A_0_sparse_sqrt': A_0_sparse_sqrt,
              'D_e_sparse': D_e_sparse,
              'A_2_sparse_sqrt': A_2_sqrt}

    return output