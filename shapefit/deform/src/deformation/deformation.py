import multiprocessing
import sys
import time
from datetime import datetime

import scipy
import pickle
import os

from tqdm.autonotebook import tqdm
import numpy as np
import torch

from shapefit.deform.src.deformation.optimization import LBFGS_P
from shapefit.deform.src.deformation.utils import remove_degeneracies, plot_info
from shapefit.deform.src.deformation.transformation import edges_deformation_from_vertices, deformation_matrix
from shapefit.deform.src.deformation.preproc import compute_bitriangles


class Deformer:
    def __init__(self, 
                 mesh,
                 target_vertices_transforms,
                 sigma=0.5,
                 voxel_centers_nn=None,
                 surface_samples_nn=None,
                 voxel_centers_p2p=None,
                 surface_samples_p2p=None,
                 unlabeled_points=None,
                 sharp_edges=None,
                 kernel='gauss_kernel',
                 mapping='nn',
                 cuda=None,
                 device_mode='cpu',
                 deg_thr=1e-3,
                 ampl_factor=1,
                 verbose=1):
        """ Class for performing deformations on the set of meshes, labeled with different classes

        Parameters:
            parts (list<trimesh.base.Trimesh>): list of trimesh meshes
            initial_vertices_transforms (list<torch.Tensor>): list of transform matrices given by AlignmentHeatmap [n_parts, (n_vertices, 4, 4)]
            starting_vertices_transforms (list<torch.Tensor>): list of transform matrices given by user-defined deformations [n_parts, (n_vertices, 4, 4)]

        """
        self.cuda = cuda
        self.device_mode = device_mode
        
        if self.device_mode == 'gpu':
            self.device = [self.cuda[0], self.cuda[0], self.cuda[0], self.cuda[0]] if len(self.cuda) == 1 else \
                    [self.cuda[0], self.cuda[1], self.cuda[1], self.cuda[1]]
            if len(self.cuda) == 4:
                self.device = [self.cuda[0], self.cuda[1], self.cuda[2], self.cuda[3]]
        else:
            self.device = ['cpu', 'cpu', 'cpu', 'cpu']
        
        self.mesh = mesh
        self.alpha_0 = 1.0
        self.alpha_reg = 1.0
        self.alpha_data = 1.0
        self.verbose = verbose
        self.deg_thr = deg_thr
        self.ampl_factor = ampl_factor
        
        if verbose == 1:
            print('initializing parts descriptions...')
        self.num_total_vertices = 0
        self.target_vertices = []
        # list of parameters to optimize
        self.segmentation_soft_indicators = []
        self.voxel_centers_nn = voxel_centers_nn
        self.voxel_centers_p2p = voxel_centers_p2p
        self.kernel = kernel
        self.mapping = mapping
        self.surface_samples_nn = surface_samples_nn
        self.surface_samples_p2p = surface_samples_p2p
        self.unlabeled_points = unlabeled_points
        self.precond = None
        self.sharp_edges = sharp_edges
        self.vertices_approximation = None

        self.num_total_vertices += len(mesh.vertices)

        self.mesh_unique_edges = np.array(mesh.edges_unique)
        self.mesh_unique_faces, _ = np.unique(np.sort(mesh.faces, axis=1), axis=0, return_index=True)

        self.torch_vertices = torch.DoubleTensor(mesh.vertices)
        self.torch_unique_edges = torch.IntTensor(self.mesh_unique_edges)
        self.torch_unique_faces = torch.IntTensor(self.mesh_unique_faces)
        self.target_vertices_transforms = target_vertices_transforms.double()
        self.torch_vertices_4d = torch.DoubleTensor(np.hstack([mesh.vertices, np.ones(len(mesh.vertices))[:, None]]))

        torch_tmp_vertices = torch.zeros_like(self.torch_vertices_4d).double()
        for j, transform in enumerate(self.target_vertices_transforms):
            torch_tmp_vertices[j] = torch.mv(transform, self.torch_vertices_4d[j]).double()
        self.target_vertices = torch_tmp_vertices

        if not mesh.is_watertight:
            raise ValueError('Mesh should be watertight')

        if verbose == 1:
            print('computing bitriangles maps...')
        self.bitriangles_map = torch.IntTensor(compute_bitriangles(self.mesh_unique_faces, self.mesh_unique_edges))

        if verbose == 1:
            print('computing faces-to-edges maps...')
        # list<np.array>[n_parts, (n_faces, 3)]
        self.faces_to_edges_map = np.unique(np.sort(mesh.faces_unique_edges, axis=1), axis=0)
           
        if verbose == 1:
            print('computing adjacent edges for each face...')
        # list<np.array>[n_parts, (3*n_faces, 2)]
        tmp_adjacent_edges = torch.zeros((len(self.faces_to_edges_map) * 3, 2), dtype=torch.int)
        for j, face in enumerate(self.faces_to_edges_map):
            tmp_adjacent_edges[3*j] = torch.IntTensor([face[0], face[1]])
            tmp_adjacent_edges[3*j+1] = torch.IntTensor([face[0], face[2]])
            tmp_adjacent_edges[3*j+2] = torch.IntTensor([face[1], face[2]])
        self.adjacent_edges = tmp_adjacent_edges.long()
            
            
        (bitriangles_map_updated, 
         torch_vertices_4d_updated,
         target_vertices_updated,
         updated_edges,
         n_vertices,
         n_vertices_old,
         n_edges,
         n_edges_old) = remove_degeneracies(self.torch_vertices_4d,
                                            self.target_vertices,
                                            self.torch_unique_edges,
                                            self.bitriangles_map,
                                            self.deg_thr,
                                            self.ampl_factor)

        self.bitriangles_map_updated = bitriangles_map_updated
        self.torch_vertices_4d_updated = torch_vertices_4d_updated
        self.target_vertices_updated = target_vertices_updated
        self.updated_edges = updated_edges
        self.n_vertices = n_vertices
        self.n_vertices_old = n_vertices_old
        self.n_edges = n_edges
        self.n_edges_old = n_edges_old
        
        if verbose == 1:
            print('computing edges deformations...')
        # list<torch.Tensor>[n_parts, (n_edges, 4, 4)]
        self.target_edges_deformations = edges_deformation_from_vertices(self.torch_vertices_4d_updated,
                                                                         self.torch_unique_edges,
                                                                         self.target_vertices_updated,
                                                                         self.bitriangles_map_updated)

        if (self.voxel_centers_nn is not None) or (self.voxel_centers_p2p is not None):
            if verbose == 1:
                print('constructing soft indicators...')

            def indicator_factory(a):
                a = torch.Tensor(a).double()
                if self.cuda != None:
                    a = a.to(self.device[0])

                def aux_function(x):
                    kernel_value, count_close_vertices = 0, 0
                    if self.kernel == 'ep_kernel':
                        if self.mapping == 'p2p':
                            distances = torch.sqrt(torch.sum((a[:, None, :] - x[None, ...]) ** 2, dim=2))
                            close_points = (distances < sigma).double()
                            close_points_count = (distances < 0.02).double()
                            count_close_vertices = torch.sum(close_points_count, dim=1)
                            filter_count_close_points = (count_close_vertices <= 0).double()[:, None, None]
                            kernel_value = torch.sum(filter_count_close_points * close_points[..., None] * (a[:, None, :] - x[None, ...]) ** 2, dim=2)
                        elif self.mapping == 'nn':
                            kernel_value = torch.sum((a - x) ** 2, dim=1)

                        if self.mapping != 'p2p':
                            return kernel_value
                        else:
                            return kernel_value, count_close_vertices
                return aux_function
                
        if self.mapping == 'p2p':
            self.segmentation_soft_indicators_p2p = []
            for i in range(len(self.voxel_centers_p2p)):
                if len(self.voxel_centers_p2p[i]) == 0:
                    self.segmentation_soft_indicators_p2p += [0]
                else:
                    self.segmentation_soft_indicators_p2p += [indicator_factory(self.voxel_centers_p2p[i])]
        else:
            self.segmentation_soft_indicators_nn = indicator_factory(self.voxel_centers_nn)
            
        self.edges_deformations = []
        if verbose == 1:
            print('initialization done')
        
        # timing
        self.deformations_time = 0
        self.cost_time = 0
        self.backward_time = 0

    def save_inv_matrix(self, path, alpha_0=1, alpha_reg=1, alpha_sharp=1):
        
        os.makedirs(path, exist_ok=True)

        deformation_dict = deformation_matrix(self.target_vertices_updated,
                                              self.torch_vertices_4d_updated,
                                              self.torch_unique_edges,
                                              self.bitriangles_map_updated,
                                              self.target_edges_deformations,
                                              self.faces_to_edges_map,
                                              self.torch_unique_faces,
                                              self.sharp_edges,
                                              alpha_0,
                                              alpha_reg,
                                              alpha_sharp)

        self.A_matrices = deformation_dict['A_matrices']

        msize = self.A_matrices[0].shape[0]
        A_1 = self.A_matrices[0][:msize//3, :msize//3].tocsc()
        A_1_inv = scipy.sparse.linalg.inv(A_1)
        
        with open(os.path.join(path, 'A_1_inv.pkl'), 'wb') as f:
            pickle.dump(A_1_inv, f)
        
    def solve_data(self, 
                   iterations,
                   alpha_0=1,
                   alpha_reg=1,
                   alpha_sharp=1,
                   alpha_data_nn=1,
                   alpha_data_p2p=1,
                   alpha_quad=1,
                   lr=1e-0,
                   print_freq=5,
                   use_precond=False,
                   hessian_cpu=True,
                   load_hessian_path=None,
                   plot=True):

        deformation_dict = deformation_matrix(self.target_vertices_updated,
                                              self.torch_vertices_4d_updated,
                                              self.torch_unique_edges,
                                              self.bitriangles_map_updated,
                                              self.target_edges_deformations,
                                              self.faces_to_edges_map,
                                              self.torch_unique_faces,
                                              self.sharp_edges,
                                              alpha_0,
                                              alpha_reg,
                                              alpha_sharp)

        A_matrices = deformation_dict['A_matrices']
        b_vectors = deformation_dict['b_vectors']
        A_0_sparse = deformation_dict['A_0_sparse']
        A_2_sparse = deformation_dict['A_2_sparse']
        A_3_sparse = deformation_dict['A_3_sparse']
        affine_components = deformation_dict['affine_components']
        translation_components = deformation_dict['translation_components']
        A_0_sparse_sqrt = deformation_dict['A_0_sparse_sqrt']
        D_e_sparse = deformation_dict['D_e_sparse']
        A_2_sparse_sqrt = deformation_dict['A_2_sparse_sqrt']

        vertices = self.torch_vertices_4d_updated[:, :3]
        vertices_concat = torch.cat([vertices[:, 0],
                                     vertices[:, 1],
                                     vertices[:, 2]]).double()
        vertices_target = self.target_vertices_updated[:, :3]

        A_0_sparse /= self.n_vertices
        b_vectors[0] /= self.n_vertices
        A_2_sparse /= self.n_vertices
        A_3_sparse /= self.n_vertices

        A_0_sparse_sqrt /= np.sqrt(self.n_vertices)
        D_e_sparse /= np.sqrt(self.n_vertices)
        A_2_sparse_sqrt /= np.sqrt(self.n_vertices)
        
        try:
            lin_grad = alpha_0 * A_0_sparse @ vertices_concat.numpy() - alpha_0 * b_vectors[0].toarray().reshape(-1) + alpha_reg * A_2_sparse @ vertices_concat.numpy() + alpha_sharp * A_3_sparse @ vertices_concat.numpy()
            lin_grad_safe = alpha_0 * (A_0_sparse_sqrt.T) @ (A_0_sparse_sqrt @ vertices_concat.numpy()) + \
                            alpha_0 * (D_e_sparse.T) @ (D_e_sparse @ vertices_concat.numpy()) - \
                            alpha_0 * b_vectors[0].toarray().reshape(-1) + \
                            alpha_reg * (A_2_sparse_sqrt.T) @ (A_2_sparse_sqrt @ vertices_concat.numpy()) + \
                            alpha_sharp * A_3_sparse @ vertices_concat.numpy()
        except:
            lin_grad = alpha_0 * A_0_sparse @ vertices_concat.numpy() - alpha_0 * b_vectors[0].toarray().reshape(-1) + alpha_reg * A_2_sparse @ vertices_concat.numpy()
            lin_grad_safe = alpha_0 * (A_0_sparse_sqrt.T) @ (A_0_sparse_sqrt @ vertices_concat.numpy()) + \
                            alpha_0 * (D_e_sparse.T) @ (D_e_sparse @ vertices_concat.numpy()) - \
                            alpha_0 * b_vectors[0].toarray().reshape(-1) + \
                            alpha_reg * (A_2_sparse_sqrt.T) @ (A_2_sparse_sqrt @ vertices_concat.numpy())
        lin_grad = torch.Tensor(lin_grad)
        lin_grad_safe = torch.Tensor(lin_grad_safe)

        lin_grad = torch.cat([lin_grad[:self.n_vertices],
                              lin_grad[self.n_vertices:2 * self.n_vertices],
                              lin_grad[2 * self.n_vertices:]]).to(self.device[0])
        lin_grad_safe = torch.cat([lin_grad_safe[:self.n_vertices],
                                   lin_grad_safe[self.n_vertices:2 * self.n_vertices],
                                   lin_grad_safe[2 * self.n_vertices:]]).to(self.device[0])

        vertices_concat = torch.cat([vertices[:self.n_vertices, 0],
                                     vertices[:self.n_vertices, 1],
                                     vertices[:self.n_vertices, 2]]).double().clone().to(self.device[0]).requires_grad_(True)
        vertices_concat_target = torch.cat([vertices_target[:self.n_vertices, 0],
                                            vertices_target[:self.n_vertices, 1],
                                            vertices_target[:self.n_vertices, 2]]).double().clone()
        vertices_concat_init = vertices_concat.cpu().detach().clone()

        if use_precond:
            time1 = time.time()
            if self.precond is None:
                if load_hessian_path is None:
                    msize = A_matrices[0].shape[0]
                    # A_1 = A_matrices[0][:msize//3, :msize//3].tocsc()
                    # A_1_inv = scipy.sparse.linalg.inv(A_1)

                    A_1 = (self.n_vertices * A_0_sparse)[:msize//3, :msize//3].tocsc()
                    A_1_inv = scipy.sparse.linalg.inv(A_1)
                else:
                    with open(os.path.join(load_hessian_path, 'A_1_inv.pkl'), 'rb') as f:
                        A_1_inv = pickle.load(f)
                A_inv = A_1_inv.tocoo()

                if hessian_cpu:
                    values = A_inv.data
                    indices = np.vstack((A_inv.row, A_inv.col))

                    i = torch.LongTensor(indices)
                    v = torch.DoubleTensor(values)
                    shape = A_inv.shape

                    precond = torch.sparse_coo_tensor(i, v, torch.Size(shape)).double()
                else:         
                    A_1_inv = A_1_inv.tocoo()

                    values = A_1_inv.data
                    indices = np.vstack((A_1_inv.row, A_1_inv.col))           
                    i = torch.LongTensor(indices)
                    v = torch.DoubleTensor(values)
                    shape = A_1_inv.shape       
                    precond_1 = torch.sparse.DoubleTensor(i, v, torch.Size(shape)).to(self.device[0])

                    precond = precond_1
                    
                self.precond = precond
                self.indices = i
                self.values = v
                self.dense_shape = shape

            time2 = time.time()
            optimizer = LBFGS_P([vertices_concat], lr=lr, max_iter=2, history_size=1000, precond=self.precond, hessian_cpu=hessian_cpu,
                                indices=self.indices, values=self.values, dense_shape=self.dense_shape)
            
            if load_hessian_path is None:
                print('Hessian computing time:', time2 - time1, 's')
            else:
                print('Hessian loading time:', time2 - time1, 's')
        else:
            optimizer = torch.optim.LBFGS([vertices_concat], lr=lr, max_iter=2, history_size=1000)
                
        history_loss = []
        history_grad_norm = []
        history_data_grad_norm = []
        history_quad_loss = []
        history_smooth_loss = []
        history_p_deviation = []
        history_p_deviation_target = []
        history_p_deviation_mean = []
        history_p_deviation_target_mean = []
        history_data_deviation = []
        
        data_grad_norm = None

        if self.mapping == 'nn':
            num_parts = 1
        if self.mapping == 'p2p':
            num_parts = len(self.surface_samples_p2p)

        count_close_vertices_track = [[] for _ in range(num_parts)]

        # grad_data = True
        for k in tqdm(range(iterations)):
            # if k % 2 == 0:
            #     grad_data = True
            # else:
            #     grad_data = False
        
            def closure():

                time1 = time.time()
                
                nonlocal history_loss
                nonlocal history_data_grad_norm
                nonlocal history_data_deviation
                nonlocal lin_grad
                nonlocal lin_grad_safe
                nonlocal vertices_concat
                nonlocal data_grad_norm
                nonlocal vertices
                nonlocal count_close_vertices_track
                nonlocal num_parts
                # nonlocal grad_data
                
                optimizer.zero_grad()

                cost_data = 0

                v_coords = torch.cat([vertices_concat[:self.n_vertices][..., None],
                                      vertices_concat[self.n_vertices:2*self.n_vertices][..., None],
                                      vertices_concat[2*self.n_vertices:][..., None]], dim=1).double()
                
                for i in range(num_parts):
                    if self.mapping == 'nn':
                        cost_data += 1e5 * alpha_data_nn * (torch.sum((self.segmentation_soft_indicators_nn(v_coords[self.surface_samples_nn])) ** 2)) / len(self.surface_samples_nn)
                    elif self.mapping == 'p2p':
                        if self.segmentation_soft_indicators_p2p[i] != 0:
                            len_samples = sum([len(x) for x in self.surface_samples_p2p])
                            kernel_value, count_close_vertices = self.segmentation_soft_indicators_p2p[i](v_coords[self.surface_samples_p2p[i]])
                            count_close_vertices_track[i] += [count_close_vertices]
                            cost_data += alpha_data_p2p * torch.sum(kernel_value ** 2) / len_samples
                    else:
                        raise ValueError('This mapping does not exist')

                time2 = time.time()

                cost_data.backward()

                time3 = time.time()

                data_grad_norm = np.sqrt(torch.sum(vertices_concat.grad.cpu().detach() ** 2).numpy())
                history_data_grad_norm += [data_grad_norm]
                vertices_concat.grad += alpha_quad * lin_grad_safe.double()
                # if not grad_data:
                #     vertices_concat.grad = alpha_quad * lin_grad_safe.double()
                
                vertices_concat_detached = vertices_concat.cpu().detach()
                try:
                    lin_grad = alpha_0 * A_0_sparse @ vertices_concat_detached.numpy() - alpha_0 * b_vectors[0].toarray().reshape(-1) + alpha_reg * A_2_sparse @ vertices_concat_detached.numpy() + alpha_sharp * A_3_sparse @ vertices_concat_detached.numpy()
                    lin_grad_safe = alpha_0 * (A_0_sparse_sqrt.T) @ (A_0_sparse_sqrt @ vertices_concat_detached.numpy()) + \
                                    alpha_0 * (D_e_sparse.T) @ (D_e_sparse @ vertices_concat_detached.numpy()) - \
                                    alpha_0 * b_vectors[0].toarray().reshape(-1) + \
                                    alpha_reg * (A_2_sparse_sqrt.T) @ (A_2_sparse_sqrt @ vertices_concat_detached.numpy()) + \
                                    alpha_sharp * A_3_sparse @ vertices_concat_detached.numpy()
                except:
                    lin_grad = alpha_0 * A_0_sparse @ vertices_concat_detached.numpy() - alpha_0 * b_vectors[0].toarray().reshape(-1) + alpha_reg * A_2_sparse @ vertices_concat_detached.numpy()
                    lin_grad_safe = alpha_0 * (A_0_sparse_sqrt.T) @ (A_0_sparse_sqrt @ vertices_concat_detached.numpy()) + \
                                    alpha_0 * (D_e_sparse.T) @ (D_e_sparse @ vertices_concat_detached.numpy()) - \
                                    alpha_0 * b_vectors[0].toarray().reshape(-1) + \
                                    alpha_reg * (A_2_sparse_sqrt.T) @ (A_2_sparse_sqrt @ vertices_concat_detached.numpy())
                lin_grad = torch.Tensor(lin_grad)
                lin_grad_safe = torch.Tensor(lin_grad_safe)

                lin_grad = torch.cat([lin_grad[:self.n_vertices],
                                      lin_grad[self.n_vertices:2 * self.n_vertices],
                                      lin_grad[2 * self.n_vertices:]]).to(self.device[0])
                lin_grad_safe = torch.cat([lin_grad_safe[:self.n_vertices],
                                           lin_grad_safe[self.n_vertices:2 * self.n_vertices],
                                           lin_grad_safe[2 * self.n_vertices:]]).to(self.device[0])
            
                history_loss += [float(cost_data.cpu().detach().numpy())]
                if self.mapping != 'p2p':
                    history_data_deviation += [np.sqrt(np.sum((v_coords[self.surface_samples_nn].cpu().detach().numpy()-self.voxel_centers_nn) ** 2))]

                time4 = time.time()

                # print('Cost computation', time2 - time1)
                # print('Backward', time3 - time2)
                # print('Postcomputation', time4 - time3)
                # print()

                return cost_data

            time5 = time.time()
            if use_precond:
                optimizer.step(closure, self.cuda, self.device_mode)
            else:
                optimizer.step(closure)
            time6 = time.time()
            # print('Step', time6 - time5)

            vertices_concat_detached = vertices_concat.cpu().detach()

            energy = (vertices_concat_detached.numpy().T @ A_0_sparse @ vertices_concat_detached.numpy() - 2 * b_vectors[0].toarray().reshape(-1).T @ vertices_concat_detached.numpy() + torch.sum(affine_components ** 2).numpy() + torch.sum(translation_components ** 2).numpy()) / self.n_vertices
            energy_0 = (vertices_concat_detached.numpy().T @ A_0_sparse @ vertices_concat_detached.numpy() - 2 * b_vectors[0].toarray().reshape(-1).T @ vertices_concat_detached.numpy()) / self.n_vertices
            energy_0_safe = ((A_0_sparse_sqrt @ vertices_concat_detached.numpy()).T @ (A_0_sparse_sqrt @ vertices_concat_detached.numpy()) +
                             (D_e_sparse @ vertices_concat_detached.numpy()).T @ (D_e_sparse @ vertices_concat_detached.numpy()) -
                             2 * b_vectors[0].toarray().reshape(-1).T @ vertices_concat_detached.numpy()) / self.n_vertices
            
            energy_smooth = (vertices_concat_detached.numpy().T @ A_2_sparse @ vertices_concat_detached.numpy()) / self.n_vertices
            energy_smooth_safe = ((A_2_sparse_sqrt @ vertices_concat_detached.numpy()).T @ (A_2_sparse_sqrt @ vertices_concat_detached.numpy())) / self.n_vertices

            grad_norm = np.sqrt(torch.sum(lin_grad.cpu() ** 2).numpy())
            grad_norm_safe = np.sqrt(torch.sum(lin_grad_safe.cpu() ** 2).numpy())
            history_grad_norm += [grad_norm]
            history_quad_loss += [energy]
            history_smooth_loss += [energy_smooth]
            p_deviation = torch.sqrt(torch.sum((vertices_concat_init - vertices_concat_detached) ** 2))
            history_p_deviation += [p_deviation]
            p_deviation_target = torch.sqrt(torch.sum((vertices_concat_target - vertices_concat_detached) ** 2))
            history_p_deviation_target += [p_deviation_target]
            
            v_coords_init = vertices_concat_init.numpy().reshape((-1, 3), order='F')
            v_coords_detached = vertices_concat_detached.numpy().reshape((-1, 3), order='F')
            v_coords_target = vertices_concat_target.numpy().reshape((-1, 3), order='F')
            p_deviation_mean = np.sqrt(np.sum((v_coords_init - v_coords_detached) ** 2, axis=1).mean())
            history_p_deviation_mean += [p_deviation_mean]
            p_deviation_target_mean = np.sqrt(np.sum((v_coords_target - v_coords_detached) ** 2, axis=1).mean())
            history_p_deviation_target_mean += [p_deviation_target_mean]
            
            if k % print_freq == 0:
                print(
                    'p_dev {p_dev:3.3f} '
                    'E0 {E0:3.3f} E0_safe {E0_safe:3.3f} ES {ES:3.3f} ES_safe {ES_safe:3.3f} ED {ED:3.3f} '
                    'LG_norm {LG_norm:3.3f} LG_norm_safe {LG_norm_safe:3.3f} '
                    'DG_norm {DG_norm:3.3f}'.format(
                    **{
                        'timestamp': datetime.now().strftime("%d.%m.%Y %H:%M:%S.%f"),
                        'p_dev': p_deviation.numpy(),
                        'E0': energy_0,
                        'E0_safe': energy_0_safe,
                        'ES': energy_smooth,
                        'ES_safe': energy_smooth_safe,
                        'ED': history_loss[-1],
                        'LG_norm': grad_norm,
                        'LG_norm_safe': grad_norm_safe,
                        'DG_norm': data_grad_norm
                    }
                ), file=sys.stdout)
                sys.stdout.flush()

            time7 = time.time()
            # print('Printing', time7 - time6)
            # print()
            
        vertices_concat = vertices_concat.cpu().detach().numpy().reshape((-1, 3), order='F')
        self.vertices_approximation = vertices_concat
        
        if plot:
            plot_info(history_grad_norm, history_quad_loss,
                      history_smooth_loss, history_loss,
                      history_p_deviation, history_p_deviation_target,
                      history_p_deviation_mean, history_p_deviation_target_mean)

        
        if self.mapping != 'p2p':
            return history_data_deviation
        else: 
            return history_data_deviation, count_close_vertices_track, vertices_concat_init, vertices_concat_detached
    
    def get_new_parts_vertices(self):
        
        parts_vertices_pred_numpy = [x.detach().numpy()[:, :3] for x in self.parts_vertices_pred]
        return parts_vertices_pred_numpy 
    
    def get_starting_parts_vertices(self):
        
        parts_vertices_pred_numpy = [x.numpy()[:, :3] for x in self.parts_starting_vertices]
        return parts_vertices_pred_numpy 
    
    def get_initial_parts_vertices(self):

        parts_vertices_initial_numpy = self.target_vertices.numpy()[:, :3]
        return parts_vertices_initial_numpy 
    
    def get_indicator_functions(self):
        
        return self.segmentation_soft_indicators
    
    def get_bitriangles_map(self):
        
        return self.bitriangles_map
    
    def get_first_approximation(self):

        return self.vertices_approximation
    
    def print_times(self):
        
        print('Deformations time:', self.deformations_time, 's')
        print('Cost time:', self.cost_time, 's')
        print('Backward time:', self.backward_time, 's')

    def get_target_deformations(self):

        return self.target_edges_deformations