import numpy as np
import opt_einsum as oe
import cotengra as ctg
import tensornetwork as tn
import functools
from collections import Counter
from itertools import chain
from cotengra.core import ContractionTree
from tn_qsim.utils import from_nodes_to_str
import jax
import cupy as cp
from concurrent.futures import ThreadPoolExecutor
from jax.interpreters import xla
import functools
from tn_qsim.utils import from_tn_to_quimb
import time
#import cupy as cp

class TensorNetwork():
    """base class of Tensor Network

    Attributes:
        n (int) : the number of tensors
        edges (list of tn.Edge) : the list of each edge connected to each tensor
        nodes (list of tn.Node) : the list of each tensor

    """

    def __init__(self, edges, tensors):
        """initialization of Tensor Network
        Args:
            edges (list of list of int): the orderd indexes of each edge connected to each tensor
            tensors (list of np.array) : each tensor

        """
        self.n = len(tensors)
        if len(edges) != self.n:
            raise ValueError("the length of edges and tensors do not correspond")
        self.nodes = []
        self.edges = []
        self.tree = None

        edge_list = dict()

        for i in range(self.n):
            self.nodes.append(tn.Node(tensors[i], name=f"node {i}"))
            node_edges = self.nodes[i].get_all_edges()
            for j in range(len(edges[i])):
                if edges[i][j] in edge_list:
                    if edge_list[edges[i][j]][2] == True:
                        raise ValueError("HyperEdge is not allowed")
                    conn = tn.connect(self.nodes[edge_list[edges[i][j]][0]][edge_list[edges[i][j]][1]], self.nodes[i][j], name=f"edge {edges[i][j]}")
                    edge_list[edges[i][j]][2] = True
                    self.edges.append(conn)
                else:
                    edge_list[edges[i][j]] = [i, j, False]
                    node_edges[j].set_name(f"edge {edges[i][j]}")


    def contract_old(self, output_edge_order=None):
        """contract the whole Tensor Network destructively

        Args:
            output_edge_order (list of tn.Edge) : the order of output edge
        
        Returns:
            np.array: tensor after contraction
        """

        if output_edge_order == None:
            return tn.contractors.auto(self.nodes, ignore_edge_order=True).tensor

        return tn.contractors.auto(self.nodes, output_edge_order=output_edge_order).tensor

    
    def prepare_contract(self, output_edge_list):
        cp_nodes = tn.replicate_nodes(self.nodes)
        output_edge_order = []
        for nidx, eidx in output_edge_list:
            output_edge_order.append(cp_nodes[nidx][eidx])
        node_list = [node for node in cp_nodes]

        return node_list, output_edge_order


    def find_contraction_tree(self, output_edge_list, algorithm=None, seq="ADCRS", visualize=False):
        """contract the whole Tensor Network

        Args:
            output_edge_order (list of list of int) : [node, edge_idx],...
        
        Returns:
            np.array: tensor after contraction
        """
        
        node_list, output_edge_order = self.prepare_contract(output_edge_list)

        tn, output_inds = from_tn_to_quimb(node_list, output_edge_order)

        if visualize:
            print(f"before simplification  |V|: {tn.num_tensors}, |E|: {tn.num_indices}")

        return self.find_contract_tree_by_quimb(tn, output_inds, algorithm, seq=seq)

    
    def contract(self, output_edge_list, algorithm=None, tn=None, tree=None, target_size=None, gpu=True, thread=1, seq=None):
        """contract MERA and generate full state

        Args:
            algorithm : the algorithm to find contraction path

        Returns:
            np.array: tensor after contraction
        """

        if tn is None:
            node_list, output_edge_order = self.prepare_contract(output_edge_list)
            tn, _ = from_tn_to_quimb(node_list, output_edge_order)

        return self.contract_tree_by_quimb(tn, algorithm, tree, None, target_size, gpu, thread, seq)


    def visualize_tree(self, tree, node_list, output_edge_order=None, path=None, visualize=False):
        """calc contraction cost and visualize contract path for given tree and nodes

        Args:
            tree (ctg.ContractionTree) : the contraction tree
            node_list (list of tn.Node) : the nodes contracted
            output_edge_order (list of tn.Edge) : the order of output edge
            path (list of tuple of int) : the contraction path. If tree is None, this is converted to the tree
            visualize (bool) : if or not visualize contraction process
        
        Returns:
            total_cost (int) : the total contraction cost
            max_sp_cost (int) : the max space cost
            einsum_str_list (list of str) : contraction einsum for each step
            contract_einsum_str_list (list of str) : actual contraction einsum for each step
            edge_alpha_dims (dict) : dict from edge to alphabet
            cost_list (list of int) : contraction cost for each step
            total_cost (int) : total contraction cost
            sp_cost_list (list of int) : space contraction cost for each step
            max_sp_cost (int) : maximum space contraction cost
        """

        if tree == None:
            inputs, output, size_dict = from_nodes_to_str(node_list, output_edge_order)
            tree = ContractionTree.from_path(inputs, output, size_dict, path=path)
            print("tree.path: ", tree.path())
            self.tree = tree
        if path == None:
            path = tree.path()

        input_sets = [set(node.edges) for node in node_list]
        output_set = set()
        for edge in tn.get_all_edges(node_list):
            if edge.is_dangling() or not set(edge.get_nodes()) <= set(node_list):
                output_set.add(edge)
        size_dict = {edge: edge.dimension for edge in tn.get_all_edges(node_list)}

        edge_alpha = dict()
        edge_alpha_dims = dict()
        alpha_offset = 0
        for node in node_list:
            for e in node.edges:
                if not e in edge_alpha:
                    edge_alpha[e] = oe.get_symbol(alpha_offset)
                    alpha_offset += 1
                    edge_alpha_dims[edge_alpha[e]] = size_dict[e]

        input_alpha = []
        for node in node_list:
            str = ""
            for e in node.edges:
                str += edge_alpha[e]
            input_alpha.append(str)

        output_alpha = None
        if output_edge_order is not None:
            str = ""
            for e in output_edge_order:
                str += edge_alpha[e]
            output_alpha = str
        else:
            str = ""
            for e in output_set:
                str += edge_alpha[e]
            output_alpha = str

        input_edge = [node.edges for node in node_list]
        total_cost = 0
        max_sp_cost = 0
        cost_list = []
        sp_cost_list = []
        einsum_str_list = []
        contract_einsum_str_list = []

        for cidx, contract_inds in enumerate(path):
            tensor_edge_list = [input_edge[c] for c in contract_inds]
            tensor_edge_alpha = [input_alpha[c] for c in contract_inds]
            freq = Counter(chain.from_iterable(tensor_edge_list))
            remain_edge_list = []
            contract_edge_list = []

            for idx in freq:
                if freq[idx] == 1:
                    remain_edge_list.append(idx)
                else:
                    contract_edge_list.append(idx)
            
            remain_edge_alpha = ""
            cost = 1
            sp_cost = 1
            for re in remain_edge_list:
                remain_edge_alpha += edge_alpha[re]
                cost *= size_dict[re]
                sp_cost *= size_dict[re]
            for ce in contract_edge_list:
                cost *= size_dict[ce]

            if cidx == len(path) - 1:
                remain_edge_list = output_edge_order
                remain_edge_alpha = output_alpha
            
            einsum_str = ",".join(input_alpha) + "->" + remain_edge_alpha
            contract_einsum_str = ",".join(tensor_edge_alpha) + "->" + remain_edge_alpha
            total_cost += cost
            max_sp_cost = max(max_sp_cost, sp_cost)
            cost_list.append(cost)
            sp_cost_list.append(sp_cost)
            einsum_str_list.append(einsum_str)
            contract_einsum_str_list.append(contract_einsum_str)

            #右側のindexから消去
            contract_inds = tuple(sorted(list(contract_inds), reverse=True))
            for c in contract_inds:
                del input_alpha[c]
                del input_edge[c]
            input_alpha.append(remain_edge_alpha)
            input_edge.append(remain_edge_list)

        if visualize: 
            print(f"path: {path}")
            print(f"sliced index:", tree.sliced_inds)
            print("edge_dims:", edge_alpha_dims)
            for i in range(len(path)):
                print(f"{i}: {einsum_str_list[i]}")
                print(f"contract {contract_einsum_str_list[i]} cost:{cost_list[i]:,} sp_cost:{sp_cost_list[i]:,}")
            print(f"estimated total cost: {total_cost:,} max sp cost: {max_sp_cost:,}")
            print(f"total_flops: {tree.total_flops():,}, max_size: {tree.max_size():,}")
        
        return einsum_str_list, contract_einsum_str_list, edge_alpha_dims, cost_list, total_cost, sp_cost_list, max_sp_cost


    def find_contract_tree(self, node_list, output_edge_order=None, algorithm=None, memory_limit=2**28, visualize=False):
        """calc contract path for given input and algorithm

        Args:
            node_list (list of tn.Node) : the nodes contracted
            output_edge_order (list of tn.Edge) : the order of output edge
            algorithm : the algorithm to find contraction path
            memory_limit (int) : memory limit, default 2**28
            visualize (bool) : if or not visualize contraction process
        
        Returns:
            tree (ctg.ContractionTree) : the contraction tree
            total_cost (int) : the total contraction cost
            max_sp_cost (int) : the max space cost
        """
        
        inputs, output, size_dict = from_nodes_to_str(node_list, output_edge_order)

        if algorithm == None:
            algorithm = functools.partial(oe.paths.greedy, memory_limit=memory_limit)
        elif algorithm == "optimal":
            algorithm = functools.partial(oe.paths.optimal, memory_limit=memory_limit)

        path, tree = None, None

        if type(algorithm) is not ctg.HyperOptimizer:
            # oe.greedy etc...
            path = algorithm(inputs, output, size_dict)
            tree = ContractionTree.from_path(inputs, output, size_dict, path=path)
        else:
            tree = algorithm.search(inputs, output, size_dict)
            path = tree.path()

        einsum_str_list, contract_einsum_str_list, edge_alpha_dims, cost_list, total_cost, sp_cost_list, max_sp_cost = self.visualize_tree(tree, node_list, output_edge_order, visualize=visualize)

        return tree, total_cost, max_sp_cost

    def find_contract_tree_by_quimb(self, tnq, output_inds, algorithm=None, seq="ADCRS", visualize=False):
        """find contraction tree for given tnq (quimb.tensor.TensorNetwork) using quimb function

        Args:
            tnq (quimb.tensor.TensorNetwork) : tn we want to find contraction path
            output_inds (str) : output index ordering
            algorithm : the algorithm to find contraction path
            seq (str) : sequence of "ADCRS" for simplify. if seq is None, simplify is skipped
        """
        if visualize:
            print(f"before simplification  |V|: {tnq.num_tensors}, |E|: {tnq.num_indices}")
        if seq is not None:
            tnq = tnq.full_simplify(seq, output_inds=output_inds)
        if len(tnq.tensors) == 1:
            if visualize:
                print("tensor network becomes one tensor after simplification")
            inputs, output, size_dict = tnq.get_inputs_output_size_dict(output_inds=output_inds)
            tree = ctg.core.ContractionTree(inputs, output, size_dict, track_flops=True)
            tree.multiplicity = 1
            tree._flops = 1.0
            tree.sliced_inds = ""
            return tnq, tree
        tree = tnq.contraction_tree(optimize=algorithm, output_inds=output_inds)
        # print(tree.path())
        if visualize:
            print(f"after simplification  |V|: {tnq.num_tensors}, |E|: {tnq.num_indices}")
            print(f"slice: {tree.sliced_inds} tree cost: {tree.total_flops():,}, sp_cost: {tree.max_size():,}, log2_FLOP: {np.log2(tree.total_flops()):.4g} tree_width: {tree.contraction_width()}".encode("utf-8").strip())
        return tnq, tree

    def contract_tree_by_quimb(self, tn, algorithm=None, tree=None, output_inds=None, target_size=None, gpu=True, thread=1, seq="ADCRS", backend="jax", precision="complex64", is_visualize=False):   
        """execute contraction for given input and algorithm or tree

        Args:
            tn (quimb.tensor.TensorNetwork) : tn we contract by quimb
            output_inds (str) : output index ordering, if tree == None
            algorithm : the algorithm to find contraction path
            target_size : the target size we slice
            
        
        Returns:
            np.array: the tensor after contraction
        """

        if tree is None:
            tn, tree = self.find_contract_tree_by_quimb(tn, output_inds, algorithm, seq)
        if len(tn.tensors) == 1:
            if tn.tensors[0].ndim == 0:
                return tn.tensors[0].data
            else:
                #print("contract or reshape")
                #print(output_inds)
                inputs, output, _ = tn.get_inputs_output_size_dict(output_inds=output_inds)
                #print(inputs, output)
                #print(inputs[0] + "->" + output)
                #print(tn.tensors[0].shape)
                return np.einsum(inputs[0] + "->" + output, tn.tensors[0].data)
        tree_s = tree
        if target_size is not None:
            tree_s = tree.slice(target_size=target_size)
        
        if is_visualize:
            print(f"overhead : {tree_s.contraction_cost() / tree.contraction_cost():.2f} nslice: {tree_s.nslices}")

        if backend == "jax":
            if gpu and tree_s.total_flops() > 1e8:
                arrays = [jax.numpy.array(tensor.data) for tensor in tn.tensors]
                # use jax to use jit and GPU
                pool = ThreadPoolExecutor(1)

                contract_core_jit = jax.jit(functools.partial(tree_s.contract_core, backend="jax"))

                fs = [
                    pool.submit(contract_core_jit, tree_s.slice_arrays(arrays, i))
                    for i in range(tree_s.nslices)
                ]

                slices = (np.array(f.result()) for f in fs)

                x = tree_s.gather_slices(slices, progbar=True)
                return x
            else:
                arrays = [jax.numpy.array(tensor.data) for tensor in tn.tensors]
                # use jax to use jit
                contract_core_jit = jax.jit(functools.partial(tree_s.contract_core, backend="jax"), backend="cpu")
                
                slices = []

                for t in range(0, tree_s.nslices, thread):
                    if is_visualize:
                        print(f"{t}th parallel")
                    end_thread = tree_s.nslices if tree_s.nslices < (t+1)*thread else (t+1)*thread
                    #pool = ThreadPoolExecutor(end_thread - t*thread) if tree_s.nslices < (t+1)*thread else ThreadPoolExecutor(thread)
                    pool = ThreadPoolExecutor(1)

                    fs = [
                        pool.submit(contract_core_jit, tree_s.slice_arrays(arrays, i))
                        for i in range(t*thread, end_thread)
                    ]

                    slices = slices + [np.array(f.result()) for f in fs]

                x = tree_s.gather_slices(slices, progbar=False)
                return x
        elif backend == "cupy":
            if precision == "complex128":
                arrays = [cp.array(tensor.data, dtype=np.complex128) for tensor in tn.tensors]
            else:
                arrays = [cp.array(tensor.data, dtype=np.complex64) for tensor in tn.tensors]
            # use jax to use jit and GPU
            pool = ThreadPoolExecutor(1)

            contract_core = functools.partial(tree_s.contract_core, backend="cupy")

            fs = [
                pool.submit(contract_core, tree_s.slice_arrays(arrays, i))
                for i in range(tree_s.nslices)
            ]

            slices = (f.result().get() for f in fs)

            x = tree_s.gather_slices(slices, progbar=True)
            return x

    def contract_tree(self, node_list, output_edge_order=None, algorithm=None, memory_limit=2**28, tree=None, path=None, visualize=False):   
        """execute contraction for given input and algorithm or tree

        Args:
            node_list (list of tn.Node) : the nodes contracted
            output_edge_order (list of tn.Edge) : the order of output edge
            algorithm : the algorithm to find contraction path
            memory_limit (int) : memory limit, default 2**28
            path (tuple of tuple of int) : contraction path
            visualize (bool) : if or not visualize contraction process
        
        Returns:
            np.array: the tensor after contraction
        """

        if tree == None:
            if path is not None:
                inputs, output, size_dict = from_nodes_to_str(node_list, output_edge_order)
                tree = ContractionTree.from_path(inputs, output, size_dict, path=path)
            else:
                # if no tree or path is specified, find
                tree, _, _ = self.find_contract_tree(node_list, output_edge_order, algorithm, memory_limit)
        
        self.tree = tree
           
        if visualize: 
            self.visualize_tree(tree, node_list, output_edge_order, visualize=visualize)

        arrays = [node.tensor for node in node_list]

        if tree.total_flops() > 1e10:
            # use jax to use jit and GPU
            pool = ThreadPoolExecutor(1)

            contract_core_jit = jax.jit(tree.contract_core)

            fs = [
                pool.submit(contract_core_jit, tree.slice_arrays(arrays, i))
                for i in range(tree.nslices)
            ]

            slices = (np.array(f.result()) for f in fs)

            x = tree.gather_slices(slices, progbar=True)
            for node in node_list:
                node.tensor = None
            return x
        
        else:
            results = [
                tree.contract_slice(arrays, i)
                for i in range(tree.nslices)
            ]

            result = tree.gather_slices(results)
            #xla._xla_callable.cache_clear()
            for node in node_list:
                node.tensor = None
            return result

    
    """def fix_gauge_and_find_optimal_truncation_by_Gamma(self, Gamma, truncate_dim, trials=10, threshold=None, visualize=False):
        find optimal truncation U, Vh given Gamma and trun_dim
        Args:
            Gamma (np.array) : env-tensor Gamma_iIjJ
            turncate_dim (int) : target bond dimension
            trials (int) : the number of iteration
            visualize (bool) : print or not
        Returns:
            U (np.array) : left gauge tensor after optimization, shape (bond, trun)
            Vh (np.array) : right gauge tensor after optimization, shape (trun, bond)
        
        bond_dim = Gamma.shape[0]
        trun_dim = truncate_dim
        if visualize:
            print(f"bond: {bond_dim}, trun: {trun_dim}")

        # fix gauge
        # step 1
        leig, leigv = np.linalg.eig(Gamma.reshape(Gamma.shape[0]*Gamma.shape[1], -1))
        asc_order = np.argsort(np.abs(leig))
        lambda0 = leig[asc_order[-1]]
        L0 = leigv[:,asc_order[-1]]
        reig, reigv = np.linalg.eig(Gamma.reshape(Gamma.shape[0]*Gamma.shape[1], -1).T)
        asc_order = np.argsort(np.abs(reig))
        R0 = reigv[:,asc_order[-1]]

        # step 2
        L0 = L0 + 1e-10
        R0 = R0 + 1e-10
        ul, dl, ulh = np.linalg.svd(L0.reshape(Gamma.shape[0], -1), full_matrices=False)
        ur, dr, urh = np.linalg.svd(R0.reshape(Gamma.shape[0], -1), full_matrices=False)

        print(dl, dr)

        # step 3
        sigma_p = oe.contract("ab,bc,cd,de->ae",np.diag(np.sqrt(dl)),ul.conj().T,ur,np.diag(np.sqrt(dr)))
        wl, sigma, wrh = np.linalg.svd(sigma_p, full_matrices=False)
        print(sigma)
        sigma = np.diag(sigma)
        
        # step 4
        x = oe.contract("ab,bc,cd->ad",wl.conj().T,np.diag(np.sqrt(dl)),ul.conj().T)
        y = oe.contract("ab,bc,cd->ad",ur,np.diag(np.sqrt(dr)),wrh.conj().T)
        xinv = np.linalg.pinv(x)
        yinv = np.linalg.pinv(y)

        Gamma = oe.contract("iIjJ,ia,IA,bj,BJ->aAbB",Gamma,xinv,xinv.conj(),yinv,yinv.conj())

        U, s, Vh = np.linalg.svd(sigma)
        U = U[:,:trun_dim]
        S = np.diag(s[:trun_dim])
        Vh = Vh[:trun_dim, :]

        truncated_s = np.diag(sigma)[trun_dim:]
        if len(truncated_s[truncated_s>1e-9]) == 0:
            # perfect truncation
            Fid = 1.0
            U = np.dot(xinv, np.dot(U, S))
            Vh = np.dot(Vh, yinv)

            return U, Vh, Fid

        Fid = oe.contract("iIjJ,ij,IJ", Gamma, sigma, sigma)
        if visualize:
            print(f"Fid before truncation: {Fid}")

        R = oe.contract("pq,qj->pj",S,Vh).flatten()
        P = oe.contract("iIjJ,ij,IP->PJ",Gamma,sigma,U.conj()).flatten()
        A = oe.contract("a,b->ab",P,P.conj())
        B = oe.contract("iIjJ,ip,IP->PJpj",Gamma,U,U.conj()).reshape(trun_dim*bond_dim, -1)
        Fid = np.dot(R.conj(), np.dot(A, R)) / np.dot(R.conj(), np.dot(B, R))
        if visualize:
            print(f"Fid before optimization: {Fid}")
            print(np.dot(R.conj(), np.dot(A, R)), np.dot(R.conj(), np.dot(B, R)))

        for i in range(trials):
            ## step1
            R = oe.contract("pq,qj->pj",S,Vh).flatten()
            P = oe.contract("iIjJ,ij,IP->PJ",Gamma,sigma,U.conj()).flatten()
            A = oe.contract("a,b->ab",P,P.conj())
            B = oe.contract("iIjJ,ip,IP->PJpj",Gamma,U,U.conj()).reshape(trun_dim*bond_dim, -1)

            #Fid = np.dot(R.conj(), np.dot(A, R)) / np.dot(R.conj(), np.dot(B, R))

            Rmax = np.dot(np.linalg.pinv(B), P)
            Fid = np.dot(Rmax.conj(), np.dot(A, Rmax)) / np.dot(Rmax.conj(), np.dot(B, Rmax))
            if visualize:
                print(f"fid at trial {i} step1: {Fid}")

            Utmp, stmp, Vh = np.linalg.svd(Rmax.reshape(trun_dim, -1), full_matrices=False)
            S = np.dot(Utmp, np.diag(stmp))

            ## step2
            R = oe.contract("ip,pq->qi",U,S).flatten()
            P = oe.contract("iIjJ,ij,QJ->QI",Gamma,sigma,Vh.conj()).flatten()
            A = oe.contract("a,b->ab",P,P.conj())
            B = oe.contract("iIjJ,qj,QJ->QIqi",Gamma,Vh,Vh.conj()).reshape(trun_dim*bond_dim, -1)

            Rmax = np.dot(np.linalg.pinv(B), P)
            Fid = np.dot(Rmax.conj(), np.dot(A, Rmax)) / np.dot(Rmax.conj(), np.dot(B, Rmax))
            if visualize:
                print(f"fid at trial {i} step2: {Fid}")

            U, stmp, Vhtmp = np.linalg.svd(Rmax.reshape(trun_dim, -1).T, full_matrices=False)
            S = np.dot(np.diag(stmp), Vhtmp)
        
        R = oe.contract("pq,qj->pj",S,Vh).flatten()
        #P = oe.contract("iIjJ,ij,IP->PJ",Gamma,I,U.conj()).flatten()
        P = oe.contract("iIjJ,ij,IP->PJ",Gamma,sigma,U.conj()).flatten()
        A = oe.contract("a,b->ab",P,P.conj())
        B = oe.contract("iIjJ,ip,IP->PJpj",Gamma,U,U.conj()).reshape(trun_dim*bond_dim, -1)
        Fid = np.dot(R.conj(), np.dot(A, R)) / np.dot(R.conj(), np.dot(B, R))
        print(Fid)
        
        U = np.dot(xinv, np.dot(U, S)) / np.sqrt(Fid)
        Vh = np.dot(Vh, yinv)

        return U, Vh, Fid"""

    
    def find_optimal_truncation_by_Gamma(self, Gamma, truncate_dim, trials=10, gpu=False, visualize=False):
        """find optimal truncation U, Vh given Gamma and trun_dim
        Args:
            Gamma (np.array) : env-tensor Gamma_iIjJ
            turncate_dim (int) : target bond dimension
            trials (int) : the number of iteration
            visualize (bool) : print or not
        Returns:
            U (np.array) : left gauge tensor after optimization, shape (bond, trun)
            Vh (np.array) : right gauge tensor after optimization, shape (trun, bond)
        """
        bond_dim = Gamma.shape[0]
        trun_dim = truncate_dim
        if visualize:
            print(f"bond: {bond_dim}, trun: {trun_dim}")
        print("gpu", gpu)

        if not gpu:
            print("using cpu")
            I = np.eye(bond_dim)
            U, s, Vh = np.linalg.svd(I)
            U = U[:,:trun_dim]
            S = np.diag(s[:trun_dim])
            Vh = Vh[:trun_dim, :]

            Fid = oe.contract("iIiI", Gamma)
            if visualize:
                print(f"Fid before truncation: {Fid}")

            R = oe.contract("pq,qj->pj",S,Vh).flatten()
            P = oe.contract("iIjJ,ij,IP->PJ",Gamma,I,U.conj()).flatten()
            A = oe.contract("a,b->ab",P,P.conj())
            B = oe.contract("iIjJ,ip,IP->PJpj",Gamma,U,U.conj()).reshape(trun_dim*bond_dim, -1)
            trace = np.dot(R.conj(), np.dot(B, R))
            #if np.abs(trace) < 1e-10:
            #    print("initial trace too small")
            #    return None, None, 0.0

            Fid = np.dot(R.conj(), np.dot(A, R)) / trace
            if np.isnan(Fid) or np.isinf(Fid):
                print("initial trace too small")
                return None, None, 0.0

            if visualize:
                print(f"Fid before optimization: {Fid}")
            
            if Fid > 1 + 1e-6:
                print("numerically unstable")
                return None, None, 0.0
            
            Rmax = None

            past_fid = 0.0
            past_trace = trace
            first_fid = Fid
            firstU = np.dot(U, S) / np.sqrt(trace)
            firstVh = Vh
            try_idx = 0

            if trials == None:
                trials = 20

            while (try_idx < trials):
                ## step1
                R = oe.contract("pq,qj->pj",S,Vh).flatten()
                P = oe.contract("iIjJ,ij,IP->PJ",Gamma,I,U.conj()).flatten()
                A = oe.contract("a,b->ab",P,P.conj())
                B = oe.contract("iIjJ,ip,IP->PJpj",Gamma,U,U.conj()).reshape(trun_dim*bond_dim, -1)

                Rmax = np.dot(np.linalg.pinv(B), P)
                trace = np.dot(Rmax.conj(), np.dot(B, Rmax))
                Fid = np.dot(Rmax.conj(), np.dot(A, Rmax)) / trace
                if visualize:
                    print(f"fid at trial {try_idx} step1: {Fid}")
                if past_fid > Fid or Fid > 1.0 + 1e-6:
                    print("numerically unstable")
                    break
                elif np.abs(Fid - past_fid) < 1e-8:
                    print("no more improvement")
                    break
                past_fid = Fid
                past_trace = trace

                Utmp, stmp, Vh = np.linalg.svd(Rmax.reshape(trun_dim, -1), full_matrices=False)
                S = np.dot(Utmp, np.diag(stmp))

                """Binv = np.linalg.inv(B)
                Aprime = np.dot(Binv, A)
                eig, w = np.linalg.eig(Aprime)
                print(eig)"""

                ## step2
                R = oe.contract("ip,pq->qi",U,S).flatten()
                P = oe.contract("iIjJ,ij,QJ->QI",Gamma,I,Vh.conj()).flatten()
                A = oe.contract("a,b->ab",P,P.conj())
                B = oe.contract("iIjJ,qj,QJ->QIqi",Gamma,Vh,Vh.conj()).reshape(trun_dim*bond_dim, -1)

                Rmax = np.dot(np.linalg.pinv(B), P)
                trace = np.dot(Rmax.conj(), np.dot(B, Rmax))
                Fid = np.dot(Rmax.conj(), np.dot(A, Rmax)) / trace
                if visualize:
                    print(f"fid at trial {try_idx} step2: {Fid}")
                if past_fid > Fid or Fid > 1.0 + 1e-6:
                    print("numerically unstable")
                    break
                elif np.abs(Fid - past_fid) < 1e-8:
                    print("no more improvement")
                    break
                past_fid = Fid
                past_trace = trace

                U, stmp, Vhtmp = np.linalg.svd(Rmax.reshape(trun_dim, -1).T, full_matrices=False)
                S = np.dot(np.diag(stmp), Vhtmp)

                try_idx += 1
            
            if first_fid < past_fid:
                #print(past_fid, first_fid)
                #print(past_trace)
                #print(np.sqrt(past_trace))
                R = oe.contract("pq,qj->pj",S,Vh).flatten()
                P = oe.contract("iIjJ,ij,IP->PJ",Gamma,I,U.conj()).flatten()
                A = oe.contract("a,b->ab",P,P.conj())
                B = oe.contract("iIjJ,ip,IP->PJpj",Gamma,U,U.conj()).reshape(trun_dim*bond_dim, -1)
                trace = np.dot(R.conj(), np.dot(B, R))
                print("trace", trace)

                U = np.dot(U, S) / np.sqrt(past_trace)

                trace = oe.contract("ijIJ,ip,pj,IP,PJ",Gamma,U,Vh,U.conj(),Vh.conj())
                print("new trace:", trace)


                return U, Vh, past_fid
            else:
                return firstU, firstVh, first_fid
        else:
            print("using jax")
            Gamma = jax.numpy.array(Gamma)
            I = jax.numpy.eye(bond_dim)
            U, s, Vh = jax.numpy.linalg.svd(I)
            U = U[:,:trun_dim]
            S = jax.numpy.diag(s[:trun_dim])
            Vh = Vh[:trun_dim, :]

            Fid = oe.contract("iIiI", Gamma, backend="jax")
            if visualize:
                print(f"Fid before truncation: {Fid}")

            R = oe.contract("pq,qj->pj",S,Vh, backend="jax").flatten()
            P = oe.contract("iIjJ,ij,IP->PJ",Gamma,I,U.conj(), backend="jax").flatten()
            A = oe.contract("a,b->ab",P,P.conj(), backend="jax")
            B = oe.contract("iIjJ,ip,IP->PJpj",Gamma,U,U.conj(), backend="jax").reshape(trun_dim*bond_dim, -1)
            trace = jax.numpy.dot(R.conj(), jax.numpy.dot(B, R))
            Fid = jax.numpy.dot(R.conj(), jax.numpy.dot(A, R)) / trace
            if visualize:
                print(f"Fid before optimization: {Fid}")
            
            Rmax = None

            past_fid = 0.0
            past_trace = trace
            first_fid = Fid
            firstU = jax.numpy.dot(U, S) / np.sqrt(trace)
            firstVh = Vh
            try_idx = 0

            if trials == None:
                trials = 20

            while (try_idx < trials):
                ## step1
                R = oe.contract("pq,qj->pj",S,Vh, backend="jax").flatten()
                P = oe.contract("iIjJ,ij,IP->PJ",Gamma,I,U.conj(), backend="jax").flatten()
                A = oe.contract("a,b->ab",P,P.conj(), backend="jax")
                B = oe.contract("iIjJ,ip,IP->PJpj",Gamma,U,U.conj(), backend="jax").reshape(trun_dim*bond_dim, -1)

                Rmax = jax.numpy.dot(jax.numpy.linalg.pinv(B), P)
                trace = jax.numpy.dot(Rmax.conj(), jax.numpy.dot(B, Rmax))
                Fid = jax.numpy.dot(Rmax.conj(), jax.numpy.dot(A, Rmax)) / trace
                if visualize:
                    print(f"fid at trial {try_idx} step1: {Fid}")
                if past_fid > Fid or Fid > 1.0 + 1e-6:
                    print("numerically unstable")
                    break
                elif np.abs(Fid - past_fid) < 1e-5:
                    print("no more improvement")
                    break
                past_fid = Fid
                past_trace = trace

                Utmp, stmp, Vh = jax.numpy.linalg.svd(Rmax.reshape(trun_dim, -1), full_matrices=False)
                S = jax.numpy.dot(Utmp, jax.numpy.diag(stmp))

                """Binv = jax.numpy.linalg.inv(B)
                Aprime = jax.numpy.dot(Binv, A)
                eig, w = jax.numpy.linalg.eig(Aprime)
                print(eig)"""

                ## step2
                R = oe.contract("ip,pq->qi",U,S, backend="jax").flatten()
                P = oe.contract("iIjJ,ij,QJ->QI",Gamma,I,Vh.conj(), backend="jax").flatten()
                A = oe.contract("a,b->ab",P,P.conj(), backend="jax")
                B = oe.contract("iIjJ,qj,QJ->QIqi",Gamma,Vh,Vh.conj()).reshape(trun_dim*bond_dim, -1)

                Rmax = jax.numpy.dot(jax.numpy.linalg.pinv(B), P)
                trace = jax.numpy.dot(Rmax.conj(), jax.numpy.dot(B, Rmax))
                Fid = jax.numpy.dot(Rmax.conj(), jax.numpy.dot(A, Rmax)) / trace
                if visualize:
                    print(f"fid at trial {try_idx} step2: {Fid}")
                if past_fid > Fid or Fid > 1.0 + 1e-6:
                    print("numerically unstable")
                    break
                elif np.abs(Fid - past_fid) < 1e-5:
                    print("no more improvement")
                    break
                past_fid = Fid
                past_trace = trace

                U, stmp, Vhtmp = jax.numpy.linalg.svd(Rmax.reshape(trun_dim, -1).T, full_matrices=False)
                S = jax.numpy.dot(jax.numpy.diag(stmp), Vhtmp)

                try_idx += 1
            
            if first_fid < past_fid:
                #print(past_fid, first_fid)
                #print(past_trace)
                #print(np.sqrt(past_trace))
                U = jax.numpy.dot(U, S) / jax.numpy.sqrt(past_trace)

                return np.array(U), np.array(Vh), past_fid
            else:
                return np.array(firstU), np.array(firstVh), first_fid

    
    def fix_gauge_and_find_optimal_truncation_by_Gamma(self, Gamma, truncate_dim, trials=10, gpu=False, visualize=False):
        """find optimal truncation U, Vh given Gamma and trun_dim
        Args:
            Gamma (np.array) : env-tensor Gamma_iIjJ
            turncate_dim (int) : target bond dimension
            trials (int) : the number of iteration
            visualize (bool) : print or not
        Returns:
            U (np.array) : left gauge tensor after optimization, shape (bond, trun)
            Vh (np.array) : right gauge tensor after optimization, shape (trun, bond)
        """
        bond_dim = Gamma.shape[0]
        trun_dim = truncate_dim
        if visualize:
            print(f"bond: {bond_dim}, trun: {trun_dim}")

        # fix gauge
        # step 1
        leig, leigv = np.linalg.eig(Gamma.reshape(Gamma.shape[0]*Gamma.shape[1], -1))
        asc_order = np.argsort(np.abs(leig))
        lambda0 = leig[asc_order[-1]]
        L0 = leigv[:,asc_order[-1]]
        reig, reigv = np.linalg.eig(Gamma.reshape(Gamma.shape[0]*Gamma.shape[1], -1).T)
        asc_order = np.argsort(np.abs(reig))
        R0 = reigv[:,asc_order[-1]]

        # step 2
        L0 = L0 + 1e-10
        R0 = R0 + 1e-10
        ul, dl, ulh = np.linalg.svd(L0.reshape(Gamma.shape[0], -1), full_matrices=False)
        ur, dr, urh = np.linalg.svd(R0.reshape(Gamma.shape[0], -1), full_matrices=False)

        print(dl, dr)

        # step 3
        sigma_p = oe.contract("ab,bc,cd,de->ae",np.diag(np.sqrt(dl)),ul.conj().T,ur,np.diag(np.sqrt(dr)))
        wl, sigma, wrh = np.linalg.svd(sigma_p, full_matrices=False)
        print(sigma)
        sigma = np.diag(sigma)
        
        # step 4
        x = oe.contract("ab,bc,cd->ad",wl.conj().T,np.diag(np.sqrt(dl)),ul.conj().T)
        y = oe.contract("ab,bc,cd->ad",ur,np.diag(np.sqrt(dr)),wrh.conj().T)
        xinv = np.linalg.pinv(x)
        yinv = np.linalg.pinv(y)

        Gamma = oe.contract("iIjJ,ia,IA,bj,BJ->aAbB",Gamma,xinv,xinv.conj(),yinv,yinv.conj())

        U, s, Vh = np.linalg.svd(sigma)
        U = U[:,:trun_dim]
        S = np.diag(s[:trun_dim])
        Vh = Vh[:trun_dim, :]

        truncated_s = np.diag(sigma)[trun_dim:]
        if len(truncated_s[truncated_s>1e-9]) == 0:
            # perfect truncation
            Fid = 1.0
            U = np.dot(xinv, np.dot(U, S))
            Vh = np.dot(Vh, yinv)

            return U, Vh, Fid

        if not gpu and trun_dim < 8:
            Fid = oe.contract("iIiI", Gamma)
            if visualize:
                print(f"Fid before truncation: {Fid}")

            R = oe.contract("pq,qj->pj",S,Vh).flatten()
            P = oe.contract("iIjJ,ij,IP->PJ",Gamma,I,U.conj()).flatten()
            A = oe.contract("a,b->ab",P,P.conj())
            B = oe.contract("iIjJ,ip,IP->PJpj",Gamma,U,U.conj()).reshape(trun_dim*bond_dim, -1)
            Fid = np.dot(R.conj(), np.dot(A, R)) / np.dot(R.conj(), np.dot(B, R))
            if visualize:
                print(f"Fid before optimization: {Fid}")

            past_fid = Fid
            
            Rmax = None

            try_idx = 0

            if trials == None:
                trials = 1000

            while (try_idx < trials):
                ## step1
                R = oe.contract("pq,qj->pj",S,Vh).flatten()
                P = oe.contract("iIjJ,ij,IP->PJ",Gamma,I,U.conj()).flatten()
                A = oe.contract("a,b->ab",P,P.conj())
                B = oe.contract("iIjJ,ip,IP->PJpj",Gamma,U,U.conj()).reshape(trun_dim*bond_dim, -1)

                #Fid = np.dot(R.conj(), np.dot(A, R)) / np.dot(R.conj(), np.dot(B, R))

                Rmax = np.dot(np.linalg.pinv(B), P)
                Fid = np.dot(Rmax.conj(), np.dot(A, Rmax)) / np.dot(Rmax.conj(), np.dot(B, Rmax))
                if visualize:
                    print(f"fid at trial {try_idx} step1: {Fid}")
                if past_fid > Fid or Fid > 1.0 + 1e-6:
                    print("numerically unstable")
                    break
                elif np.abs(Fid - past_fid) < 1e-5:
                    print("no more improvement")
                    break
                past_fid = Fid

                Utmp, stmp, Vh = np.linalg.svd(Rmax.reshape(trun_dim, -1), full_matrices=False)
                S = np.dot(Utmp, np.diag(stmp))

                """Binv = np.linalg.inv(B)
                Aprime = np.dot(Binv, A)
                eig, w = np.linalg.eig(Aprime)
                print(eig)"""

                ## step2
                R = oe.contract("ip,pq->qi",U,S).flatten()
                P = oe.contract("iIjJ,ij,QJ->QI",Gamma,I,Vh.conj()).flatten()
                A = oe.contract("a,b->ab",P,P.conj())
                B = oe.contract("iIjJ,qj,QJ->QIqi",Gamma,Vh,Vh.conj()).reshape(trun_dim*bond_dim, -1)

                Rmax = np.dot(np.linalg.pinv(B), P)
                Fid = np.dot(Rmax.conj(), np.dot(A, Rmax)) / np.dot(Rmax.conj(), np.dot(B, Rmax))
                if visualize:
                    print(f"fid at trial {try_idx} step2: {Fid}")
                if past_fid > Fid or Fid > 1.0 + 1e-6:
                    print("numerically unstable")
                    break
                elif np.abs(Fid - past_fid) < 1e-5:
                    print("no more improvement")
                    break
                past_fid = Fid

                U, stmp, Vhtmp = np.linalg.svd(Rmax.reshape(trun_dim, -1).T, full_matrices=False)
                S = np.dot(np.diag(stmp), Vhtmp)

                try_idx += 1
            
            trace = np.dot(Rmax.conj(), np.dot(B, Rmax))
            U = np.dot(U, S) / np.sqrt(trace)

            return U, Vh, Fid
        
        else:
            print("using jax")
            Gamma = jax.numpy.array(Gamma)

            sigma, U, S, Vh = jax.numpy.array(sigma), jax.numpy.array(U), jax.numpy.array(S), jax.numpy.array(Vh)

            Fid = oe.contract("iIiI", Gamma, backend="jax")
            if visualize:
                print(f"Fid before truncation: {Fid}")

            R = oe.contract("pq,qj->pj",S,Vh, backend="jax").flatten()
            P = oe.contract("iIjJ,ij,IP->PJ",Gamma,sigma,U.conj(), backend="jax").flatten()
            A = oe.contract("a,b->ab",P,P.conj(), backend="jax")
            B = oe.contract("iIjJ,ip,IP->PJpj",Gamma,U,U.conj(), backend="jax").reshape(trun_dim*bond_dim, -1)
            trace = jax.numpy.dot(R.conj(), jax.numpy.dot(B, R))
            Fid = jax.numpy.dot(R.conj(), jax.numpy.dot(A, R)) / trace
            if visualize:
                print(f"Fid before optimization: {Fid}")
            
            Rmax = None

            past_fid = 0.0
            past_trace = trace
            first_fid = Fid
            firstU = jax.numpy.dot(U, S) / np.sqrt(trace)
            firstVh = Vh
            try_idx = 0

            if trials == None:
                trials = 20

            while (try_idx < trials):
                ## step1
                R = oe.contract("pq,qj->pj",S,Vh, backend="jax").flatten()
                P = oe.contract("iIjJ,ij,IP->PJ",Gamma,sigma,U.conj(), backend="jax").flatten()
                A = oe.contract("a,b->ab",P,P.conj(), backend="jax")
                B = oe.contract("iIjJ,ip,IP->PJpj",Gamma,U,U.conj(), backend="jax").reshape(trun_dim*bond_dim, -1)

                Rmax = jax.numpy.dot(jax.numpy.linalg.pinv(B), P)
                trace = jax.numpy.dot(Rmax.conj(), jax.numpy.dot(B, Rmax))
                Fid = jax.numpy.dot(Rmax.conj(), jax.numpy.dot(A, Rmax)) / trace
                if visualize:
                    print(f"fid at trial {try_idx} step1: {Fid}")
                if past_fid > Fid or Fid > 1.0 + 1e-6:
                    print("numerically unstable")
                    break
                elif np.abs(Fid - past_fid) < 1e-5:
                    print("no more improvement")
                    break
                past_fid = Fid
                past_trace = trace

                Utmp, stmp, Vh = jax.numpy.linalg.svd(Rmax.reshape(trun_dim, -1), full_matrices=False)
                S = jax.numpy.dot(Utmp, jax.numpy.diag(stmp))

                """Binv = jax.numpy.linalg.inv(B)
                Aprime = jax.numpy.dot(Binv, A)
                eig, w = jax.numpy.linalg.eig(Aprime)
                print(eig)"""

                ## step2
                R = oe.contract("ip,pq->qi",U,S, backend="jax").flatten()
                P = oe.contract("iIjJ,ij,QJ->QI",Gamma,sigma,Vh.conj(), backend="jax").flatten()
                A = oe.contract("a,b->ab",P,P.conj(), backend="jax")
                B = oe.contract("iIjJ,qj,QJ->QIqi",Gamma,Vh,Vh.conj()).reshape(trun_dim*bond_dim, -1)

                Rmax = jax.numpy.dot(jax.numpy.linalg.pinv(B), P)
                trace = jax.numpy.dot(Rmax.conj(), jax.numpy.dot(B, Rmax))
                Fid = jax.numpy.dot(Rmax.conj(), jax.numpy.dot(A, Rmax)) / trace
                if visualize:
                    print(f"fid at trial {try_idx} step2: {Fid}")
                if past_fid > Fid or Fid > 1.0 + 1e-6:
                    print("numerically unstable")
                    break
                elif np.abs(Fid - past_fid) < 1e-5:
                    print("no more improvement")
                    break
                past_fid = Fid
                past_trace = trace

                U, stmp, Vhtmp = jax.numpy.linalg.svd(Rmax.reshape(trun_dim, -1).T, full_matrices=False)
                S = jax.numpy.dot(jax.numpy.diag(stmp), Vhtmp)

                try_idx += 1
            
            if first_fid < past_fid:
                #print(past_fid, first_fid)
                #print(past_trace)
                #print(np.sqrt(past_trace))
                U = jax.numpy.dot(U, S) / jax.numpy.sqrt(past_trace)

                return np.array(U), np.array(Vh), past_fid
            else:
                return np.array(firstU), np.array(firstVh), first_fid


    def replace_tensors(self, tensor_indexes, r_tensors):
        """replace tensors
        
        Args:
            tensor_indexes (list of int): specify indexes of tensors which are replaced
            r_tensors (list of np.array): alternate tensors
        """

        for i, tidx in enumerate(tensor_indexes):
            if r_tensors[i].ndim != self.nodes[tidx].get_rank():
                raise ValueError("the ndim of replaced tensors do not correspond")
            self.nodes[tidx].set_tensor(r_tensors[i])

    @property
    def tensors(self):
        tensor_list = []
        for node in self.nodes:
            tensor_list.append(node.tensor)
        return tensor_list