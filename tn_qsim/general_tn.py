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
from concurrent.futures import ThreadPoolExecutor

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


    def contract(self, output_edge_order=None):
        """contract the whole Tensor Network destructively

        Args:
            output_edge_order (list of tn.Edge) : the order of output edge
        
        Returns:
            np.array: tensor after contraction
        """

        if output_edge_order == None:
            return tn.contractors.auto(self.nodes, ignore_edge_order=True).tensor

        return tn.contractors.auto(self.nodes, output_edge_order=output_edge_order).tensor


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
            return x
        
        else:
            results = [
                tree.contract_slice(arrays, i)
                for i in range(tree.nslices)
            ]
            return tree.gather_slices(results)

    
    def find_optimal_truncation_by_Gamma(self, Gamma, truncate_dim, trials=10, visualize=False):
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
        Fid = np.dot(R.conj(), np.dot(A, R)) / np.dot(R.conj(), np.dot(B, R))
        if visualize:
            print(f"Fid before optimization: {Fid}")

        for i in range(trials):
            ## step1
            R = oe.contract("pq,qj->pj",S,Vh).flatten()
            P = oe.contract("iIjJ,ij,IP->PJ",Gamma,I,U.conj()).flatten()
            A = oe.contract("a,b->ab",P,P.conj())
            B = oe.contract("iIjJ,ip,IP->PJpj",Gamma,U,U.conj()).reshape(trun_dim*bond_dim, -1)

            #Fid = np.dot(R.conj(), np.dot(A, R)) / np.dot(R.conj(), np.dot(B, R))

            Rmax = np.dot(np.linalg.pinv(B), P)
            Fid = np.dot(Rmax.conj(), np.dot(A, Rmax)) / np.dot(Rmax.conj(), np.dot(B, Rmax))
            if visualize:
                print(f"fid at trial {i} step1: {Fid}")

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
                print(f"fid at trial {i} step2: {Fid}")

            U, stmp, Vhtmp = np.linalg.svd(Rmax.reshape(trun_dim, -1).T, full_matrices=False)
            S = np.dot(np.diag(stmp), Vhtmp)
        
        U = np.dot(U, S)

        return U, Vh


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