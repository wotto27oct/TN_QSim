import numpy as np
from numpy.lib.arraysetops import ediff1d
import opt_einsum as oe
import cotengra as ctg
import tensornetwork as tn
import functools
import itertools
from collections import Counter
from itertools import chain

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


    def contract_by_oe(self, node_list, output_edge_order=None):
        input_sets = [set(node.edges) for node in node_list]
        output_set = set()
        for edge in tn.get_all_edges(node_list):
            if edge.is_dangling() or not set(edge.get_nodes()) <= set(node_list):
                output_set.add(edge)
        size_dict = {edge: edge.dimension for edge in tn.get_all_edges(node_list)}

        edge_alpha = dict()
        alpha_offset = 0
        for node in node_list:
            for e in node.edges:
                if not e in edge_alpha:
                    edge_alpha[e] = oe.get_symbol(alpha_offset)
                    alpha_offset += 1
        input_alpha = []
        for node in node_list:
            str = ""
            for e in node.edges:
                str += edge_alpha[e]
            input_alpha.append(str)
        print(input_alpha)

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
        print(output_alpha)

        #opt = ctg.HyperOptimizer(methods="kahypar")
        alg = functools.partial(oe.paths.greedy, memory_limit=2**28)
        #alg = functools.partial(oe.paths.optimal, memory_limit=None)
        #alg = functools.partial(ctg.HyperOptimizer(methods="greedy"), memory_limit=2**24)
        path = alg(input_sets, output_set, size_dict)
        #optimizer = oe.paths.DynamicProgramming()
        #path, pathinfo = oe.contract_path(input_sets, output_set, size_dict, optimize="greedy")
        print(path)
        #print(pathinfo)

        input_edge = [node.edges for node in node_list]
        total_cost = 0
        contract_einsum_str_list = []

        for cidx, contract_inds in enumerate(path):
            """contract_edge_list = []
            for c in contract_inds:
                contract_edge_list.append(input_alpha[c])

            freq = Counter(chain.from_iterable(contract_edge_list))

            remain_edge_alpha = ""
            contract_edge_alpha = ""
            print(freq)
            for idx in freq:
                if freq[idx] == 1:
                    remain_edge_alpha += idx
                else:
                    contract_edge_alpha += idx"""

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
            for re in remain_edge_list:
                remain_edge_alpha += edge_alpha[re]
                cost *= size_dict[re]
            for ce in contract_edge_list:
                cost *= size_dict[ce]
                

            einsum_str = ",".join(input_alpha) + "->" + remain_edge_alpha
            contract_einsum_str = ",".join(tensor_edge_alpha) + "->" + remain_edge_alpha
            print(f"{cidx}: {einsum_str}")
            print(f"contract {contract_einsum_str} cost:{cost}")
            total_cost += cost
            contract_einsum_str_list.append(contract_einsum_str)

            #右側のindexから消去
            contract_inds = tuple(sorted(list(contract_inds), reverse=True))
            for c in contract_inds:
                del input_alpha[c]
                del input_edge[c]
            input_alpha.append(remain_edge_alpha)
            input_edge.append(remain_edge_list)
        
        print(f"estimated total cost: {total_cost}")

        #path2, pathinfo = oe.contract_path(contract_einsum_str_list[12])



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