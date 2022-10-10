import numpy as np
from numpy.core.fromnumeric import reshape
import tensornetwork as tn
from tn_qsim.general_tn import TensorNetwork
from tn_qsim.mpo import MPO
from tn_qsim.utils import from_tn_to_quimb

class TN3D(TensorNetwork):
    """class of 3D tensor network

    physical bond: 0, 1, ..., n-1

    Args:
        n (int) : the number of qubits

    Attributes:
        n (int) : the number of tensors
        nodes (list[tn.Node]) : the list of each tensor
        top_nodes (list[tn.Node]) : the list of top nodes for each qubits
        top_edges (list[tuple[int, int]] : the list of top edges, (node_idx, edge_idx of node)
    """

    def __init__(self, tensors, n):
        self.n = n
        self.nodes = [tn.Node(tensor) for tensor in tensors]
        self.top_nodes = self.nodes
        self.top_edges = [(i, 0) for i in range(self.n)]
    
    def prepare_amplitude(self, tensors):
        cp_nodes = tn.replicate_nodes(self.nodes)

        node_list = [node for node in cp_nodes]
        output_edge_order = []

        # add product state
        for i in range(self.n):
            # if tensors[i] is None, leave it open
            if tensors[i] is None:
                output_edge_order.append(node_list[self.top_edges[i][0]][self.top_edges[i][1]])
            else:
                state = tn.Node(tensors[i].conj())
                tn.connect(node_list[self.top_edges[i][0]][self.top_edges[i][1]], state[0])
                node_list.append(state)

        return node_list, output_edge_order

    def find_amplitude_tree(self, tensors, algorithm=None, seq="ADCRS", visualize=False):
        """contract amplitude with given product states by using quimb (typically computational basis)

        Args:
            tensors (list[np.array]) : the amplitude index represented by the list of tensor
            algorithm : the algorithm to find contraction path

        Returns:
            np.array: tensor after contraction
        """

        node_list, output_edge_order = self.prepare_amplitude(tensors)

        tn, output_inds = from_tn_to_quimb(node_list, output_edge_order)

        if visualize:
            print(f"before simplification  |V|: {tn.num_tensors}, |E|: {tn.num_indices}")

        return self.find_contract_tree_by_quimb(tn, output_inds, algorithm, seq=seq)

    
    def amplitude(self, tensors, algorithm=None, tn=None, tree=None, target_size=None, gpu=True, thread=1, seq="", backend="jax", precision="complex64"):
        """contract amplitude with given product states by using quimb (typically computational basis)

        Args:
            tensors (list[np.array]) : the amplitude index represented by the list of tensor
            algorithm : the algorithm to find contraction path

        Returns:
            np.array: tensor after contraction
        """

        if tn is None:
            node_list, output_edge_order = self.prepare_amplitude(tensors)
            tn, output_inds = from_tn_to_quimb(node_list, output_edge_order)

        return self.contract_tree_by_quimb(tn, algorithm, tree, output_inds, target_size, gpu, thread, seq, backend=backend, precision=precision)


    def apply_MPO(self, tidx, mpo):
        """ apply MPO
        
        Args:
            tidx (list[int]) : list of qubit index we apply to.
            mpo (MPO) : MPO tensornetwork.
        """

        if len(tidx) == 1:
            node = mpo.nodes[0]
            node_contract_list = [node]
            node_edge_list = [node[0], node[1]]
            one = tn.Node(np.array([1]))
            tn.connect(node[2], one[0])
            node_contract_list.append(one)
            one2 = tn.Node(np.array([1]))
            tn.connect(node[3], one2[0])
            node_contract_list.append(one2)
            self.nodes.append(tn.contractors.auto(node_contract_list, output_edge_order=node_edge_list))
            tn.connect(self.nodes[-1][1], self.nodes[self.top_edges[tidx[0]][0]][self.top_edges[tidx[0]][1]])
            self.top_edges[tidx[0]] = [len(self.nodes)-1, 0]
        else:
            # multi qubit gate - leave it
            for i, node in enumerate(mpo.nodes):
                if i == 0:
                    one = tn.Node(np.array([1]))
                    tn.connect(node[2], one[0])
                    node_edge_list = [node[e] for e in range(len(node.edges)) if e != 2]
                    node = tn.contractors.auto([node, one], output_edge_order=node_edge_list)
                elif i == mpo.n - 1:
                    one = tn.Node(np.array([1]))
                    tn.connect(node[3], one[0])
                    node_edge_list = [node[e] for e in range(len(node.edges)) if e != 3]
                    node = tn.contractors.auto([node, one], output_edge_order=node_edge_list)
                tn.connect(node[1], self.nodes[self.top_edges[tidx[i]][0]][self.top_edges[tidx[i]][1]])
                self.nodes.append(node)
                self.top_edges[tidx[i]] = [len(self.nodes)-1, 0]