import numpy as np
from numpy.core.fromnumeric import reshape
import tensornetwork as tn
from tn_qsim.general_tn import TensorNetwork
from tn_qsim.mpo import MPO
from tn_qsim.utils import from_tn_to_quimb

class MERA3D(TensorNetwork):
    """class of infinite MERA3D

    physical bond: 0, 1, ..., n-1

    Args:
        n (int) : the number of qubits
        H (np.array) : Hamiltonian which we renormalize
        tidx (list of int) : the support of Hamiltonian

    Attributes:
        n (int) : the number of qubits
        Hnode (tn.Node) : Hamiltonian which we renormalize
        top_edges (list of tn.Edge) : the list of top edges
        down_edges (list of tn.Edge) : the list of down edges
        top_nodes (list of tn.Node) : the list of nodes
        down_nodes (list of tn.Node) : the list of adjoint nodes
    """

    def __init__(self, n, H, tidx):
        self.n = n
        self.top_edges = [None for _ in range(n)]
        self.down_edges = [None for _ in range(n)]
        self.nodes = []
        self.Hnode = tn.Node(H)
        for idx, t in enumerate(tidx):
            self.top_edges[t] = self.Hnode[idx]
            self.down_edges[t] = self.Hnode[idx+len(tidx)]

    
    def prepare_renormalize(self):
        cp_top_nodes = tn.replicate_nodes(self.top_nodes)
        cp_down_nodes = tn.replicate_nodes(self.down_nodes)

        node_list = [self.Hnode] + cp_top_nodes + cp_down_nodes

        # output edge of renormalized Hamiltonian
        output_edge_order = []
        for i in range(self.n):
            if self.top_edges[i] is not None:
                output_edge_order.append(self.top_edges[i])
        for i in range(self.n):
            if self.down_edges[i] is not None:
                output_edge_order.append(self.down_edges[i])
        
        return node_list, output_edge_order

    
    def find_renormalize_tree(self, algorithm=None, seq="", visualize=False):
        """compute contraction tree of renormalization

        Args:
            algorithm : the algorithm to find contraction path

        Returns:
            tn (TensorNetwork) : tn for contract
            tree (ContractionTree) : contraction tree for contract
        """

        node_list, output_edge_order = self.prepare_renormalize()

        tn, output_inds = from_tn_to_quimb(node_list, output_edge_order)

        if visualize:
            print(f"before simplification  |V|: {tn.num_tensors}, |E|: {tn.num_indices}")

        return self.find_contract_tree_by_quimb(tn, output_inds, algorithm, seq=seq)


    def renormalize(self, algorithm=None, tn=None, tree=None, target_size=None, gpu=True, thread=1, seq=None):
        """renormalize MERA

        Args:
            algorithm : the algorithm to find contraction path
            memory_limit : the maximum sp cost in contraction path
            tree (ctg.ContractionTree) : the contraction tree
            path (list of tuple of int) : the contraction path
            visualize (bool) : if visualize whole contraction process
        Returns:
            np.array: tensor after contraction
        """

        if tn is None:
            node_list, output_edge_order = self.prepare_renormalize()
            tn, _ = from_tn_to_quimb(node_list, output_edge_order)

        return self.contract_tree_by_quimb(tn, algorithm, tree, None, target_size, gpu, thread, seq)
    

    def apply_isometry(self, input_support, output_support, tensor):
        """ apply Unitary or Isometry
        
        Args:
            input_support (list of int) : list of qubit index we apply to.
            output_support (list of int) : list of qubit index of output.
            tensor (np.array) : the tensor of Unitary or Isometry.
        """
        Unode = tn.Node(tensor)
        Adnode = tn.Node(tensor.conj())
        self.top_nodes.append(Unode)
        self.down_nodes.append(Adnode)
        # connect edges
        for idx, tidx in enumerate(input_support):
            if self.top_edges[tidx] is None:
                tn.connect(Unode[idx+len(output_support)], Adnode[idx+len(output_support)])
            else:
                tn.connect(Unode[idx+len(output_support)], self.top_edges[tidx])
                tn.connect(Adnode[idx+len(output_support)], self.down_edges[tidx])
        
        for tidx in input_support:
            self.top_edges[tidx] = None
            self.down_edges[tidx] = None
        
        for idx, tidx in enumerate(output_support):
            self.top_edges[tidx] = Unode[idx]
            self.down_edges[tidx] = Adnode[idx]
    

    def apply_MPO(self, tidx, mpo):
        """ apply MPO
        
        Args:
            tidx (list of int) : list of qubit index we apply to.
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
            self.past_nodes.append(tn.contractors.auto(node_contract_list, output_edge_order=node_edge_list))
            tn.connect(self.past_nodes[-1][1], self.past_nodes[self.top_edges[tidx[0]][0]][self.top_edges[tidx[0]][1]])
            self.top_edges[tidx[0]] = [len(self.past_nodes)-1, 0]
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
                tn.connect(node[1], self.past_nodes[self.top_edges[tidx[i]][0]][self.top_edges[tidx[i]][1]])
                self.past_nodes.append(node)
                self.top_edges[tidx[i]] = [len(self.past_nodes)-1, 0]