import numpy as np
from numpy.core.fromnumeric import reshape
import tensornetwork as tn
from tn_qsim.general_tn import TensorNetwork
from tn_qsim.mpo import MPO
from tn_qsim.utils import from_tn_to_quimb

class PEPS3D(TensorNetwork):
    """class of PEPS3D

    physical bond: 0, 1, ..., n-1
    for initialization:
        vertical virtual bond: n, n+1, ..., n+(height+1)-1, n+(height1), ..., n+(height+1)*width-1
        horizontal virtual bond: n+(height+1)*width, ..., n+(height+1)*width+(width+1)-1, ..., n + (height+1)*height + (height+1)*width-1

    Attributes:
        width (int) : PEPS width
        height (int) : PEPS height
        n (int) : the number of tensors
        edges (list of tn.Edge) : the list of each edge connected to each tensor
        nodes (list of tn.Node) : the list of each tensor
        top_nodes (list of tn.Node) : the list of top nodes for each qubits
        past_nodes (list of tn.Node) : the list of nodes that is not top
    """

    def __init__(self, tensors, height, width):
        self.n = len(tensors)
        self.height = height
        self.width = width
        self.path = None
        edge_info = []
        buff =2*self.n + (self.height+1)*self.width
        for h in range(self.height):
            for w in range(self.width):
                i = h*self.width + w
                shape = tensors[i].shape
                edge_info_original = [i, 2*self.n+w*(self.height+1)+h, buff+h*(self.width+1)+w+1, 2*self.n+w*(self.height+1)+h+1, buff+h*(self.width+1)+w]
                reshape_list = [s for s in shape if s != 1]
                edge_info_list = [edge_info_original[sidx] for sidx in range(5) if shape[sidx] != 1]
                tensors[i] = tensors[i].reshape(reshape_list)
                edge_info.append(edge_info_list)
        super().__init__(edge_info, tensors)
        self.tree, self.trace_tree = None, None
        self.top_nodes = [self.nodes[i] for i in range(self.height * self.width)]
        self.past_nodes = []


    def contract(self, algorithm=None, memory_limit=None, tree=None, path=None, visualize=False):
        """contract PEPS3D and generate full state

        Args:
            algorithm : the algorithm to find contraction path
            memory_limit : the maximum sp cost in contraction path
            tree (ctg.ContractionTree) : the contraction tree
            path (list of tuple of int) : the contraction path
            visualize (bool) : if visualize whole contraction process
        Returns:
            np.array: tensor after contraction
        """
        cp_nodes = tn.replicate_nodes(self.past_nodes + self.top_nodes)

        node_list = [node for node in cp_nodes]

        output_edge_order = []
        for i in range(self.n):
            output_edge_order.append(cp_nodes[-(self.n-i)][0])

        return self.contract_tree(node_list, output_edge_order, algorithm, memory_limit, tree, path, visualize=visualize)


    def prepare_amplitude(self, tensors):
        cp_nodes = tn.replicate_nodes(self.past_nodes + self.top_nodes)

        node_list = [cp_nodes[i] for i in range(len(cp_nodes) - self.n)]
        output_edge_order = []

        # contract product state first
        for i in range(self.n):
            # if tensors[i] is None, leave it open
            if tensors[i] is None:
                output_edge_order.append(cp_nodes[-(self.n-i)][0])
                node_list.append(cp_nodes[-(self.n-i)])
            else:
                state = tn.Node(tensors[i].conj())
                tn.connect(cp_nodes[-(self.n-i)][0], state[0])
                edge_order = [cp_nodes[-(self.n-i)].edges[j] for j in range(1, len(cp_nodes[-(self.n-i)].edges))]
                node_list.append(tn.contractors.auto([cp_nodes[-(self.n-i)], state], edge_order))
                cp_nodes[-(self.n-i)].tensor = None
                state.tensor = None

        return node_list, output_edge_order

    def find_amplitude_tree(self, tensors, algorithm=None, memory_limit=None, path=None, visualize=False):
        """contract amplitude with given product states (typically computational basis)

        Args:
            tensors (list of np.array) : the amplitude index represented by the list of tensor
            algorithm : the algorithm to find contraction path
            memory_limit : the maximum sp cost in contraction path
            path (list of tuple of int) : the contraction path
            visualize (bool) : if visualize whole contraction process

        Returns:
            tree (ctg.ContractionTree) : the contraction tree
            total_cost (int) : total temporal cost
            max_sp_cost (int) : max spatial cost
        """

        node_list, output_edge_order = self.prepare_amplitude(tensors)

        tree, total_cost, max_sp_cost = self.find_contract_tree(node_list, output_edge_order, algorithm, memory_limit, visualize=visualize)
        return tree, total_cost, max_sp_cost

    
    def amplitude(self, tensors, algorithm=None, memory_limit=None, tree=None, path=None, visualize=False):
        """contract amplitude with given product states (typically computational basis)

        Args:
            tensors (list of np.array) : the amplitude index represented by the list of tensor
            algorithm : the algorithm to find contraction path
            memory_limit : the maximum sp cost in contraction path
            tree (ctg.ContractionTree) : the contraction tree
            path (list of tuple of int) : the contraction path
            visualize (bool) : if visualize whole contraction process

        Returns:
            np.array: tensor after contraction
        """
        
        node_list, output_edge_order = self.prepare_amplitude(tensors)

        return self.contract_tree(node_list, output_edge_order, algorithm, memory_limit, tree, path, visualize=visualize)
    

    def find_amplitude_tree_by_quimb(self, tensors, algorithm=None, seq="ADCRS", visualize=False):
        """contract amplitude with given product states by using quimb (typically computational basis)

        Args:
            tensors (list of np.array) : the amplitude index represented by the list of tensor
            algorithm : the algorithm to find contraction path

        Returns:
            np.array: tensor after contraction
        """
        
        node_list, output_edge_order = self.prepare_amplitude(tensors)

        tn = from_tn_to_quimb(node_list, output_edge_order)

        if visualize:
            print(f"before simplification  |V|: {tn.num_tensors}, |E|: {tn.num_indices}")

        return self.find_contract_tree_by_quimb(tn, algorithm, seq)

    
    def amplitude_by_quimb(self, tensors, algorithm=None, tn=None, tree=None, target_size=None, gpu=True, thread=1, seq="ADCRS"):
        """contract amplitude with given product states by using quimb (typically computational basis)

        Args:
            tensors (list of np.array) : the amplitude index represented by the list of tensor
            algorithm : the algorithm to find contraction path

        Returns:
            np.array: tensor after contraction
        """
        
        if tn is None:
            node_list, output_edge_order = self.prepare_amplitude(tensors)
            tn = from_tn_to_quimb(node_list, output_edge_order)

        return self.contract_tree_by_quimb(tn, algorithm, tree, target_size, gpu, thread, seq)

    
    def apply_MPO(self, tidx, mpo):
        """ apply MPO
        
        Args:
            tidx (list of int) : list of qubit index we apply to.
            mpo (MPO) : MPO tensornetwork.
        """
        # single qubit gate - contract
        if len(tidx) == 1:
            node = mpo.nodes[0]
            node_contract_list = [node, self.top_nodes[tidx[0]]]
            node_edge_list = [node[0]]
            if len(self.top_nodes[tidx[0]].edges) > 1:
                node_edge_list = node_edge_list + self.top_nodes[tidx[0]][1:]
            one = tn.Node(np.array([1]))
            tn.connect(node[2], one[0])
            node_contract_list.append(one)
            one2 = tn.Node(np.array([1]))
            tn.connect(node[3], one2[0])
            node_contract_list.append(one2)
            tn.connect(node[1], self.top_nodes[tidx[0]][0])
            self.top_nodes[tidx[0]] = tn.contractors.auto(node_contract_list, output_edge_order=node_edge_list)
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
                tn.connect(node[1], self.top_nodes[tidx[i]][0])
                # if the top tensor is product state, absorb it
                if len(self.top_nodes[tidx[i]].edges) == 1:
                    node_edge_list = [node[e] for e in range(len(node.edges)) if e != 1]
                    self.top_nodes[tidx[i]] = tn.contractors.auto([node, self.top_nodes[tidx[i]]], output_edge_order=node_edge_list)
                else:
                    self.past_nodes.append(self.top_nodes[tidx[i]])
                    self.top_nodes[tidx[i]] = node