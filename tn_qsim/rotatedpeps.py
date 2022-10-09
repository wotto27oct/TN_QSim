from platform import node
import numpy as np
import tensornetwork as tn
from tn_qsim.mpo import MPO
from tn_qsim.mps import MPS
from tn_qsim.general_tn import TensorNetwork
from tn_qsim.utils import from_tn_to_quimb

class rotatedPEPS(TensorNetwork):
    """class of PEPS

    physical bond: 0, 1, ..., n-1
    vertical virtual bond: n, n+1, ..., n+(height+1)-1, n+(height1), ..., n+(height+1)*width-1
    horizontal virtual bond: n+(height+1)*width, ..., n+(height+1)*width+(width+1)-1, ..., n + (height+1)*height + (height+1)*width-1

    edge index order for each node: 0(physical) 1(up) 2(right) 3(down) 4(left)

    Attributes:
        height (int) : PEPS height
        width (int) : PEPS width
        n (int) : the number of tensors
        edges (list of tn.Edge) : the list of each edge connected to each tensor
        nodes (list of tn.Node) : the list of each tensor
        truncate_dim (int) : truncation dim of virtual bond, default None
        threshold_err (float) : the err threshold of singular values we keep
    """

    def __init__(self, tensors, height, width, truncate_dim=None, threshold_err=None, bmps_truncate_dim=None):
        self.n = len(tensors)
        self.height = height
        self.width = width
        self.path = None
        edge_info = []
        buff = self.n + (self.height+1)*self.width
        for h in range(self.height):
            for w in range(self.width):
                i = h*self.width + w
                if h % 2 == 0:
                    buff = h*self.n+self.n
                    edge_info.append([i, buff+2*w, buff+2*w+1, buff+self.n+2*w+1, buff+self.n+2*w])
                else:
                    buff = h*self.n+1+self.n
                    edge_info.append([i, buff+2*w, buff+2*w+1, buff+self.n+2*w+1, buff+self.n+2*w])

        print(edge_info)
        super().__init__(edge_info, tensors)
        self.truncate_dim = truncate_dim
        self.threshold_err = threshold_err
        self.bmps_truncate_dim = bmps_truncate_dim
        self.inner_tree = None

    
    def __clear_dangling(self, cp_nodes):
        output_edge_order = []
        def clear_dangling(node_idx, dangling_index):
            one = tn.Node(np.array([1]))
            tn.connect(cp_nodes[node_idx][dangling_index], one[0])
            edge_order = []
            for i in range(len(cp_nodes[node_idx].edges)):
                if i != dangling_index:
                    edge_order.append(cp_nodes[node_idx][i])
            cp_nodes[node_idx] = tn.contractors.auto([cp_nodes[node_idx], one], edge_order)

        # 4,3,2,1の順に消す
        for h in range(0, self.height, 2):
            if cp_nodes[self.width*h].get_dimension(4) == 1:
                clear_dangling(self.width*h, 4)
            else:
                output_edge_order.append(cp_nodes[h*self.width][4])
        for w in range(1, self.width):
            if cp_nodes[self.width*(self.height-1)+w].get_dimension(4) == 1:
                clear_dangling(self.width*(self.height-1)+w, 4)
            else:
                output_edge_order.append(cp_nodes[self.width*(self.height-1)+w][4])

        for h in range(1, self.height, 2):
            if cp_nodes[self.width*(h+1)-1].get_dimension(3) == 1:
                clear_dangling(self.width*(h+1)-1, 3)
            else:
                output_edge_order.append(cp_nodes[self.width*(h+1)-1][3])
        for w in range(self.width):
            if cp_nodes[self.width*(self.height-1)+w].get_dimension(3) == 1:
                clear_dangling(self.width*(self.height-1)+w, 3)
            else:
                output_edge_order.append(cp_nodes[self.width*(self.height-1)+w][3])

        for h in range(1, self.height, 2):
            if cp_nodes[self.width*(h+1)-1].get_dimension(2) == 1:
                clear_dangling(self.width*(h+1)-1, 2)
            else:
                output_edge_order.append(cp_nodes[self.width*(h+1)-1][2])
        for w in range(self.width):
            if cp_nodes[w].get_dimension(2) == 1:
                clear_dangling(w, 2)
            else:
                output_edge_order.append(cp_nodes[w][2])

        for h in range(0, self.height, 2):
            if cp_nodes[self.width*h].get_dimension(1) == 1:
                clear_dangling(self.width*h, 1)
            else:
                output_edge_order.append(cp_nodes[h*self.width][1])
        for w in range(1, self.width):
            if cp_nodes[w].get_dimension(1) == 1:
                clear_dangling(w, 1)
            else:
                output_edge_order.append(cp_nodes[w][1])

        return cp_nodes, output_edge_order

    
    def prepare_contract(self):
        cp_nodes = tn.replicate_nodes(self.nodes)

        # if there are dangling edges which dimension is 1, contract first
        cp_nodes, output_edge_order = self.__clear_dangling(cp_nodes)

        node_list = [node for node in cp_nodes]

        """for i in range(self.n):
            for dangling in cp_nodes[i].get_all_dangling():
                output_edge_order.append(dangling)"""
        for i in range(self.n):
            output_edge_order.append(cp_nodes[i][0])

        return node_list, output_edge_order


    def prepare_amplitude(self, tensors):
        cp_nodes = tn.replicate_nodes(self.nodes)

        # if there are dangling edges which dimension is 1, contract first
        cp_nodes, output_edge_order = self.__clear_dangling(cp_nodes)

        node_list = []

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

    def find_amplitude_tree_by_quimb(self, tensors, algorithm=None, seq="ADCRS", visualize=False):
        """contract amplitude with given product states by using quimb (typically computational basis)

        Args:
            tensors (list of np.array) : the amplitude index represented by the list of tensor
            algorithm : the algorithm to find contraction path

        Returns:
            np.array: tensor after contraction
        """
        
        node_list, output_edge_order = self.prepare_amplitude(tensors)

        tn, output_inds = from_tn_to_quimb(node_list, output_edge_order)

        if visualize:
            print(f"before simplification  |V|: {tn.num_tensors}, |E|: {tn.num_indices}")

        return self.find_contract_tree_by_quimb(tn, output_inds, algorithm, seq=seq)

    
    def amplitude(self, tensors, algorithm=None, tn=None, tree=None, target_size=None, gpu=True, thread=1, seq=None):
        """contract amplitude with given product states by using quimb (typically computational basis)

        Args:
            tensors (list of np.array) : the amplitude index represented by the list of tensor
            algorithm : the algorithm to find contraction path

        Returns:
            np.array: tensor after contraction
        """
        
        if tn is None:
            node_list, output_edge_order = self.prepare_amplitude(tensors)
            tn, _ = from_tn_to_quimb(node_list, output_edge_order)

        return self.contract_tree_by_quimb(tn, algorithm, tree, None, target_size, gpu, thread, seq)

    
    def amplitude_BMPS(self, tensors):
        """calculate amplitude with given product states (typically computational basis) using BMPS
        !!warning!! open qubits must be at the top row.

        Args:
            tensor (list of np.array) : the amplitude index represented by the list of tensor
        Returns:
            np.array: tensor after contraction
        """
        cp_nodes = tn.replicate_nodes(self.nodes)
        truncate_dim = self.bmps_truncate_dim

        # if there are dangling edges which dimension is 1, contract first
        cp_nodes, output_edge_order = self.__clear_dangling(cp_nodes)

        # contract product state first
        for i in range(self.n):
            if tensors[i] is not None:
                state = tn.Node(tensors[i].conj())
                tn.connect(cp_nodes[i][0], state[0])
                edge_order = [cp_nodes[i].edges[j] for j in range(1, len(cp_nodes[i].edges))]
                cp_nodes[i] = tn.contractors.auto([cp_nodes[i], state], edge_order)

        total_fidelity = 1.0
        # contract from upper side
        node_list = []
        # reshape upper tensors to mps-like
        for w in range(self.width):
            if w == 0:
                node_list.append(tn.Node(cp_nodes[w].tensor.reshape(1, -1)))
            elif w != self.width - 1:
                shape = cp_nodes[w].tensor.shape
                node_list.append(tn.Node(cp_nodes[w].tensor.reshape(1,shape[0],shape[1],1)))
                tn.connect(node_list[w-1][0], node_list[w][3])
            else:
                shape = cp_nodes[w].tensor.shape
                node_list.append(tn.Node(cp_nodes[w].tensor.reshape(shape[0],shape[1],1)))
                tn.connect(node_list[w-1][0], node_list[w][2])

        for h in range(self.height-1):
            if h % 2 == 0:
                for w in range(self.width-1):
                    print(h*self.width+w)
                    if w == 0:
                        # split right tensor
                        r_l_edges = node_list[1][2:]
                        r_r_edges = node_list[1][:2]
                        rR, rQ = tn.split_node_rq(node_list[1], r_l_edges, r_r_edges, edge_name="qr_right")
                        qr_right_edge = rR.get_edge("qr_right")
                        rR = rR.reorder_edges([qr_right_edge] + r_l_edges)
                        rQ = rQ.reorder_edges(r_r_edges + [qr_right_edge])

                        # contract and svd
                        apply_node = tn.Node(cp_nodes[(h+1)*self.width])
                        tn.connect(apply_node[0], node_list[0][1])
                        tn.connect(apply_node[1], rR[1])
                        svd_node_list = [node_list[0], rR, apply_node]
                        svd_node_edge_list = apply_node[2:] + [qr_right_edge]
                        svd_node = tn.contractors.optimal(svd_node_list, output_edge_order=svd_node_edge_list)

                        # split via SVD for truncation
                        U, s, Vh, trun_s = tn.split_node_full_svd(svd_node, svd_node[:2], svd_node[2:], truncate_dim)

                        # calc fidelity
                        s_sq = np.dot(np.diag(s.tensor), np.diag(s.tensor))
                        trun_s_sq = np.dot(trun_s, trun_s)
                        fidelity = s_sq / (s_sq + trun_s_sq)
                        total_fidelity *= fidelity

                        # reorder
                        l_edge_order = [s[0]] + U[:2]
                        node_list[0] = U.reorder_edges(l_edge_order)

                        r_edge_order = rQ[:2] + [s[0]]
                        node_list[1] = tn.contractors.optimal([s, Vh, rQ], output_edge_order=r_edge_order)
                    elif w != self.width-2:
                        # split left tensor
                        l_l_edges = node_list[w][2:]
                        l_r_edges = node_list[w][:2]
                        lQ, lR = tn.split_node_qr(node_list[w], l_l_edges, l_r_edges, edge_name="qr_left")
                        qr_left_edge = lQ.get_edge("qr_left")
                        lQ = lQ.reorder_edges([qr_left_edge] + l_l_edges)
                        lR = lR.reorder_edges(l_r_edges + [qr_left_edge])

                        # split right tensor
                        r_l_edges = node_list[w+1][2:]
                        r_r_edges = node_list[w+1][:2]
                        rR, rQ = tn.split_node_rq(node_list[w+1], r_l_edges, r_r_edges, edge_name="qr_right")
                        qr_right_edge = rR.get_edge("qr_right")
                        rR = rR.reorder_edges([qr_right_edge] + r_l_edges)
                        rQ = rQ.reorder_edges(r_r_edges + [qr_right_edge])

                        # contract and svd
                        apply_node = tn.Node(cp_nodes[(h+1)*self.width+w])
                        tn.connect(apply_node[0], lR[1])
                        tn.connect(apply_node[1], rR[1])
                        svd_node_list = [lR, rR, apply_node]
                        svd_node_edge_list = apply_node[2:] + [qr_left_edge] + [qr_right_edge]
                        svd_node = tn.contractors.optimal(svd_node_list, output_edge_order=svd_node_edge_list)
                        
                        # split via SVD for truncation
                        U, s, Vh, trun_s = tn.split_node_full_svd(svd_node, svd_node[:3], svd_node[3:], truncate_dim)

                        # calc fidelity
                        s_sq = np.dot(np.diag(s.tensor), np.diag(s.tensor))
                        trun_s_sq = np.dot(trun_s, trun_s)
                        fidelity = s_sq / (s_sq + trun_s_sq)
                        total_fidelity *= fidelity

                        # reorder
                        l_edge_order = [s[0]] + U[:2] + [lQ[1]]
                        node_list[w] = tn.contractors.optimal([lQ, U], output_edge_order=l_edge_order)

                        r_edge_order = rQ[:2] + [s[0]]
                        node_list[w+1] = tn.contractors.optimal([s, Vh, rQ], output_edge_order=r_edge_order)
                    elif w == self.width-2:
                        # split left tensor
                        l_l_edges = node_list[w][2:]
                        l_r_edges = node_list[w][:2]
                        lQ, lR = tn.split_node_qr(node_list[w], l_l_edges, l_r_edges, edge_name="qr_left")
                        qr_left_edge = lQ.get_edge("qr_left")
                        lQ = lQ.reorder_edges([qr_left_edge] + l_l_edges)
                        lR = lR.reorder_edges(l_r_edges + [qr_left_edge])

                        # split right tensor
                        r_l_edges = node_list[w+1][1:]
                        r_r_edges = node_list[w+1][:1]
                        rR, rQ = tn.split_node_rq(node_list[w+1], r_l_edges, r_r_edges, edge_name="qr_right")
                        qr_right_edge = rR.get_edge("qr_right")
                        rR = rR.reorder_edges([qr_right_edge] + r_l_edges)
                        rQ = rQ.reorder_edges(r_r_edges + [qr_right_edge])

                        # contract and svd
                        apply_node = tn.Node(cp_nodes[(h+1)*self.width+w].tensor)
                        tn.connect(apply_node[0], lR[1])
                        tn.connect(apply_node[1], rR[1])
                        svd_node_list = [lR, rR, apply_node]
                        svd_node_edge_list = apply_node[2:] + [qr_left_edge] + [qr_right_edge]
                        svd_node = tn.contractors.optimal(svd_node_list, output_edge_order=svd_node_edge_list)
                        
                        # split via SVD for truncation
                        U, s, Vh, trun_s = tn.split_node_full_svd(svd_node, svd_node[:3], svd_node[3:], truncate_dim)

                        # calc fidelity
                        s_sq = np.dot(np.diag(s.tensor), np.diag(s.tensor))
                        trun_s_sq = np.dot(trun_s, trun_s)
                        fidelity = s_sq / (s_sq + trun_s_sq)
                        total_fidelity *= fidelity
                        
                        # reorder
                        l_edge_order = [s[0]] + U[:2] + [lQ[1]]
                        node_list[w] = tn.contractors.optimal([lQ, U], output_edge_order=l_edge_order)

                        last_apply_node = tn.Node(cp_nodes[(h+1)*self.width+w+1].tensor)
                        tn.connect(last_apply_node[0], rQ[0])
                        r_edge_order =  [last_apply_node[1]] + [s[0]]
                        node_list[w+1] = tn.contractors.optimal([s, Vh, rQ, last_apply_node], output_edge_order=r_edge_order)

                    print(total_fidelity)
            elif h != self.height-2 and h % 2 == 1:
                for w in range(self.width-1, 0, -1):
                    print(h*self.width+w)
                    if w == self.width-1:
                        # split left tensor
                        l_l_edges = node_list[w-1][2:]
                        l_r_edges = node_list[w-1][:2]
                        lQ, lR = tn.split_node_qr(node_list[w-1], l_l_edges, l_r_edges, edge_name="qr_left")
                        qr_left_edge = lQ.get_edge("qr_left")
                        lQ = lQ.reorder_edges([qr_left_edge] + l_l_edges)
                        lR = lR.reorder_edges(l_r_edges + [qr_left_edge])

                        # contract and svd
                        apply_node = tn.Node(cp_nodes[(h+1)*self.width+w])
                        tn.connect(apply_node[0], lR[1])
                        tn.connect(apply_node[1], node_list[w][0])
                        svd_node_list = [node_list[w], lR, apply_node]
                        svd_node_edge_list = apply_node[2:] + [qr_left_edge]
                        svd_node = tn.contractors.optimal(svd_node_list, output_edge_order=svd_node_edge_list)

                        # split via SVD for truncation
                        U, s, Vh, trun_s = tn.split_node_full_svd(svd_node, svd_node[:2], svd_node[2:], truncate_dim)

                        # calc fidelity
                        s_sq = np.dot(np.diag(s.tensor), np.diag(s.tensor))
                        trun_s_sq = np.dot(trun_s, trun_s)
                        fidelity = s_sq / (s_sq + trun_s_sq)
                        total_fidelity *= fidelity

                        # reorder
                        r_edge_order = U[:2] + [s[0]]
                        node_list[w] = U.reorder_edges(r_edge_order)

                        l_edge_order = [s[0]] + lQ[1:]
                        node_list[w-1] = tn.contractors.optimal([s, Vh, lQ], output_edge_order=l_edge_order)

                    elif w != 1:
                        # split left tensor
                        l_l_edges = node_list[w-1][2:]
                        l_r_edges = node_list[w-1][:2]
                        lQ, lR = tn.split_node_qr(node_list[w-1], l_l_edges, l_r_edges, edge_name="qr_left")
                        qr_left_edge = lQ.get_edge("qr_left")
                        lQ = lQ.reorder_edges([qr_left_edge] + l_l_edges)
                        lR = lR.reorder_edges(l_r_edges + [qr_left_edge])

                        # split right tensor
                        r_l_edges = node_list[w][1:]
                        r_r_edges = node_list[w][:1]
                        rR, rQ = tn.split_node_rq(node_list[w], r_l_edges, r_r_edges, edge_name="qr_right")
                        qr_right_edge = rR.get_edge("qr_right")
                        rR = rR.reorder_edges([qr_right_edge] + r_l_edges)
                        rQ = rQ.reorder_edges(r_r_edges + [qr_right_edge])

                        # contract and svd
                        apply_node = tn.Node(cp_nodes[(h+1)*self.width+w])
                        tn.connect(apply_node[0], lR[1])
                        tn.connect(apply_node[1], rR[1])
                        svd_node_list = [lR, rR, apply_node]
                        svd_node_edge_list = [qr_right_edge] + apply_node[2:] + [qr_left_edge]
                        svd_node = tn.contractors.optimal(svd_node_list, output_edge_order=svd_node_edge_list)
                        
                        # split via SVD for truncation
                        U, s, Vh, trun_s = tn.split_node_full_svd(svd_node, svd_node[:3], svd_node[3:], truncate_dim)

                        # calc fidelity
                        s_sq = np.dot(np.diag(s.tensor), np.diag(s.tensor))
                        trun_s_sq = np.dot(trun_s, trun_s)
                        fidelity = s_sq / (s_sq + trun_s_sq)
                        total_fidelity *= fidelity

                        # reorder
                        r_edge_order = [rQ[0]] + U[1:3] + [s[0]]
                        node_list[w] = tn.contractors.optimal([rQ, U], output_edge_order=r_edge_order)

                        l_edge_order = [s[0]] + lQ[1:]
                        node_list[w-1] = tn.contractors.optimal([s, Vh, lQ], output_edge_order=l_edge_order)
                    elif w == 1:
                        # split left tensor
                        l_l_edges = node_list[w-1][2:]
                        l_r_edges = node_list[w-1][:2]
                        lQ, lR = tn.split_node_qr(node_list[w-1], l_l_edges, l_r_edges, edge_name="qr_left")
                        qr_left_edge = lQ.get_edge("qr_left")
                        lQ = lQ.reorder_edges([qr_left_edge] + l_l_edges)
                        lR = lR.reorder_edges(l_r_edges + [qr_left_edge])

                        # split right tensor
                        r_l_edges = node_list[w][1:]
                        r_r_edges = node_list[w][:1]
                        rR, rQ = tn.split_node_rq(node_list[w], r_l_edges, r_r_edges, edge_name="qr_right")
                        qr_right_edge = rR.get_edge("qr_right")
                        rR = rR.reorder_edges([qr_right_edge] + r_l_edges)
                        rQ = rQ.reorder_edges(r_r_edges + [qr_right_edge])

                        # contract and svd
                        apply_node = tn.Node(cp_nodes[(h+1)*self.width+w])
                        tn.connect(apply_node[0], lR[1])
                        tn.connect(apply_node[1], rR[1])
                        svd_node_list = [lR, rR, apply_node]
                        svd_node_edge_list = [qr_right_edge] + apply_node[2:] + [qr_left_edge]
                        svd_node = tn.contractors.optimal(svd_node_list, output_edge_order=svd_node_edge_list)
                        
                        # split via SVD for truncation
                        U, s, Vh, trun_s = tn.split_node_full_svd(svd_node, svd_node[:3], svd_node[3:], truncate_dim)

                        # calc fidelity
                        s_sq = np.dot(np.diag(s.tensor), np.diag(s.tensor))
                        trun_s_sq = np.dot(trun_s, trun_s)
                        fidelity = s_sq / (s_sq + trun_s_sq)
                        total_fidelity *= fidelity

                        # reorder
                        r_edge_order = [rQ[0]] + U[1:3] + [s[0]]
                        node_list[w] = tn.contractors.optimal([rQ, U], output_edge_order=r_edge_order)

                        last_apply_node = tn.Node(cp_nodes[(h+1)*self.width+w-1].tensor)
                        tn.connect(last_apply_node[0], lQ[1])
                        l_edge_order = [s[0]] + [last_apply_node[1]]
                        node_list[w-1] = tn.contractors.optimal([s, Vh, lQ, last_apply_node], output_edge_order=l_edge_order)
                    print(total_fidelity)
            else:
                # final layer 
                for w in range(self.width-1, 0, -1):
                    print(h*self.width+w)
                    if w != 1:
                        l_edge_order = node_list[w-1][2:]
                        apply_node = tn.Node(cp_nodes[(h+1)*self.width+w].tensor)
                        tn.connect(apply_node[0], node_list[w-1][1])
                        tn.connect(apply_node[1], node_list[w][0])
                        node_list[w-1] = tn.contractors.optimal([node_list[w], node_list[w-1], apply_node], output_edge_order=l_edge_order)
                    elif w == 1:
                        apply_node = tn.Node(cp_nodes[(h+1)*self.width+w].tensor)
                        tn.connect(apply_node[0], node_list[w-1][1])
                        tn.connect(apply_node[1], node_list[w][0])
                        last_apply_node = tn.Node(cp_nodes[(h+1)*self.width+w-1].tensor)
                        tn.connect(last_apply_node[0], node_list[w-1][2])
                        node_list[w-1] = tn.contractors.optimal([node_list[w], node_list[w-1], apply_node, last_apply_node])
                    print(total_fidelity)

        print(node_list[0].tensor)
        print(total_fidelity)
        return node_list[0].tensor, total_fidelity

    def apply_MPO_with_truncation(self, tidx, mpo, truncate_dim=None, last_dir=None):
        return self.apply_MPO(tidx, mpo, truncate_dim=truncate_dim, last_dir=last_dir)
    
    def apply_MPO(self, tidx, mpo, truncate_dim=None, last_dir=None):
        """ apply MPO with simple update
        
        Args:
            tidx (list of int) : list of qubit index we apply to.
            mpo (MPO) : MPO tensornetwork.
            truncate_dim (int) : truncation dim
        """
        if truncate_dim is None:
            truncate_dim = self.truncate_dim

        def return_dir(diff, hpos):
            if hpos % 2 == 0:
                if diff == -self.width:
                    return 1
                elif diff == -self.width+1:
                    return 2
                elif diff == self.width+1:
                    return 3
                elif diff == self.width:
                    return 4
                else:
                    raise ValueError("must be applied sequentially")
            else:
                if diff == -self.width-1:
                    return 1
                elif diff == -self.width:
                    return 2
                elif diff == self.width:
                    return 3
                elif diff == self.width-1:
                    return 4
                else:
                    raise ValueError("must be applied sequentially")


        edge_list = []
        node_list = []

        # not accurate
        total_fidelity = 1.0

        if len(tidx) == 1:
            node = mpo.nodes[0]
            node_contract_list = [node, self.nodes[tidx[0]]]
            node_edge_list = [node[0]] + [self.nodes[tidx[0]][j] for j in range(1, 5)]
            one = tn.Node(np.array([1]))
            tn.connect(node[2], one[0])
            node_contract_list.append(one)
            one2 = tn.Node(np.array([1]))
            tn.connect(node[3], one2[0])
            node_contract_list.append(one2)
            tn.connect(node[1], self.nodes[tidx[0]][0])
            node_list.append(tn.contractors.auto(node_contract_list, output_edge_order=node_edge_list))
        else:
            for i, node in enumerate(mpo.nodes):
                if i == 0:
                    node_contract_list = [node, self.nodes[tidx[i]]]
                    node_edge_list = [node[0]] + [self.nodes[tidx[i]][j] for j in range(1, 5)] + [node[3]]
                    one = tn.Node(np.array([1]))
                    tn.connect(node[2], one[0])
                    node_contract_list.append(one)
                    tn.connect(node[1], self.nodes[tidx[i]][0])
                    node_list.append(tn.contractors.optimal(node_contract_list, output_edge_order=node_edge_list))
                    edge_list.append(node_edge_list)
                else:
                    tn.connect(node[1], self.nodes[tidx[i]][0])

                    # calc direction  up:1 right:2 down:3 left:4
                    dir = return_dir(tidx[i] - tidx[i-1], tidx[i]//self.width)

                    # split nodes of PEPS via QR
                    l_l_edges = [node_list[i-1][j] for j in range(0, 5) if j != dir]
                    l_r_edges = [node_list[i-1][dir]] + [node_list[i-1][5]]
                    lQ, lR = tn.split_node_qr(node_list[i-1], l_l_edges, l_r_edges, edge_name="qr_left")
                    qr_left_edge = lQ.get_edge("qr_left")
                    lQ = lQ.reorder_edges(l_l_edges + [qr_left_edge])
                    lR = lR.reorder_edges(l_r_edges + [qr_left_edge])
                    r_l_edges = [self.nodes[tidx[i]][0]] + [self.nodes[tidx[i]][(dir+1)%4+1]]
                    r_r_edges = [self.nodes[tidx[i]][j] for j in range(1, 5) if j != (dir+1)%4+1]
                    rR, rQ = tn.split_node_rq(self.nodes[tidx[i]], r_l_edges, r_r_edges, edge_name="qr_right")
                    qr_right_edge = rR.get_edge("qr_right")
                    rR = rR.reorder_edges(r_l_edges + [qr_right_edge])
                    rQ = rQ.reorder_edges(r_r_edges + [qr_right_edge])

                    # contract left_R, right_R, node
                    svd_node_edge_list = None
                    svd_node_list = [lR, rR, node]
                    if i == mpo.n - 1 and last_dir is None:
                        one = tn.Node(np.array([1]))
                        tn.connect(node[3], one[0])
                        svd_node_edge_list = [qr_left_edge, node[0], qr_right_edge]
                        svd_node_list.append(one)
                    else:
                        svd_node_edge_list = [qr_left_edge, node[0], node[3], qr_right_edge]
                    svd_node = tn.contractors.optimal(svd_node_list, output_edge_order=svd_node_edge_list)

                    # split via SVD for truncation
                    U, s, Vh, trun_s = tn.split_node_full_svd(svd_node, [svd_node[0]], [svd_node[i] for i in range(1, len(svd_node.edges))], truncate_dim)

                    # calc fidelity
                    s_sq = np.dot(np.diag(s.tensor), np.diag(s.tensor))
                    trun_s_sq = np.dot(trun_s, trun_s)
                    fidelity = s_sq / (s_sq + trun_s_sq)
                    total_fidelity *= fidelity

                    # reorder and flatten edges
                    l_edge_order = [lQ.edges[i] for i in range(0, dir)] + [s[0]] + [lQ.edges[i] for i in range(dir, 4)]
                    node_list[i-1] = tn.contractors.optimal([lQ, U], output_edge_order=l_edge_order)
                    new_node = None
                    if i == mpo.n - 1 and last_dir is None:
                        r_edge_order = [Vh[1]] + [rQ.edges[i] for i in range(0, (dir+1)%4)] + [s[0]] + [rQ.edges[i] for i in range((dir+1)%4, 3)]
                        new_node = tn.contractors.optimal([s, Vh, rQ], output_edge_order=r_edge_order)
                    else:
                        r_edge_order = [Vh[1]] + [rQ.edges[i] for i in range(0, (dir+1)%4)] + [s[0]] + [rQ.edges[i] for i in range((dir+1)%4, 3)] + [Vh[2]]
                        new_node = tn.contractors.optimal([s, Vh, rQ], output_edge_order=r_edge_order)
                        if i == mpo.n-1 and last_dir is not None:
                            tn.flatten_edges([new_node[last_dir], new_node[5]])
                            reorder_list = [new_node[i] for i in range(last_dir)] + [new_node[4]] + [new_node[i] for i in range(last_dir, 4)]
                            new_node.reorder_edges(reorder_list)

                    node_list.append(new_node)

        for i in range(len(tidx)):
            self.nodes[tidx[i]] = node_list[i]

        return total_fidelity

    
    def prepare_Gamma(self, trun_node_idx):
        trun_node_idx, op_node_idx = trun_node_idx[0], trun_node_idx[1]
        trun_edge_idx = 0
        op_edge_idx = 0
        if trun_node_idx - op_node_idx == self.width:
            trun_edge_idx = 1
            op_edge_idx = 3
        elif trun_node_idx - op_node_idx == -1:
            trun_edge_idx = 2
            op_edge_idx = 4
        elif trun_node_idx - op_node_idx == -self.width:
            trun_edge_idx = 3
            op_edge_idx = 1
        else:
            trun_edge_idx = 4
            op_edge_idx = 2

        cp_nodes = tn.replicate_nodes(self.nodes)
        cp_nodes.extend(tn.replicate_nodes(self.nodes))

        for i in range(self.n):
            cp_nodes[i+self.n].tensor = cp_nodes[i+self.n].tensor.conj()
            tn.connect(cp_nodes[i][0], cp_nodes[i+self.n][0])
        
        cp_nodes[trun_node_idx][trun_edge_idx].disconnect("i", "j")
        cp_nodes[trun_node_idx+self.n][trun_edge_idx].disconnect("I", "J")
        edge_i = cp_nodes[trun_node_idx][trun_edge_idx]
        edge_I = cp_nodes[trun_node_idx+self.n][trun_edge_idx]
        edge_j = cp_nodes[op_node_idx][op_edge_idx]
        edge_J = cp_nodes[op_node_idx+self.n][op_edge_idx]
        output_edge_order = [edge_i, edge_I, edge_j, edge_J]

        # if there are dangling edges which dimension is 1, contract first (including inner dim)
        cp_nodes1, output_edge_order1 = self.__clear_dangling(cp_nodes[:self.n])
        cp_nodes2, output_edge_order2 = self.__clear_dangling(cp_nodes[self.n:])
        node_list = [node for node in cp_nodes1 + cp_nodes2]

        return trun_node_idx, op_node_idx, trun_edge_idx, op_edge_idx, node_list, output_edge_order


    def find_Gamma_tree(self, trun_node_idx, algorithm=None, memory_limit=None, visualize=False):
        """find contraction tree of Gamma

        Args:
            trun_node_idx (list ofint) : the node index connected to the target edge
            truncate_dim (int) : the target bond dimension
            trial (int) : the number of iterations
            visualize (bool) : if printing the optimization process or not
        """
        for i in range(self.n):
            self.nodes[i].name = f"node{i}"

        trun_node_idx, op_node_idx, trun_edge_idx, op_edge_idx, node_list, output_edge_order = self.prepare_Gamma(trun_node_idx)

        tree, cost, sp_cost = self.find_contract_tree(node_list, output_edge_order, algorithm, memory_limit, visualize=visualize)
        return tree, cost, sp_cost


    def find_optimal_truncation(self, trun_node_idx, min_truncate_dim=None, max_truncate_dim=None, truncate_buff=None, threshold=None, trials=None, gauge=False, algorithm=None, tnq=None, tree=None, target_size=None, gpu=True, thread=1, seq="ADCRS", visualize=False, calc_lim=None):
        """truncate the specified index using FET method

        Args:
            trun_node_idx (int) : the node index connected to the target edge
            truncate_dim (int) : the target bond dimension
            trial (int) : the number of iterations
            visualize (bool) : if printing the optimization process or not
        """
        for i in range(self.n):
            self.nodes[i].name = f"node{i}"

        trun_node_idx, op_node_idx, trun_edge_idx, op_edge_idx, node_list, output_edge_order = self.prepare_Gamma(trun_node_idx)

        if min_truncate_dim is not None and self.nodes[trun_node_idx][trun_edge_idx].dimension <= min_truncate_dim:
            print("trun_dim already satisfied")
            return 1.0

        max_truncate_dim = min(max_truncate_dim, self.nodes[trun_node_idx][trun_edge_idx].dimension)

        # includes tree == None case
        output_inds = None
        if tnq is None:
            tnq, output_inds = from_tn_to_quimb(node_list, output_edge_order)
            tnq, tree = self.find_contract_tree_by_quimb(tnq, output_inds, algorithm, seq, visualize)
            if calc_lim is not None and tree.total_flops() > calc_lim:
                print("Gamma calc lim exceeded.")
                return None
            #tnq, tree = self.find_Gamma_tree([trun_node_idx, op_node_idx], algorithm=algorithm, seq=seq, visualize=visualize)

        #print("calc Gamma...")
        Gamma = self.contract_tree_by_quimb(tn=tnq, tree=tree, output_inds=output_inds) #iIjJ

        U, Vh, Fid = None, None, 1.0
        truncate_dim = None
        if threshold is not None:
            for cur_truncate_dim in range(min_truncate_dim, max_truncate_dim+1, truncate_buff):
                if cur_truncate_dim == Gamma.shape[0]:
                    print("no truncation done")
                    return 1.0
                if not gauge:
                    U, Vh, Fid = self.find_optimal_truncation_by_Gamma(Gamma, cur_truncate_dim, trials, gpu=gpu, visualize=visualize)
                else:
                    U, Vh, Fid = self.fix_gauge_and_find_optimal_truncation_by_Gamma(Gamma, cur_truncate_dim, trials, gpu=gpu, visualize=visualize)
                
                truncate_dim = cur_truncate_dim
                if Fid > threshold:
                    break
        print(f"truncate dim: {truncate_dim}")

        """# truncate while Fid < threshold
        if truncate_dim is None:
            truncate_dim = 1
        U, Vh, Fid = None, None, 1.0
        nU, nVh, nFid = None, None, 1.0
        if threshold is not None:
            for cur_truncate_dim in range(Gamma.shape[0] - 1, truncate_dim-1, -1):
                nU, nVh, nFid = self.find_optimal_truncation_by_Gamma(Gamma, cur_truncate_dim, trials, visualize=visualize)
                if nFid < threshold:
                    truncate_dim = cur_truncate_dim + 1
                    break
                U, Vh, Fid = nU, nVh, nFid
        else:
            # must be some truncate_dim
            U, Vh, Fid = self.find_optimal_truncation_by_Gamma(Gamma, truncate_dim, trials, visualize=visualize)"""

        # if truncation is executed        
        if U is not None:
            Unode = tn.Node(U)
            Vhnode = tn.Node(Vh)
            tn.connect(Unode[1], Vhnode[0])

            left_edge, right_edge = self.nodes[trun_node_idx][trun_edge_idx].disconnect()
            if left_edge.node1 != self.nodes[trun_node_idx]:
                left_edge, right_edge = right_edge, left_edge
            op_node = self.nodes[op_node_idx]

            # connect self.node[trun_node_idx] and Unode
            tn.connect(left_edge, Unode[0])
            node_contract_list = [self.nodes[trun_node_idx], Unode]
            node_edge_list = []
            for i in range(5):
                if i == trun_edge_idx:
                    node_edge_list.append(Unode[1])
                else:
                    node_edge_list.append(self.nodes[trun_node_idx][i])
            self.nodes[trun_node_idx] = tn.contractors.auto(node_contract_list, output_edge_order=node_edge_list)

            # connect op_node and Vhnode
            tn.connect(Vhnode[1], right_edge)
            node_contract_list = [op_node, Vhnode]
            node_edge_list = []
            for i in range(5):
                if i == op_edge_idx:
                    node_edge_list.append(Vhnode[0])
                else:
                    node_edge_list.append(op_node[i])
            self.nodes[op_node_idx] = tn.contractors.auto(node_contract_list, output_edge_order=node_edge_list)

        print(f"truncated from {Gamma.shape[0]} to {truncate_dim}, Fidelity: {Fid}")
        return Fid