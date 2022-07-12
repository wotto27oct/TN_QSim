import opt_einsum as oe
import tensornetwork as tn
import quimb.tensor as qtn
import numpy as np

def from_nodes_to_str(node_list, output_edge_order):
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
        str = []
        for e in node.edges:
            str.append(edge_alpha[e])
        input_alpha.append(str)

    output_alpha = None
    if output_edge_order is not None:
        str = []
        for e in output_edge_order:
            str.append(edge_alpha[e])
        output_alpha = str
    else:
        str = []
        for e in output_set:
            str.append(edge_alpha[e])
        output_alpha = str
    
    return input_alpha, output_alpha, edge_alpha_dims

def from_tn_to_quimb(node_list, output_edge_order):
    input_alpha, output_alpha, edge_alpha_dims = from_nodes_to_str(node_list, output_edge_order)
    #print(input_alpha)
    #print(output_alpha)
    tensors = []
    for idx, node in enumerate(node_list):
        tensors.append(qtn.Tensor(data=node.tensor, inds=list(input_alpha[idx])))
        node.tensor = None
    tn = qtn.TensorNetwork(tensors)
    output_alpha = "".join(output_alpha)
    #print(tn.get_equation(output_alpha))
    return tn, output_alpha

def is_WTG(Gamma, sigma):
    simga = np.real_if_close(sigma)
    bond_dim = Gamma.shape[0]
    for s in np.diag(sigma):
        if s < 0:
            print("Error! sigma is not positive")
            print(np.diag(sigma))
            return False
    rho_L = oe.contract("kj,kJ,iIjJ->iI",sigma,sigma,Gamma)
    rho_L = rho_L / np.trace(rho_L)
    eye_L = np.eye(bond_dim) / bond_dim
    if np.linalg.norm(rho_L - eye_L) > 1e-5:
        print("Error! rho_L is not proportional to eye_L")
        print(rho_L)
        print(eye_L)
        return False
    rho_R = oe.contract("ik,Ik,iIjJ->jJ",sigma,sigma,Gamma)
    rho_R = rho_R / np.trace(rho_R)
    eye_R = np.eye(bond_dim) / bond_dim
    if np.linalg.norm(rho_R - eye_R) > 1e-5:
        print("Error! rho_R is not proportional to eye_R")
        print(rho_R)
        print(eye_R)
        return False
    return True

def fix_gauge(Gamma, visualize=False):
    """find weight trace gauge
    Args:
        Gamma (np.array) : env-tensor Gamma_iIjJ
    Returns:
        Gamma (np.array) : new env-tensor Gamma_iIjJ
        sigma (np.array) : new bond matrix sigma_ij
        xinv (np.array) : new tensor to be contracted to the original tensor
        yinv (np.array)
    """
    # assume initial bond matrix is identity
    bond_dim = Gamma.shape[0]

    # fix gauge
    # step 1
    leig, leigv = np.linalg.eig(Gamma.reshape(Gamma.shape[0]*Gamma.shape[1], -1))
    if visualize:
        print("gamma for l:", Gamma.reshape(Gamma.shape[0]*Gamma.shape[1], -1)[:max(5, Gamma.shape[0]),:max(5, Gamma.shape[1])])
        print("leig:", leig)
        #print("leigv:", leigv)
    asc_order = np.argsort(np.abs(leig))
    lambda0 = leig[asc_order[-1]]
    # if the dominant eigenvector has degeneracy, sum them in the same weight
    L0 = leigv[:,asc_order[-1]]
    L0num = 1
    for idx in asc_order[::-1][1:]:
        if np.abs(leig[idx] - lambda0) < 1e-5:
            L0 += np.random.uniform() * leigv[:, idx]
            L0num += 1
        else:
            break
    #L0 /= L0num

    reig, reigv = np.linalg.eig(Gamma.reshape(Gamma.shape[0]*Gamma.shape[1], -1).T)
    if visualize:
        print("gamma for r:", Gamma.reshape(Gamma.shape[0]*Gamma.shape[1], -1).T[:max(5, Gamma.shape[0]),:max(5, Gamma.shape[1])])
        print("reig:", reig)
        #print("reigv:", reigv)
    asc_order = np.argsort(np.abs(reig))
    # if the dominant eigenvector has degeneracy, sum them in the same weight
    R0 = reigv[:,asc_order[-1]]
    R0num = 1
    for idx in asc_order[::-1][1:]:
        if np.abs(reig[idx] - lambda0) < 1e-5:
            R0 += np.random.uniform() * reigv[:, idx]
            R0num += 1
        else:
            break
    #R0 /= R0num

    if visualize:
        print("lambda0, L0:", lambda0, L0)
        print("R0:", R0)
        print("L0num, R0num:", L0num, R0num)

    # step 2
    L0 = L0 + 1e-10
    R0 = R0 + 1e-10
    ul, dl, ulh = np.linalg.svd(L0.reshape(Gamma.shape[0], -1), full_matrices=False)
    ur, dr, urh = np.linalg.svd(R0.reshape(Gamma.shape[0], -1), full_matrices=False)

    if visualize:
        print("ul, dl, ulh", ul, dl, ulh)
        print("ur, dr, urh", ur, dr, urh)

    #print(dl, dr)

    # step 3
    sigma_p = oe.contract("ab,bc,cd,de->ae",np.diag(np.sqrt(dl)),ul.conj().T,ur,np.diag(np.sqrt(dr)))
    wl, sigma, wrh = np.linalg.svd(sigma_p, full_matrices=False)
    #print(sigma)
    sigma = np.diag(sigma)
    
    # step 4
    x = oe.contract("ab,bc,cd->ad",wl.conj().T,np.diag(np.sqrt(dl)),ul.conj().T)
    y = oe.contract("ab,bc,cd->ad",ur,np.diag(np.sqrt(dr)),wrh.conj().T)
    xinv = np.linalg.pinv(x)
    yinv = np.linalg.pinv(y)

    if visualize:
        print("xinv, yinv", xinv, yinv)

    Gamma = oe.contract("iIjJ,ia,IA,bj,BJ->aAbB",Gamma,xinv,xinv.conj(),yinv,yinv.conj())

    return Gamma, sigma, xinv, yinv

def calc_cycle_entropy(Gamma, sigma):
    bond_dim = Gamma.shape[0]
    Gamma_sigma_L = oe.contract("kj,KJ,iIjJ->iIkK",sigma,sigma,Gamma).reshape(bond_dim**2, -1)
    eig, _ = np.linalg.eig(Gamma_sigma_L)
    eig = eig[eig > 1e-15]
    eig = np.abs(eig) / np.sum(np.abs(eig))
    return eig, -np.dot(eig, np.log2(eig+1e-15))

def calc_optimal_truncation(Gamma, sigma, truncate_dim, precision=1e-10, trials=50, visualize=False):
    """calc optimal truncation given Gamma, sigma
    Args:
        Gamma (np.array) : env-tensor Gamma_iIjJ
        sigma (np.array) : bond matrix sigma_ij
        truncate_dim : target bond dimension
        precision (float) : ignore improvement under precision, return unstable if fidelity > 1 + 10*precision
        trials (int) : trial num
    Returns:
        U, S, Vh (np.array) : optimal truncated tensor, all matrix
        fidelity, trace (float) : fidelity, trace after optimal truncation
    """
    if Gamma.shape[0] <= truncate_dim:
        if visualize:
            print("truncate dim already satistfied")
        return None, None, None, 0.0, 0.0
        
    if visualize:
        print(f"truncate from {Gamma.shape[0]} to {truncate_dim}")

    Fid = oe.contract("iIjJ,ij,IJ", Gamma, sigma, sigma)
    if visualize:
        print(f"Fid before truncation: {Fid}")

    trun_dim = truncate_dim
    bond_dim = Gamma.shape[0]

    U, s, Vh = np.linalg.svd(sigma)
    perm = np.random.permutation(sigma.shape[0])
    U = U[perm]
    Vh = Vh.T[perm].T
    U = U[:, :trun_dim]
    S = np.diag(s[:trun_dim])
    Vh = Vh[:trun_dim, :]

    R = oe.contract("pq,qj->pj",S,Vh).flatten()
    P = oe.contract("iIjJ,ij,IP->PJ",Gamma,sigma,U.conj()).flatten()
    A = oe.contract("a,b->ab",P,P.conj())
    B = oe.contract("iIjJ,ip,IP->PJpj",Gamma,U,U.conj()).reshape(trun_dim*bond_dim, -1)
    trace = np.dot(R.conj(), np.dot(B, R))
    Fid = np.dot(R.conj(), np.dot(A, R)) / trace

    if visualize:
        print(f"initial trace: {trace}")
        print(f"Fid before optimization: {Fid}")

    if np.isnan(Fid) or np.isinf(Fid):
        print("initial trace too small")
        return None, None, None, 0.0, 0.0
    
    if Fid > 1 + 1e-4:
        print("numerically unstable")
        return None, None, None, 0.0, 0.0
    
    Rmax = None

    past_fid = Fid
    past_trace = trace
    try_idx = 0

    init_trial = 5

    while (try_idx < trials):
        ## step1
        R = oe.contract("pq,qj->pj",S,Vh).flatten()
        P = oe.contract("iIjJ,ij,IP->PJ",Gamma,sigma,U.conj()).flatten()
        A = oe.contract("a,b->ab",P,P.conj())
        B = oe.contract("iIjJ,ip,IP->PJpj",Gamma,U,U.conj()).reshape(trun_dim*bond_dim, -1)

        if try_idx < init_trial:
            B += 1e-2 * np.diag(np.random.uniform(size=B.shape[0]))

        Rmax = np.dot(np.linalg.pinv(B), P)
        trace = np.dot(Rmax.conj(), np.dot(B, Rmax))
        Fid = np.dot(Rmax.conj(), np.dot(A, Rmax)) / trace
        if visualize:
            print(f"fid at trial {try_idx} step1: {Fid}")
        if try_idx >= init_trial:
            if Fid > 1.0 + 10*precision:
                print("numerically unstable")
                break
            if np.abs(Fid - past_fid) < precision:
                print("no more improvement")
                break
        past_fid = Fid
        past_trace = trace

        Utmp, stmp, Vh = np.linalg.svd(Rmax.reshape(trun_dim, -1), full_matrices=False)
        S = np.dot(Utmp, np.diag(stmp))

        ## step2
        R = oe.contract("ip,pq->qi",U,S).flatten()
        P = oe.contract("iIjJ,ij,QJ->QI",Gamma,sigma,Vh.conj()).flatten()
        A = oe.contract("a,b->ab",P,P.conj())
        B = oe.contract("iIjJ,qj,QJ->QIqi",Gamma,Vh,Vh.conj()).reshape(trun_dim*bond_dim, -1)

        if try_idx < init_trial:
            B += 1e-2 * np.diag(np.random.uniform(size=B.shape[0]))

        Rmax = np.dot(np.linalg.pinv(B), P)
        trace = np.dot(Rmax.conj(), np.dot(B, Rmax))
        Fid = np.dot(Rmax.conj(), np.dot(A, Rmax)) / trace
        if visualize:
            print(f"fid at trial {try_idx} step2: {Fid}")
        if try_idx >= init_trial:
            if Fid > 1.0 + 10*precision:
                print("numerically unstable")
                break
            if np.abs(Fid - past_fid) < precision:
                print("no more improvement")
                break
        past_fid = Fid
        past_trace = trace

        U, stmp, Vhtmp = np.linalg.svd(Rmax.reshape(trun_dim, -1).T, full_matrices=False)
        S = np.dot(np.diag(stmp), Vhtmp)

        try_idx += 1

    return U, S, Vh, past_fid, past_trace