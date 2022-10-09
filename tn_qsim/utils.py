import opt_einsum as oe
import tensornetwork as tn
import quimb.tensor as qtn
import numpy as np
import functools
import jax
import tn_qsim.optimizer as opt

def from_nodes_to_str(node_list, output_edge_order, offset=8):
    input_sets = [set(node.edges) for node in node_list]
    output_set = set()
    for edge in tn.get_all_edges(node_list):
        if edge.is_dangling() or not set(edge.get_nodes()) <= set(node_list):
            output_set.add(edge)
    size_dict = {edge: edge.dimension for edge in tn.get_all_edges(node_list)}

    edge_alpha = dict()
    edge_alpha_dims = dict()
    alpha_offset = offset
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

    init_fluct = 10
    init_trial = 10

    fluct_table = [-2 for _ in range(10)]

    while (try_idx < trials):
        ## step1
        R = oe.contract("pq,qj->pj",S,Vh).flatten()
        P = oe.contract("iIjJ,ij,IP->PJ",Gamma,sigma,U.conj()).flatten()
        A = oe.contract("a,b->ab",P,P.conj())
        B = oe.contract("iIjJ,ip,IP->PJpj",Gamma,U,U.conj()).reshape(trun_dim*bond_dim, -1)

        if try_idx < init_fluct:
            B += 10**fluct_table[try_idx] * np.diag(np.random.uniform(size=B.shape[0]))

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

        if try_idx < init_fluct:
            B += 10**fluct_table[try_idx] * np.diag(np.random.uniform(size=B.shape[0]))

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

def gen_func_from_tree(tree, backend="jax"):
    """generate renormalization function

    Args:
        tree (ctg.ContractionTree) : the contraction tree for renormalization
        gpunum (int) : the number of GPU we use
    Returns:
        func : get sliced arrays and return renormalized hamiltonian, args (H, U1, U2, ..., U1.conj(), U2.conj(), ...)
    """

    def func(arrays):
        contract_core = functools.partial(tree.contract_core, backend=backend)
        slices = []
        for i in range(tree.nslices):
            slices.append(contract_core(tree.slice_arrays(arrays, i)))

        x = tree.gather_slices(slices)
        return x
    
    return func
    
def execute_optimal_truncation(Gamma, sigma, min_truncate_dim, max_truncate_dim, truncate_buff, threshold, trials, visualize=False):
    U, S, Vh, Fid, trace = None, None, None, 1.0, 1.0
    truncate_dim = None
    if threshold is not None:
        for cur_truncate_dim in range(min_truncate_dim, max_truncate_dim+1, truncate_buff):
            if cur_truncate_dim == Gamma.shape[0]:
                print("no truncation done")
                U = None
                break
            elif Gamma.shape[0] <= cur_truncate_dim:         
                print("truncate dim already satistfied")
                U = None
                break
            for sd in range(10):
                U, S, Vh, Fid, trace = calc_optimal_truncation(Gamma, sigma, cur_truncate_dim, precision=1-threshold, trials=trials, visualize=visualize)
                truncate_dim = cur_truncate_dim
                print(f"Fid {Fid} threshold {threshold}")
                if Fid > threshold:
                    break
                else:
                    # try GD
                    tmp = np.dot(U, np.dot(S, Vh))
                    U, s, Vh = np.linalg.svd(tmp, full_matrices=False)
                    U = U[:,:cur_truncate_dim]
                    s = s[:cur_truncate_dim]
                    Vh = Vh[:cur_truncate_dim]
                    L = jax.numpy.array(np.dot(U, np.diag(np.sqrt(s))))
                    R = jax.numpy.array(np.dot(np.diag(np.sqrt(s)), Vh))

                    def loss(params):
                        l1 = oe.contract("iIjJ,ik,kj,IJ",Gamma,params[0],params[1],sigma.conj(), backend="jax")
                        l2 = oe.contract("iIjJ,ij,IK,KJ",Gamma,sigma,params[0].conj(),params[1].conj(), backend="jax")
                        l3 = oe.contract("iIjJ,ik,kj,IK,KJ",Gamma,params[0],params[1],params[0].conj(),params[1].conj(), backend="jax")
                        return jax.numpy.abs(1-(l1 * l2 / l3).real)

                    params = [L, R]
                    res = loss(params)
                    print("initial loss:", res)
                    value_and_grad_func = jax.value_and_grad(loss)

                    # lr 対応表
                    lr_table = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.01, 0.01, 0.01, 0.001, 0.001, 0.0001, 0.0001, 0.0001]
                    lr_table = [10.0**((-i-16)//8) for i in range(20)]
                    print(lr_table)
                    flr = int(np.floor(np.log10(res)))
                    optimizer = opt.Adam(lr=lr_table[-flr])
                    print("flr:", -flr, "lr:", lr_table[-flr])

                    past_val = 0.0
                    for ep in range(1000):
                        val, grad = value_and_grad_func(params)
                        for idx in range(len(grad)):
                            grad[idx] = grad[idx].conj()
                        optimizer.update(params, grad)
                        print("epoch:", ep, "loss:", val)
                        if val < (1-threshold):
                            break
                        if np.abs(past_val - val) < 1e-17:
                            break
                        past_val = val
                        if ep % 100 == 99:
                            flr = int(np.floor(np.log10(val)))
                            optimizer.lr = lr_table[-flr]
                            print("lr:", lr_table[-flr])

                    Fid = 1-val
                    trace = oe.contract("iIjJ,ik,kj,IK,KJ",Gamma,params[0],params[1],params[0].conj(),params[1].conj(), backend="jax").item()
                    U, S, Vh = np.array(params[0]), np.eye(cur_truncate_dim), np.array(params[1])
                    print(Fid, trace)
                    if Fid > threshold:
                        break   

            if Fid > threshold:
                break

        return U, S, Vh, Fid, trace, truncate_dim
