import pprint
import numpy as np
import sympy as sp
from sympy.solvers.ode.systems import dsolve_system

def pick_mapping(sols: np.ndarray, var: sp.Symbol, dim: int) -> np.ndarray:
    """Pick mapping of a certain basis

    Args:
        sols (np.ndarray): list of time-dependent solutions of elements
        var (sp.Symbol): target variable
        dim (int): dimension of matrix

    Returns:
        np.ndarray: mapped density matrix
    """
    mat = np.ndarray(shape=(dim, dim), dtype=float)
    for y in range(dim):
        for x in range(dim):
            # Expand is required for valid output. Is this bug?
            val = sp.expand(sols[y, x].rhs)
            val = val.coeff(var)
            val = complex(val)
            img_part = np.imag(val)
            assert(img_part < 1e-14)
            mat[y, x] = abs(val)
    return mat


def get_damping_process(gm: float, gp: float, tau: float, dim: int) -> np.ndarray:
    """Solve differential equation and get CPTP-map

    Args:
        gm (float): gamma minus
        gp (float): gamma plus
        tau (float): elpased time
        dim (int): dimneion

    Returns:
        np.ndarray: (x,y)-th element is a density matrix which is mapped from E(|x><y|)
    """
    # create matrix elements and initial values
    sp.init_printing()
    t = sp.var('t')

    # gm = 0
    # gp = sp.symbols("gp")

    funcs = np.ndarray(shape=(dim, dim), dtype=object)
    rho = np.ndarray(shape=(dim, dim), dtype=object)
    vars = np.ndarray(shape=(dim, dim), dtype=object)
    for y in range(dim):
        for x in range(dim):
            func = sp.symbols(f"rho{y}{x}", cls=sp.Function)
            var = sp.symbols(f"rho{y}{x}_")
            funcs[y, x] = func
            rho[y, x] = func(t)
            vars[y, x] = var

    # annihelation op
    a = np.zeros(shape=(dim, dim), dtype=float)
    for x in range(1, dim):
        a[x-1, x] = np.sqrt(x)

    # create differential equations
    diffs = gm * (a@rho@a.T - 0.5 * a.T@a@rho - 0.5 * rho@a.T@a)
    diffs += gp * (a.T@rho@a - 0.5 * a@a.T@rho - 0.5 * rho@a@a.T)
    eqs = np.ndarray(shape=(dim, dim), dtype=object)
    for y in range(dim):
        for x in range(dim):
            eqs[y, x] = sp.Eq(rho[y, x].diff(t), diffs[y, x])

    # create system
    sols = np.ndarray(shape=(dim, dim), dtype=object)
    for dif in range(-dim+1, dim):
        system = []
        ics = {}
        pos = []
        for y in range(dim):
            x = y + dif
            if not (0 <= x and x < dim):
                continue
            system.append(eqs[y, x])
            ics[funcs[y, x](0)] = vars[y, x]
            pos.append((x, y))
        raw_sols = dsolve_system(system, ics=ics)[0]
        for ind in range(len(pos)):
            x, y = pos[ind]
            # print(x, y, raw_sols[ind])
            sols[y, x] = raw_sols[ind].subs(t, tau)

    # create process
    result = np.ndarray(shape=(dim, dim), dtype=object)
    for y in range(dim):
        for x in range(dim):
            var = vars[y, x]
            mat = pick_mapping(sols, var, dim)
            result[y, x] = mat
    return result


def convert_process_to_kraus(map: np.ndarray, dim: int, eps: float) -> list:
    """Get Kraus maptrix from mpa

    Args:
        map (np.ndarray): process map
        dim (int): dimension
        eps (float): allowed eps

    Raises:
        ValueError: Map is not CP

    Returns:
        list: list of Kraus matrices
    """
    # create choi
    # coef * |y><x| x map(|y><x|)
    choi = np.zeros(shape=(dim**2, dim**2), dtype=complex)
    coef = 1/dim
    for y in range(dim):
        for x in range(dim):
            mat1 = np.zeros(shape=(dim, dim), dtype=complex)
            mat1[y, x] = 1
            mat2 = map[y, x]
            choi += coef * np.kron(mat1, mat2)

    # diagonalize Choi matrix
    kraus_list = []
    TP_check = []
    eval, evec = np.linalg.eigh(choi)
    for i in range(dim**2):
        if abs(eval[i]) < eps:
            continue
        if eval[i] < eps:
            raise ValueError(f"Choi matrix has negative eigenvalue {eval[i]}. Map is not CP.")
        vec = np.sqrt(dim) * np.sqrt(np.real(eval[i])) * evec[:, i]
        kraus = np.reshape(vec, (dim, dim)).T

        # fix phase of kraus
        pos = np.unravel_index(np.argmax(np.abs(kraus)), kraus.shape)
        max_val = kraus[pos]
        phase = max_val / np.abs(max_val)
        kraus *= phase

        kraus_list.append(kraus)
        TP_check.append(kraus.T.conj() @ kraus)

    # Check TP
    tpness = np.sum(TP_check, axis=0)
    assert(np.allclose(tpness, np.eye(dim)))
    return kraus_list


def get_kraus_list(gamma: float, kbT_over_hw: float, tau: float, dim: int, eps: float) -> list:
    """Get list of Kraus matrices of damping process

    kbT_over_hw is the value of (kB T / E01)

    Args:
        gamma (float): damping rate
        kbT_over_hw (float): temparature relative to energy gap
        tau (float): elapsed time
        dim (int): dimension
        eps (float): allowed eps

    Returns:
        list: list of Kraus matrices
    """

    # calculate master eq from temperature
    if kbT_over_hw < 1e-16:
        N = 0
    else:
        N = 1 / (np.exp(1/kbT_over_hw) - 1)
    gm = gamma * (N+1)
    gp = gamma * N
    #print(gm, gp)

    process = get_damping_process(gm, gp, tau, dim)
    kraus_list = convert_process_to_kraus(process, dim, eps)
    return kraus_list


if __name__ == "__main__":
    dim = 3
    eps = 1e-13
    gamma = 1 # in [MHz]
    kbT_over_hw = 0.5 # kb T / hbar w [a.u.]
    tau = 1.0 # in [us]
    kraus_list = get_kraus_list(gamma, kbT_over_hw, tau, dim, eps)
    np.set_printoptions(precision=3, suppress=True)
    pprint.pprint(kraus_list)
