import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# Optional maximum number of events to process.
NEVENT_MAX = None
COM_FRAME = False
STACK = True
BELL = True

# Construct 4-momentum from (E, m, theta, phi).
def get_P(E, m, theta, phi):
    p = np.sqrt(E*E - m*m)
    return np.array([
        E,
        p * np.sin(theta) * np.cos(phi),
        p * np.sin(theta) * np.sin(phi),
        p * np.cos(theta),
    ]).T

def load(path):
    # Incoming beam energy and generator cuts.
    incoming_energy, lepton_pt, lepton_eta = map(float,
        re.search(r'_([0-9.e+-]*)GeV_pT_([0-9.e+-]*)GeV_eta_([0-9.e+-]*)\.txt', path).groups())

    # Scattering particle masses.
    m_p = 0.511e-3
    m_e = 0.511e-3

    # Lorentz transformation.
    gamma = (incoming_energy + m_e) / np.sqrt(m_p**2 + m_e**2 + 2*m_e*incoming_energy)
    beta = np.sqrt(1 - 1 / gamma**2)
    def lab_to_com(P):
        P0 = gamma * (P[:,0] - beta * P[:,3])
        P3 = gamma * (P[:,3] - beta * P[:,0])
        P[:,0] = P0
        P[:,3] = P3
        return P

    # Scattering particle masses.
    m_1 = 0.511e-3
    m_2 = 0.511e-3

    # Read observables from the sample file.
    theta_1, theta_2, E_1, E_2 = np.array(
        open(path).read().strip().split(), dtype='float'
    ).reshape(-1, 4)[:(NEVENT_MAX if NEVENT_MAX else int(1e20))].T

    # Incoming beam.
    P1 = np.repeat(get_P(incoming_energy, m_1, 0, 0).reshape(1, 4), E_1.shape[0], axis=0)

    # Target particle.
    P2 = np.repeat(get_P(m_2, m_2, 0, 0).reshape(1, 4), E_2.shape[0], axis=0)

    # Outgoing beam.
    P3 = get_P(E_1, m_1, theta_1, 0)

    # Recoiled particle.
    P4 = get_P(E_2, m_2, theta_2, np.pi)

    if COM_FRAME:
        # Transform to the center of mass frame.
        P1, P2, P3, P4 = map(lab_to_com, (P1, P2, P3, P4))
        ## In the case of in a transformed frame, recompute observables.
        #theta_1 = np.arctan2(np.hypot(P3[:,1], P3[:,2]), P3[:,3])
        #theta_2 = np.arctan2(np.hypot(P4[:,1], P4[:,2]), P4[:,3])
        #E_1 = P3[:,0]
        #E_2 = P4[:,0]
    return P1, P2, P3, P4

def load_pair(epem_path, emem_path):
    P3, P5, P7, P8 = load(epem_path)
    P4, P6, P9, P10 = load(emem_path)
    return P3, P4, P5, P6, P7, P8, P9, P10


# Metric, Dirac gamma matrices, and Pauli matrices.
g = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])
gamma = [
    np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]),
    np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]]),
    np.array([[0, 0, 0, -1j], [0, 0, 1j, 0], [0, 1j, 0, 0], [-1j, 0, 0, 0]]),
    np.array([[0, 0, 1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, 0, 0]]),
]
sigma = [
    np.array([[0, 1], [1, 0]]),
    np.array([[0, -1j], [1j, 0]]),
    np.array([[1, 0], [0, -1]]),
]
sigma_kron = np.array([[
    np.kron(sigma[i], sigma[j]) for j in range(3)
] for i in range(3)])

# Compute inner product of two Lorentz vectors.
def lorentz_inner(P1, P2):
    assert P1.shape[1] == P2.shape[1] == 4
    E1, p1 = P1[:,0], P1[:,1:]
    E2, p2 = P2[:,0], P2[:,1:]
    return E1*E2 - np.sum(p1*p2, axis=1)

# Construct Dirac spinors of l-'s from 4-momenta and a unified helicity.
def u_PH(P, H):
    assert P.shape[1] == 4
    E = P[:,0]  # energy
    p3 = P[:,1:4]  # 3-momentum
    p = np.sqrt(np.sum(p3*p3, axis=1))  # 3-momentum modulo
    #theta = np.arccos(p3[:,2] / (p + (p == 0)))  # polar angle
    theta = np.arctan2(np.hypot(p3[:,0], p3[:,1]), p3[:,2])  # polar angle
    phi = np.arctan2(p3[:,1], p3[:,0])  # azimuthal angle
    m = np.sqrt(E*E - p*p)  # static mass
    if H == 1:
        return np.sqrt((m/E + 1) / 2).reshape(-1, 1) * np.array([
            np.cos(theta/2),
            np.exp(1j * phi) * np.sin(theta/2),
            p / (m + E) * np.cos(theta/2),
            p / (m + E) * np.exp(1j * phi) * np.sin(theta/2),
        ]).T
    elif H == -1:
        return np.sqrt((m/E + 1) / 2).reshape(-1, 1) * np.array([
            np.sin(theta/2),
            -np.exp(1j * phi) * np.cos(theta/2),
            -p / (m + E) * np.sin(theta/2),
            p / (m + E) * np.exp(1j * phi) * np.cos(theta/2),
        ]).T

# Construct Dirac spinors of l+'s from 4-momenta and a unified helicity.
def v_PH(P, H):
    assert P.shape[1] == 4
    E = P[:,0]  # energy
    p3 = P[:,1:4]  # 3-momentum
    p = np.sqrt(np.sum(p3*p3, axis=1))  # 3-momentum modulo
    theta = np.arctan2(np.hypot(p3[:,0], p3[:,1]), p3[:,2])  # polar angle
    phi = np.arctan2(p3[:,1], p3[:,0])  # azimuthal angle
    m = np.sqrt(E*E - p*p)  # static mass
    if H == 1:
        return np.sqrt((m/E + 1) / 2).reshape(-1, 1) * np.array([
            -p / (m + E) * np.sin(theta/2),
            p / (m + E) * np.exp(1j * phi) * np.cos(theta/2),
            np.sin(theta/2),
            -np.exp(1j * phi) * np.cos(theta/2),
        ]).T
    elif H == -1:
        return np.sqrt((m/E + 1) / 2).reshape(-1, 1) * np.array([
            p / (m + E) * np.cos(theta/2),
            p / (m + E) * np.exp(1j * phi) * np.sin(theta/2),
            np.cos(theta/2),
            np.exp(1j * phi) * np.sin(theta/2),
        ]).T

# Compute scattering amplitude matrix element from spinors and 4-momenta.
def M_uP_Bhabha(u1, u2, u3, u4, P1, P2, P3, P4):
    assert P1.shape[1] == P3.shape[1] == 4
    M_sc = sum(  # let e = 1
        np.sum(u1.conjugate() @ (gamma[0] @ gamma[i]) * u2, axis=1)
        * (g[i,i] / lorentz_inner(*((P1[:,0:4] + P2[:,0:4],)*2))) *
        np.sum(u4.conjugate() @ (gamma[0] @ gamma[i]) * u3, axis=1)
        for i in range(4)
    )
    M_tc = sum(  # let e = 1
        np.sum(u3.conjugate() @ (gamma[0] @ gamma[i]) * u1, axis=1)
        * (g[i,i] / lorentz_inner(*((P1[:,0:4] - P3[:,0:4],)*2))) *
        np.sum(u4.conjugate() @ (gamma[0] @ gamma[i]) * u2, axis=1)
        for i in range(4)
    )
    return M_sc + M_tc

# Compute scattering amplitude matrix element from spinors and 4-momenta.
def M_uP_Moller(u1, u2, u3, u4, P1, P2, P3, P4):
    assert P1.shape[1] == P3.shape[1] == 4
    M_uc = -sum(  # let e = 1
        np.sum(u3.conjugate() @ (gamma[0] @ gamma[i]) * u2, axis=1)
        * (g[i,i] / lorentz_inner(*((P2[:,0:4] - P3[:,0:4],)*2))) *
        np.sum(u4.conjugate() @ (gamma[0] @ gamma[i]) * u1, axis=1)
        for i in range(4)
    )
    M_tc = sum(  # let e = 1
        np.sum(u3.conjugate() @ (gamma[0] @ gamma[i]) * u1, axis=1)
        * (g[i,i] / lorentz_inner(*((P1[:,0:4] - P3[:,0:4],)*2))) *
        np.sum(u4.conjugate() @ (gamma[0] @ gamma[i]) * u2, axis=1)
        for i in range(4)
    )
    return M_uc + M_tc

# Construct density matrix from 4-momenta.
def rho_P_Bhabha(P1, P2, P3, P4):
    u1s, u3s = map(lambda P: [v_PH(P, 1), v_PH(P, -1)], (P1, P3))
    u2s, u4s = map(lambda P: [u_PH(P, 1), u_PH(P, -1)], (P2, P4))
    #print(*u1s, *u2s, *u3s, *u4s, sep='\n')
    rho = np.zeros((P1.shape[0], 4, 4), dtype='complex')
    # Sum rho over initial state spin states.
    for k in range(2):
        u3 = u3s[k]
        for l in range(2):
            u4 = u4s[l]
            # Compute scattering amplitude matrix given the current initial state.
            M = np.empty((P1.shape[0], 2, 2), dtype='complex')
            for i in range(2):
                u1 = u1s[i]
                for j in range(2):
                    u2 = u2s[j]
                    print(i, j, k, l)
                    M[:,i,j] = M_uP_Bhabha(u1, u2, u3, u4, P1, P2, P3, P4)
            # Add "Kronecker product of M and M*" from the current initial state to rho.
            M = M.reshape(-1, 4)
            for i in range(4):
                for j in range(4):
                    #rho[:,i,j] += M[:,i] * M[:,j].conjugate() / 4
                    rho[:,i,j] += M[:,i] * M[:,j].conjugate()
    # Normalize rho.
    rho /= np.trace(rho, axis1=1, axis2=2).reshape(-1, 1, 1)
    print(rho[:1])
    return rho

# Construct density matrix from 4-momenta.
def rho_P_Moller(P1, P2, P3, P4):
    u1s, u3s = map(lambda P: [u_PH(P, 1), u_PH(P, -1)], (P1, P3))
    u2s, u4s = map(lambda P: [u_PH(P, 1), u_PH(P, -1)], (P2, P4))
    #print(*u1s, *u2s, *u3s, *u4s, sep='\n')
    rho = np.zeros((P1.shape[0], 4, 4), dtype='complex')
    # Sum rho over initial state spin states.
    for k in range(2):
        u3 = u3s[k]
        for l in range(2):
            u4 = u4s[l]
            # Compute scattering amplitude matrix given the current initial state.
            M = np.empty((P1.shape[0], 2, 2), dtype='complex')
            for i in range(2):
                u1 = u1s[i]
                for j in range(2):
                    u2 = u2s[j]
                    print(i, j, k, l)
                    M[:,i,j] = M_uP_Moller(u1, u2, u3, u4, P1, P2, P3, P4)
            # Add "Kronecker product of M and M*" from the current initial state to rho.
            M = M.reshape(-1, 4)
            for i in range(4):
                for j in range(4):
                    #rho[:,i,j] += M[:,i] * M[:,j].conjugate() / 4
                    rho[:,i,j] += M[:,i] * M[:,j].conjugate()
    # Normalize rho.
    rho /= np.trace(rho, axis1=1, axis2=2).reshape(-1, 1, 1)
    print(rho[:1])
    return rho

def rho_P_sec(P3, P4, P5, P6, P7, P8, P9, P10):
    rho_Bhabha = rho_P_Bhabha(P3, P5, P7, P8).reshape(-1, 2, 2, 2, 2).trace(axis1=2, axis2=4)  # 3', 3
    rho_Moller = rho_P_Moller(P4, P6, P9, P10).reshape(-1, 2, 2, 2, 2).trace(axis1=2, axis2=4)  # 4', 4
    rho = np.empty((P3.shape[0], 2, 2, 2, 2), dtype='complex')  # 3', 4', 3, 4
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    rho[:,i,j,k,l] = rho_Bhabha[:,i,k] * rho_Moller[:,j,l]
    return rho.reshape(-1, 4, 4)

def Y(l, m, theta, phi):
    if l == 0:
        if m == 0:
            return np.sqrt(1 / (4 * np.pi)) * np.ones_like(theta)
    elif l == 1:
        if m == -1:
            return np.sqrt(3 / (8 * np.pi)) * np.sin(theta) * np.exp(-1j * phi)
        elif m == 0:
            return np.sqrt(3 / (4 * np.pi)) * np.cos(theta)
        elif m == 1:
            return -np.sqrt(3 / (8 * np.pi)) * np.sin(theta) * np.exp(1j * phi)
    raise ValueError('invalid l or m')

def LHS(l7, m7, l9, m9, P3, P4, P5, P6, P7, P8, P9, P10):
    theta_7 = np.arctan2(np.hypot(P7[:,1], P7[:,2]), P7[:,3])
    phi_7 = np.arctan2(P7[:,2], P7[:,1])
    theta_9 = np.arctan2(np.hypot(P9[:,1], P9[:,2]), P9[:,3])
    phi_9 = np.arctan2(P9[:,2], P9[:,1])
    return np.mean(Y(l7, m7, theta_7, phi_7) * Y(l9, m9, theta_9, phi_9))

def RHS(l7, m7, l9, m9, P3, P4, P5, P6, P7, P8, P9, P10):
    theta_7 = np.arctan2(np.hypot(P7[:,1], P7[:,2]), P7[:,3])
    phi_7 = np.arctan2(P7[:,2], P7[:,1])
    theta_9 = np.arctan2(np.hypot(P9[:,1], P9[:,2]), P9[:,3])
    phi_9 = np.arctan2(P9[:,2], P9[:,1])
    rho = rho_P_sec(P3, P4, P5, P6, P7, P8, P9, P10)
    return np.mean(rho * (Y(l7, m7, theta_7, phi_7) * Y(l9, m9, theta_9, phi_9)).reshape(-1, 1, 1), axis=0)

def rho_P(P3, P4, P5, P6, P7, P8, P9, P10):
    lhs = np.empty((16), dtype='complex')
    rhs = np.empty((16, 16), dtype='complex')
    i = 0
    for l7 in [0, 1]:
        for m7 in range(-l7, l7 + 1):
            for l9 in [0, 1]:
                for m9 in range(-l9, l9 + 1):
                    lhs[i] = LHS(l7, m7, l9, m9, P3, P4, P5, P6, P7, P8, P9, P10)
                    rhs[i] = RHS(l7, m7, l9, m9, P3, P4, P5, P6, P7, P8, P9, P10).reshape(16)
                    i += 1
    return (np.linalg.inv(rhs) @ lhs).reshape(4, 4)

P3, P4, P5, P6, P7, P8, P9, P10 = load_pair(*[  # LL
    './epem_example_LU_0.290GeV_pT_0.00e+00GeV_eta_1.00e+00.txt',
    './emem_example_LU_0.711GeV_pT_0.00e+00GeV_eta_1.00e+00.txt',
])
rho = rho_P(P3, P4, P5, P6, P7, P8, P9, P10) / np.square(2 / (4 * np.pi))
