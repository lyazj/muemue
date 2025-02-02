import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import loader
from get_p import get_p

# Optional maximum number of events to process.
NEVENT_MAX = None
COM_FRAME = True
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

def load(path, pid):
    # Outgoing beams.
    P3, P4 = loader.load(path, pid, NEVENT_MAX)
    m3 = np.mean(np.sqrt(P3[:,0]**2 - np.sum(P3[:,1:]**2, axis=1)))
    m4 = np.mean(np.sqrt(P4[:,0]**2 - np.sum(P4[:,1:]**2, axis=1)))

    # Target particle.
    P2 = np.repeat(get_P(m4, m4, 0, 0).reshape(1, 4), P4.shape[0], axis=0)

    # Incoming beam.
    P1 = P3 + P4 - P2

    if COM_FRAME:
        incoming_energy = np.mean(P1[:,0])
        gamma = (incoming_energy + m4) / np.sqrt(m3**2 + m4**2 + 2*m4*incoming_energy)
        beta = np.sqrt(1 - 1 / gamma**2)
        def lab_to_com(P):
            P0 = gamma * (P[:,0] - beta * P[:,3])
            P3 = gamma * (P[:,3] - beta * P[:,0])
            P[:,0] = P0
            P[:,3] = P3
            return P
        P1, P2, P3, P4 = map(lab_to_com, (P1, P2, P3, P4))

    return P1, P2, P3, P4

def load_pair(epem_path, emem_path):
    P3, P5, P7, P8 = load(epem_path, [-11, 11])
    P4, P6, P9, P10 = load(emem_path, [11, 11])
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
            p / (m + E) * np.cos(theta/2),
            p / (m + E) * np.exp(1j * phi) * np.sin(theta/2),
            np.cos(theta/2),
            np.exp(1j * phi) * np.sin(theta/2),
        ]).T
    elif H == -1:
        return np.sqrt((m/E + 1) / 2).reshape(-1, 1) * np.array([
            -p / (m + E) * np.sin(theta/2),
            p / (m + E) * np.exp(1j * phi) * np.cos(theta/2),
            np.sin(theta/2),
            -np.exp(1j * phi) * np.cos(theta/2),
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
    # Sum rho over final state spin states.
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
    #print(rho[:1])
    #print('trace:', rho[:1].trace(axis1=1, axis2=2))
    return rho

# Construct density matrix from 4-momenta.
def rho_P_Moller(P1, P2, P3, P4):
    u1s, u3s = map(lambda P: [u_PH(P, 1), u_PH(P, -1)], (P1, P3))
    u2s, u4s = map(lambda P: [u_PH(P, 1), u_PH(P, -1)], (P2, P4))
    #print(*u1s, *u2s, *u3s, *u4s, sep='\n')
    rho = np.zeros((P1.shape[0], 4, 4), dtype='complex')
    # Sum rho over final state spin states.
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
    #print(rho[:1])
    #print('trace:', rho[:1].trace(axis1=1, axis2=2))
    return rho

def rho_P_sec(P3, P4, P5, P6, P7, P8, P9, P10):
    #rho_Bhabha = rho_P_Bhabha(P3, P5, P7, P8).reshape(-1, 2, 2, 2, 2).trace(axis1=2, axis2=4)  # 3', 3
    #rho_Moller = rho_P_Moller(P4, P6, P9, P10).reshape(-1, 2, 2, 2, 2).trace(axis1=2, axis2=4)  # 4', 4
    rho_Bhabha = rho_P_Bhabha(P3, P5, P7, P8).reshape(-1, 2, 2, 2, 2)[:,:,1,:,1]  # 3', 3
    rho_Moller = rho_P_Moller(P4, P6, P9, P10).reshape(-1, 2, 2, 2, 2)[:,:,1,:,1]  # 4', 4
    rho = np.empty((P3.shape[0], 2, 2, 2, 2), dtype='complex')  # 3', 4', 3, 4
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    rho[:,i,j,k,l] = rho_Bhabha[:,i,k] * rho_Moller[:,j,l]
    rho = rho.reshape(-1, 4, 4)
    rho /= rho.trace(axis1=1, axis2=2).reshape(-1, 1, 1)
    print(rho[:3])
    print('trace:', rho[:1].trace(axis1=1, axis2=2))
    return rho

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

def rho_P(P3, P4, P5, P6, P7, P8, P9, P10):
    rho = rho_P_sec(P3, P4, P5, P6, P7, P8, P9, P10)
    theta_7 = np.arctan2(np.hypot(P7[:,1], P7[:,2]), P7[:,3])
    phi_7 = np.arctan2(P7[:,2], P7[:,1])
    theta_9 = np.arctan2(np.hypot(P9[:,1], P9[:,2]), P9[:,3])
    phi_9 = np.arctan2(P9[:,2], P9[:,1])

    lhs = np.empty((16), dtype='complex')
    rhs = np.empty((16, 16), dtype='complex')
    i = 0
    for l7 in [0, 1]:
        for m7 in range(-l7, l7 + 1):
            for l9 in [0, 1]:
                for m9 in range(-l9, l9 + 1):
                    weight = Y(l7, m7, theta_7, phi_7) * Y(l9, m9, theta_9, phi_9)
                    lhs[i] = np.mean(weight)
                    rhs[i] = np.mean(rho.reshape(-1, 16) * weight.reshape(-1, 1), axis=0)
                    print('rhs-%d:' % i, rhs[i], sep='\n')
                    i += 1
    print('lhs:', lhs, sep='\n')
    print('rhs-eigen:', np.linalg.eigvals(rhs), sep='\n')
    rho = (np.linalg.inv(rhs) @ lhs).reshape(4, 4)
    rho /= rho.trace()
    print('rho:', rho, sep='\n')
    print('rho-trace:', rho.trace())
    return rho

#def rotate_to(R, Ps):
#    theta = np.arctan2(np.hypot(R[1], R[2]), R[3])
#    phi = np.arctan2(R[2], R[1])
#    r = np.array([
#        [np.cos(phi), -np.sin(phi), 0],
#        [np.sin(phi),  np.cos(phi), 0],
#        [          0,            0, 1],
#    ]) @ np.array([
#        [ np.cos(theta), 0, np.sin(theta)],
#        [ 0,             1,             0],
#        [-np.sin(theta), 0, np.cos(theta)],
#    ])
#    for P in Ps:
#        P[:,1:4] = P[:,1:4] @ r.T

#theta_p_lab = 0.05
#P3_lab, P4_lab = get_p(theta_p_lab)
#print('P3:', P3_lab)
#print('P4:', P4_lab)
P3, P4, P5, P6, P7, P8, P9, P10 = load_pair(*[  # LL
    #'../epem_LU_example_0.290GeV_pT_0.00e+00GeV_eta_1.00e+00_0/Events/run_01/unweighted_events.root',
    #'../emem_LU_example_0.711GeV_pT_0.00e+00GeV_eta_1.00e+00_0/Events/run_01/unweighted_events.root',
    '../epem_LL_example_0.290GeV_pT_0.00e+00GeV_eta_1.00e+00_0/Events/run_01/unweighted_events.root',
    '../emem_LL_example_0.711GeV_pT_0.00e+00GeV_eta_1.00e+00_0/Events/run_01/unweighted_events.root',
])
#rotate_to(P3_lab, [P3, P5, P7, P8])
#rotate_to(P4_lab, [P4, P6, P9, P10])
print('P3:' , P3[0])
print('P4:' , P4[0])
print('P5:' , P5[0])
print('P6:' , P6[0])
print('P7:' , P7[0])
print('P8:' , P8[0])
print('P9:' , P9[0])
print('P10:', P10[0])

theta_7 = np.arctan2(np.hypot(P7[:,1], P7[:,2]), P7[:,3])
theta_9 = np.arctan2(np.hypot(P9[:,1], P9[:,2]), P9[:,3])
mask = np.logical_and(theta_7 > np.pi / 2, theta_9 > np.pi / 2)
print('Efficiency:', mask.mean())
for i in range(3, 11): exec(f'P{i} = P{i}[mask]')
rho = rho_P(P3, P4, P5, P6, P7, P8, P9, P10)
