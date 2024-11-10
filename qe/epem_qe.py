import re
import numpy as np

def get_P(E, m, theta, phi):
    p = np.sqrt(E*E - m*m)
    return np.array([
        E,
        p * np.sin(theta) * np.cos(phi),
        p * np.sin(theta) * np.sin(phi),
        p * np.cos(theta),
    ]).T

path = './epem_example_1.0GeV_pT_1e-02GeV_E_1e-02GeV.txt'
#path = './epem_example_1.0GeV_pT_1e-03GeV_E_1e-02GeV.txt'
#path = './epem_example_1.0GeV_pT_1e-04GeV_E_1e-02GeV.txt'
#path = './epem_example_10.0GeV_pT_1e-02GeV_E_1e-02GeV.txt'
#path = './epem_example_10.0GeV_pT_1e-03GeV_E_1e-02GeV.txt'
#path = './epem_example_10.0GeV_pT_1e-04GeV_E_1e-02GeV.txt'
e1_energy, lepton_pt, lepton_energy = map(float,
    re.search(r'_([0-9.e+-]*)GeV_pT_([0-9.e+-]*)GeV_E_([0-9.e+-]*)GeV', path).groups())
m_e = 0.511e-3
theta_e1, theta_e2, E_e1, E_e2 = np.array(
    open(path).read().strip().split(), dtype='float'
).reshape(-1, 4).T
P1 = np.repeat(get_P(e1_energy, m_e, 0, 0).reshape(1, 4), E_e1.shape[0], axis=0)
P2 = np.repeat(get_P(m_e, m_e, 0, 0).reshape(1, 4), E_e2.shape[0], axis=0)
P3 = get_P(E_e1, m_e, theta_e1, 0)
E_e2 = P1[:,0] + P2[:,0] - P3[:,0]
P4 = get_P(E_e2, m_e, theta_e2, np.pi)
print('theta1 =', theta_e1[0])
print('theta2 =', theta_e2[0])
print('E3 =', P3[0,0])

#theta_e1, P1, P2, P3, P4 = map(lambda x: x[:1], (theta_e1, P1, P2, P3, P4))  # [DEBUG]

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

def lorentz_inner(P1, P2):
    assert P1.shape[1] == P2.shape[1] == 4
    E1, p1 = P1[:,0], P1[:,1:]
    E2, p2 = P2[:,0], P2[:,1:]
    return E1*E2 - np.sum(p1*p2, axis=-1)

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

def M_uP(u1, u2, u3, u4, P1, P2, P3, P4):
    assert P1.shape[1] == P3.shape[1] == 4
    Q = P1[:,0:4] - P3[:,0:4]  # 4-momentum transfer
    return sum(  # let e = 1
        np.sum(u3.conjugate() @ (gamma[0] @ gamma[i]) * u1, axis=1)
        #* (g[i,i] / lorentz_inner(Q, Q)) *
        * g[i,i] *
        np.sum(u4.conjugate() @ (gamma[0] @ gamma[i]) * u2, axis=1)
        for i in range(4)
    )

def rho2_P(P1, P2, P3, P4):
    M = np.zeros((P1.shape[0], 2, 2), dtype='complex')
    u1s, u3s = map(lambda P: [v_PH(P, 1), v_PH(P, -1)], (P1, P3))
    u2s, u4s = map(lambda P: [u_PH(P, 1), u_PH(P, -1)], (P2, P4))
    #print(*u1s, *u2s, *u3s, *u4s, sep='\n')
    for i in range(2):
        u3 = u3s[i]
        for j in range(2):
            u4 = u4s[j]
            for k in range(2):
                u1 = u1s[k]
                for l in range(2):
                    u2 = u2s[l]
                    print(i, j, k, l)
                    M[:,i,j] += M_uP(u1, u2, u3, u4, P1, P2, P3, P4)
    M /= 4
    M = M.reshape(-1, 4)
    rho2 = np.empty((P1.shape[0], 4, 4), dtype='complex')
    for i in range(4):
        for j in range(4):
            rho2[:,i,j] = M[:,i] * M[:,j].conjugate()
    rho2 /= np.trace(rho2, axis1=1, axis2=2).reshape(-1, 1, 1)
    print(rho2[:1])
    return rho2

rho2 = rho2_P(P1, P2, P3, P4)
R = rho2 @ np.kron(sigma[1], sigma[1]) @ rho2.conj() @ np.kron(sigma[1], sigma[1])
eigenvalues = np.linalg.eigvals(R).real
eigenvalues *= (np.abs(eigenvalues) >= 1e-10)
eigenvalues = np.sort(np.sqrt(eigenvalues), axis=1)
concurrence = eigenvalues[:,-1] - np.sum(eigenvalues[:,:-1], axis=1)

import matplotlib.pyplot as plt
plt.figure(dpi=300)
data = np.sort(np.array([theta_e1, concurrence]).T, axis=0)
plt.xlabel(r'$\theta_{e_1}$')
plt.ylabel('Concurrence')
plt.plot(data[:,0], data[:,1], '.-',
     label=r'$E_{e_1}$ = %.0f GeV  $p_\mathrm{T} \geq$%.0e GeV  $E \geq$%.0e GeV' % (e1_energy, lepton_pt, lepton_energy))
plt.legend(loc='lower right')
plt.grid()
plt.tight_layout()
plt.savefig(path.replace('.txt', '.png'))
