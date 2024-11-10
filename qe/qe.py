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

path = './muemue_example_160.0GeV_pT_1e-01GeV_E_1e+00GeV.txt'
#path = './muemue_example_10.0GeV_pT_3e-02GeV_E_1e+00GeV.txt'
#path = './muemue_example_1.0GeV_pT_4e-03GeV_E_5e-02GeV.txt'
muon_energy, lepton_pt, lepton_energy = map(float,
    re.search(r'_([0-9.e+-]*)GeV_pT_([0-9.e+-]*)GeV_E_([0-9.e+-]*)GeV', path).groups())
m_e = 0.511e-3
m_mu = 106e-3
theta_mu, theta_e, E_mu, E_e = np.array(
    open(path).read().strip().split(), dtype='float'
).reshape(-1, 4).T
P1 = np.repeat(get_P(muon_energy, m_mu, 0, 0).reshape(1, 4), E_mu.shape[0], axis=0)
P2 = np.repeat(get_P(m_e, m_e, 0, 0).reshape(1, 4), E_e.shape[0], axis=0)
P3 = get_P(E_mu, m_mu, theta_mu, 0)
E_e = P1[:,0] + P2[:,0] - P3[:,0]
P4 = get_P(E_e, m_e, theta_e, np.pi)

theta_mu, P1, P2, P3, P4 = map(lambda x: x[:1], (theta_mu, P1, P2, P3, P4))  # [DEBUG]

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
    theta = np.arctan2(np.hypot(p3[:,0], p3[:,1]), p3[:,2])  # polar angle
    phi = np.arctan2(p3[:,1], p3[:,0])  # azimuthal angle
    m = np.sqrt(E*E - p*p)  # static mass
    if H == 1:
        return np.sqrt((E/m + 1) / 2).reshape(-1, 1) * np.array([
            np.cos(theta/2),
            np.exp(1j * phi) * np.sin(theta/2),
            p / (m + E) * np.cos(theta/2),
            p / (m + E) * np.exp(1j * phi) * np.sin(theta/2),
        ]).T
    elif H == -1:
        return np.sqrt((E/m + 1) / 2).reshape(-1, 1) * np.array([
            np.sin(theta/2),
            -np.exp(1j * phi) * np.cos(theta/2),
            -p / (m + E) * np.sin(theta/2),
            p / (m + E) * np.exp(1j * phi) * np.cos(theta/2),
        ]).T

def M_uP(u1, u2, u3, u4, P1, P2, P3, P4):
    assert P1.shape[1] == P3.shape[1] == 4
    Q = P1[:,0:4] - P3[:,0:4]  # 4-momentum transfer
    print(np.sum(u3.conjugate() @ (gamma[0] @ gamma[0]) * u1, axis=1))
    print(np.sum(u3.conjugate() @ (gamma[1] @ gamma[0]) * u1, axis=1))
    print(np.sum(u3.conjugate() @ (gamma[2] @ gamma[0]) * u1, axis=1))
    print(np.sum(u3.conjugate() @ (gamma[3] @ gamma[0]) * u1, axis=1))
    exit()
    return sum(  # let e = 1
        np.sum(u3.conjugate() @ (gamma[0] @ gamma[i]) * u1, axis=1)
        #* (g[i,i] / lorentz_inner(Q, Q)) *
        * g[i,i] *
        np.sum(u4.conjugate() @ (gamma[0] @ gamma[i]) * u2, axis=1)
        for i in range(4)
    )

def rho2_P(P1, P2, P3, P4):
    M = np.zeros((P1.shape[0], 2, 2), dtype='complex')
    u1s, u2s, u3s, u4s = map(lambda P: [u_PH(P, 1), u_PH(P, -1)], (P1, P2, P3, P4))
    print(*u1s, *u2s, *u3s, *u4s, sep='\n')
    for i in range(2):
        u3 = u3s[i]
        for j in range(2):
            u4 = u4s[j]
            for k in range(2):
                u1 = u1s[k]
                for l in range(2):
                    u2 = u2s[l]
                    print(i, j, k, l)
                    print(M_uP(u1, u2, u3, u4, P1, P2, P3, P4))
                    M[:,i,j] += M_uP(u1, u2, u3, u4, P1, P2, P3, P4)
    M /= 4
    M = M.reshape(-1, 4)
    rho2 = np.empty((P1.shape[0], 4, 4), dtype='complex')
    for i in range(4):
        for j in range(4):
            rho2[:,i,j] = (M[:,i].conjugate() * M[:,j])
    return rho2 / np.trace(rho2, axis1=1, axis2=2).reshape(-1, 1, 1)

rho2 = rho2_P(P1, P2, P3, P4)
R = rho2 @ np.kron(sigma[1], sigma[1]) @ rho2.conj() @ np.kron(sigma[1], sigma[1])
eigenvalues = np.linalg.eigvals(R).real
eigenvalues *= (np.abs(eigenvalues) >= 1e-10)
eigenvalues = np.sort(np.sqrt(eigenvalues), axis=1)
concurrence = eigenvalues[:,-1] - np.sum(eigenvalues[:,:-1], axis=1)

import matplotlib.pyplot as plt
plt.figure(dpi=300)
data = np.sort(np.array([theta_mu, concurrence]).T, axis=0)
plt.xlabel(r'$\theta_\mu$')
plt.ylabel('Concurrence')
plt.plot(data[:,0], data[:,1], '.-',
     label=r'$E_\mu$ = %.0f GeV  $p_\mathrm{T} \geq$%.0e GeV  $E \geq$%.0e GeV' % (muon_energy, lepton_pt, lepton_energy))
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(path.replace('.txt', '.png'))