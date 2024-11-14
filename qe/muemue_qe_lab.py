import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import colors

NEVENT_MAX = None

def get_P(E, m, theta, phi):
    p = np.sqrt(E*E - m*m)
    return np.array([
        E,
        p * np.sin(theta) * np.cos(phi),
        p * np.sin(theta) * np.sin(phi),
        p * np.cos(theta),
    ]).T

plt.figure(dpi=300)
cmap = plt.get_cmap('viridis')
cmap.set_bad('lightgray', 1.0)
for path in [
    #'muemue_example_1.0GeV_pT_0.00e+00GeV_eta_2.85e+00.txt',
    #'muemue_example_10.0GeV_pT_0.00e+00GeV_eta_4.50e+00.txt',
    'muemue_example_160.0GeV_pT_0.00e+00GeV_eta_5.20e+00.txt',
]:
    muon_energy, lepton_pt, lepton_eta = map(float,
        re.search(r'_([0-9.e+-]*)GeV_pT_([0-9.e+-]*)GeV_eta_([0-9.e+-]*)\.txt', path).groups())
    m_e = 0.511e-3
    m_mu = 106e-3
    gamma = (muon_energy + m_e) / np.sqrt(m_e**2 + m_mu**2 + 2*m_e*muon_energy)
    beta = np.sqrt(1 - 1 / gamma**2)
    def lab_to_com(P):
        P0 = gamma * (P[:,0] - beta * P[:,3])
        P3 = gamma * (P[:,3] - beta * P[:,0])
        P[:,0] = P0
        P[:,3] = P3
        return P
    theta_mu, theta_e, E_mu, E_e = np.array(
        open(path).read().strip().split(), dtype='float'
    ).reshape(-1, 4)[:(NEVENT_MAX if NEVENT_MAX else int(1e20))].T
    P1 = np.repeat(get_P(muon_energy, m_mu, 0, 0).reshape(1, 4), E_mu.shape[0], axis=0)
    P2 = np.repeat(get_P(m_e, m_e, 0, 0).reshape(1, 4), E_e.shape[0], axis=0)
    P3 = get_P(E_mu, m_mu, theta_mu, 0)
    P4 = get_P(E_e, m_e, theta_e, np.pi)
    #P1, P2, P3, P4 = map(lab_to_com, (P1, P2, P3, P4))
    theta_mu = np.arctan2(np.hypot(P3[:,1], P3[:,2]), P3[:,3])
    theta_e = np.arctan2(np.hypot(P4[:,1], P4[:,2]), P4[:,3])
    E_mu = P3[:,0]
    E_e = P4[:,0]
    print('theta1 =', theta_mu[0])
    print('theta2 =', theta_e[0])
    print('E3 =', P3[0,0])
    print('P1 = ', P1[0])
    print('P2 = ', P2[0])
    print('P3 = ', P3[0])
    print('P4 = ', P4[0])
    
    #theta_mu, P1, P2, P3, P4 = map(lambda x: x[:1], (theta_mu, P1, P2, P3, P4))  # [DEBUG]
    
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
        u1s, u2s, u3s, u4s = map(lambda P: [u_PH(P, 1), u_PH(P, -1)], (P1, P2, P3, P4))
        #print(*u1s, *u2s, *u3s, *u4s, sep='\n')
        rho2 = np.zeros((P1.shape[0], 4, 4), dtype='complex')
        for k in range(2):
            u1 = u1s[k]
            for l in range(2):
                u2 = u2s[l]
                M = np.empty((P1.shape[0], 2, 2), dtype='complex')
                for i in range(2):
                    u3 = u3s[i]
                    for j in range(2):
                        u4 = u4s[j]
                        print(i, j, k, l)
                        M[:,i,j] = M_uP(u1, u2, u3, u4, P1, P2, P3, P4)
                M = M.reshape(-1, 4)
                for i in range(4):
                    for j in range(4):
                        #rho2[:,i,j] += M[:,i] * M[:,j].conjugate() / 4
                        rho2[:,i,j] += M[:,i] * M[:,j].conjugate()
        rho2 /= np.trace(rho2, axis1=1, axis2=2).reshape(-1, 1, 1)
        print(rho2[:1])
        return rho2
    
    rho2 = rho2_P(P1, P2, P3, P4)
    R = rho2 @ np.kron(sigma[1], sigma[1]) @ rho2.conj() @ np.kron(sigma[1], sigma[1])
    eigenvalues = np.linalg.eigvals(R).real
    eigenvalues *= (np.abs(eigenvalues) >= 1e-10)
    eigenvalues = np.sort(np.sqrt(eigenvalues), axis=1)
    concurrence = eigenvalues[:,-1] - np.sum(eigenvalues[:,:-1], axis=1)
    concurrence = np.maximum(concurrence, 0)
    
    data = np.array([theta_mu, theta_e, concurrence]).T
    data = data[data[:,0].argsort()]
    plt.scatter(data[:,0], data[:,1], c=data[:,2], cmap=cmap, norm=colors.LogNorm(vmin=1e-3, vmax=1),
         #label=r'$E_\mu$ = %.0f GeV  $p_\mathrm{T} \geq$%.2e GeV  $\eta \geq$%.2e' % (muon_energy, lepton_pt, lepton_eta))
         label=r'$E_\mu$ = %.0f GeV  $\eta \geq$%.2f' % (muon_energy, lepton_eta))

    plt.xlabel(r'$\theta_\mu$ (lab frame)')
    plt.ylabel(r'$\theta_e$')
    plt.legend()
    plt.grid()
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path.replace('.txt', '.png'))
    plt.clf()
