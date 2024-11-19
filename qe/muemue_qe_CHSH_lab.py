import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.colors as mcolors
from muemue_qe_util import MuEQEUtil

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
offset = 0
for path, smear in [
    #('muemue_example_1.0GeV_pT_0.00e+00GeV_eta_2.85e+00.txt', False)
    ('muemue_example_10.0GeV_pT_0.00e+00GeV_eta_4.50e+00.txt', False),
    ('muemue_example_10.0GeV_pT_0.00e+00GeV_eta_4.50e+00.txt', True),
    #('muemue_example_160.0GeV_pT_0.00e+00GeV_eta_5.20e+00.txt', False),
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

    if smear:
        # Smear.
        util = MuEQEUtil(muon_energy)
        theta_mu_unc = 0.3e-3 * np.ones_like(theta_mu)
        theta_e_unc = 0.3e-3 * np.ones_like(theta_e)
        E_mu_unc = 0.5 * np.ones_like(E_mu)
        E_e_unc = 0.5 * np.ones_like(E_e)
        theta_mu += np.random.normal(0.0, theta_mu_unc, theta_mu.shape)
        theta_e += np.random.normal(0.0, theta_e_unc, theta_e.shape)
        E_mu += np.random.normal(0.0, E_mu_unc, E_mu.shape)
        E_e += np.random.normal(0.0, E_e_unc, E_e.shape)

        # Fit.
        #print('true:', theta_mu_org[:10], theta_e_org[:10], E_mu_org[:10], E_e_org[:10], sep='\n')
        #print('before:', theta_mu[:10], theta_e[:10], E_mu[:10], E_e[:10], sep='\n')
        obs = np.array([theta_mu, theta_e, E_mu, E_e]).T
        obs_unc = np.array([theta_mu_unc, theta_e_unc, E_mu_unc, E_e_unc]).T
        theta_mu_com, chi2 = util.fit(obs, obs_unc)
        theta_mu, theta_e, E_mu, E_e = util.get_observable(theta_mu_com).T
        #print('after:', theta_mu[:10], theta_e[:10], E_mu[:10], E_e[:10], sep='\n')

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
    rho_basis = np.array([
        np.eye(4),
        *(np.kron(sigma[i], np.eye(2)) for i in range(3)),
        *(np.kron(np.eye(2), sigma[i]) for i in range(3)),
        *(np.kron(sigma[i // 3], sigma[i % 3]) for i in range(9)),
    ]).reshape(16, 16).T / 4
    rho_to_corr = np.linalg.inv(rho_basis)[-9:].reshape(3, 3, 4, 4)
    
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
    
    def rho_P(P1, P2, P3, P4):
        u1s, u2s, u3s, u4s = map(lambda P: [u_PH(P, 1), u_PH(P, -1)], (P1, P2, P3, P4))
        #print(*u1s, *u2s, *u3s, *u4s, sep='\n')
        rho = np.zeros((P1.shape[0], 4, 4), dtype='complex')
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
                        #rho[:,i,j] += M[:,i] * M[:,j].conjugate() / 4
                        rho[:,i,j] += M[:,i] * M[:,j].conjugate()
        rho /= np.trace(rho, axis1=1, axis2=2).reshape(-1, 1, 1)
        print(rho[:1])
        return rho
    
    rho = rho_P(P1, P2, P3, P4)
    R = rho @ np.kron(sigma[1], sigma[1]) @ rho.conj() @ np.kron(sigma[1], sigma[1])
    eigenvalues = np.linalg.eigvals(R).real
    eigenvalues *= (np.abs(eigenvalues) >= 1e-10)
    eigenvalues = np.sort(np.sqrt(eigenvalues), axis=1)
    concurrence = eigenvalues[:,-1] - np.sum(eigenvalues[:,:-1], axis=1)
    concurrence = np.maximum(concurrence, 0)

    corr = np.sum(rho_to_corr.reshape(1, 3, 3, 16) * rho.reshape(-1, 1, 1, 16), axis=3)
    print(corr[0])
    eigenvalues = np.linalg.eigvals(corr.conjugate().transpose(0, 2, 1) @ corr).real
    eigenvalues *= (np.abs(eigenvalues) >= 1e-10)
    eigenvalues = np.sort(np.sqrt(eigenvalues), axis=1)
    CHSH = 2 * np.sqrt(np.sum(eigenvalues[:,-2:], axis=1))
    print('Efficiency:', np.mean(CHSH > 2))

    data = np.array([theta_mu, theta_e, CHSH]).T
    data = data[data[:,0].argsort()]
    colors = cmap(mcolors.Normalize(vmin=2, vmax=2.3)(data[:,2]))
    colors[data[:,2] <= 2] = [0.8, 0.8, 0.8, 1]
    plt.scatter(data[:,0], data[:,1] + offset, c=colors,
         #label=r'$E_\mu$ = %.0f GeV  $p_\mathrm{T} \geq$%.2e GeV  $\eta \geq$%.2e' % (muon_energy, lepton_pt, lepton_eta))
         label=r'$E_\mu$ = %.0f GeV  $\eta \geq$%.2f  (%s$\theta_e + %g$)' % (muon_energy, lepton_eta, 'smeared, ' if smear else '', offset))
    offset += 0.05

plt.xlabel(r'$\theta_\mu$ (lab frame)')
plt.ylabel(r'$\theta_e$')
plt.legend()
plt.grid()
cbar = plt.colorbar()
cbar.set_ticks(np.linspace(0, 1, 7))
cbar.set_ticklabels(map(lambda x: '%.2f' % x, np.linspace(2, 2.3, 7)))
plt.tight_layout()
plt.savefig(__file__.replace('.py', '.png'))
