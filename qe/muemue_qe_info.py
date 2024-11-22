import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.metrics import roc_curve, roc_auc_score
from muemue_qe_util import MuEQEUtil

NEVENT_MAX = None

cmap = plt.get_cmap('viridis')
cmap.set_bad('lightgray', 1.0)

def get_concurrence(P1, P2, P3, P4):
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
    return concurrence

plt.figure(dpi=300)
for path in [
    #'muemue_example_1.0GeV_pT_0.00e+00GeV_eta_2.85e+00.txt',
    'muemue_example_10.0GeV_pT_0.00e+00GeV_eta_4.50e+00.txt',
    #'muemue_example_160.0GeV_pT_0.00e+00GeV_eta_5.20e+00.txt',
]:
    muon_energy, lepton_pt, lepton_eta = map(float,
        re.search(r'_([0-9.e+-]*)GeV_pT_([0-9.e+-]*)GeV_eta_([0-9.e+-]*)\.txt', path).groups())
    util = MuEQEUtil(muon_energy)
    m_mu, m_e = util.m_mu, util.m_e
    theta_mu, theta_e, E_mu, E_e = np.array(
        open(path).read().strip().split(), dtype='float'
    ).reshape(-1, 4)[:(NEVENT_MAX if NEVENT_MAX else int(1e20))].T
    P1 = np.repeat(util.get_P(muon_energy, m_mu, 0, 0).reshape(1, 4), E_mu.shape[0], axis=0)
    P2 = np.repeat(util.get_P(m_e, m_e, 0, 0).reshape(1, 4), E_e.shape[0], axis=0)
    P3 = util.get_P(E_mu, m_mu, theta_mu, 0)
    P4 = util.get_P(E_e, m_e, theta_e, np.pi)
    #tuple(map(util.lab_to_com, (P1, P2, P3, P4)))
    theta_mu = np.arctan2(np.hypot(P3[:,1], P3[:,2]), P3[:,3])
    theta_e = np.arctan2(np.hypot(P4[:,1], P4[:,2]), P4[:,3])
    E_mu = P3[:,0]
    E_e = P4[:,0]
    concurrence = get_concurrence(P1, P2, P3, P4)

    print('Muon energy:', muon_energy)
    print('Max concurrence:', np.max(concurrence))
    print('Efficiency:', np.mean(concurrence > 0))
    print('Efficiency (theta > 0.5 mrad):', np.mean(concurrence > 0.5e-3))
    print('Max theta_mu:', np.max(theta_mu[concurrence > 0]))
    print('Max theta_e:', np.max(theta_e[concurrence > 0]))
    print('Min E_mu:', np.min(E_mu[concurrence > 0]))
    print('Min E_e:', np.min(E_e[concurrence > 0]))

    label = concurrence > 0
    theta_mu_org = theta_mu.copy()
    theta_e_org = theta_e.copy()
    E_mu_org = E_mu.copy()
    E_e_org = E_e.copy()

    # Smear.
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
    from scipy import stats
    n, bins, _ = plt.hist(chi2, 20, density=True, label='smeared and fit')
    dof = stats.chi2.fit(chi2)[0]
    x = np.linspace(bins[0], bins[-1], 1001)
    plt.plot(x, stats.chi2(dof).pdf(x), label=r'$\chi^2(%.2f)$' % dof)
    plt.xlabel('Summed square relevant error')
    plt.ylabel('Density')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(path.replace('.txt', '_chi2.png'))
    plt.clf()

    P3 = util.get_P(E_mu, m_mu, theta_mu, 0)
    P4 = util.get_P(E_e, m_e, theta_e, np.pi)
    #tuple(map(util.lab_to_com, (P1, P2, P3, P4)))
    theta_mu = np.arctan2(np.hypot(P3[:,1], P3[:,2]), P3[:,3])
    theta_e = np.arctan2(np.hypot(P4[:,1], P4[:,2]), P4[:,3])
    E_mu = P3[:,0]
    E_e = P4[:,0]
    concurrence = get_concurrence(P1, P2, P3, P4)
    fpr, tpr, thr = roc_curve(label, concurrence)
    auc = roc_auc_score(label, concurrence)
    i_max = np.argmax(fpr > 1e-3) - 1
    plt.scatter(fpr, tpr, c=thr, cmap=cmap, norm=colors.LogNorm(vmin=np.sort(thr)[1], vmax=np.sort(thr)[-2]),
         label=r'$E_\mu$ = %.0f GeV  $\eta \geq$%.2f  (AUC = %.3f, TPR = %.3f)' % (
             muon_energy, lepton_eta, auc, tpr[i_max]))
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend()
plt.grid()
plt.colorbar()
plt.tight_layout()
plt.savefig('roc.png')
plt.clf()
