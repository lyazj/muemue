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

plt.figure(dpi=300)
cmap = plt.get_cmap('viridis')
cmap.set_bad('lightgray', 1.0)
offset = 0
for path in [  # sample files
    'epem_example_1.0GeV_pT_0.00e+00GeV_eta_1.70e+00.txt',
    'epem_example_3.0GeV_pT_0.00e+00GeV_eta_2.00e+00.txt',
    'epem_example_10.0GeV_pT_0.00e+00GeV_eta_2.80e+00.txt',
]:
    # Incoming beam energy and generator cuts.
    positron_energy, lepton_pt, lepton_eta = map(float,
        re.search(r'_([0-9.e+-]*)GeV_pT_([0-9.e+-]*)GeV_eta_([0-9.e+-]*)\.txt', path).groups())

    # Scattering particle masses.
    m_p = 0.511e-3
    m_e = 0.511e-3

    # Lorentz transformation.
    gamma = (positron_energy + m_e) / np.sqrt(m_p**2 + m_e**2 + 2*m_e*positron_energy)
    beta = np.sqrt(1 - 1 / gamma**2)
    def lab_to_com(P):
        P0 = gamma * (P[:,0] - beta * P[:,3])
        P3 = gamma * (P[:,3] - beta * P[:,0])
        P[:,0] = P0
        P[:,3] = P3
        return P

    # Read observables from the sample file.
    theta_p, theta_e, E_p, E_e = np.array(
        open(path).read().strip().split(), dtype='float'
    ).reshape(-1, 4)[:(NEVENT_MAX if NEVENT_MAX else int(1e20))].T

    # Incoming beam.
    P1 = np.repeat(get_P(positron_energy, m_p, 0, 0).reshape(1, 4), E_p.shape[0], axis=0)

    # Target particle.
    P2 = np.repeat(get_P(m_e, m_e, 0, 0).reshape(1, 4), E_e.shape[0], axis=0)

    # Outgoing beam.
    P3 = get_P(E_p, m_p, theta_p, 0)

    # Recoiled particle.
    P4 = get_P(E_e, m_e, theta_e, np.pi)

    if COM_FRAME:
        # Transform to the center of mass frame.
        P1, P2, P3, P4 = map(lab_to_com, (P1, P2, P3, P4))
        # In the case of in a transformed frame, recompute observables.
        theta_p = np.arctan2(np.hypot(P3[:,1], P3[:,2]), P3[:,3])
        theta_e = np.arctan2(np.hypot(P4[:,1], P4[:,2]), P4[:,3])
        E_p = P3[:,0]
        E_e = P4[:,0]

    print('theta1 =', theta_p[0])
    print('theta2 =', theta_e[0])
    print('E3 =', P3[0,0])
    print('P1 = ', P1[0])
    print('P2 = ', P2[0])
    print('P3 = ', P3[0])
    print('P4 = ', P4[0])

    #theta_p, P1, P2, P3, P4 = map(lambda x: x[:1], (theta_p, P1, P2, P3, P4))  # [DEBUG]

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
    def M_uP(u1, u2, u3, u4, P1, P2, P3, P4):
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

    # Construct density matrix from 4-momenta.
    def rho_P(P1, P2, P3, P4):
        u1s, u3s = map(lambda P: [v_PH(P, 1), v_PH(P, -1)], (P1, P3))
        u2s, u4s = map(lambda P: [u_PH(P, 1), u_PH(P, -1)], (P2, P4))
        #print(*u1s, *u2s, *u3s, *u4s, sep='\n')
        rho = np.zeros((P1.shape[0], 4, 4), dtype='complex')
        # Sum rho over initial state spin states.
        for k in range(2):
            u1 = u1s[k]
            for l in range(2):
                u2 = u2s[l]
                # Compute scattering amplitude matrix given the current initial state.
                M = np.empty((P1.shape[0], 2, 2), dtype='complex')
                for i in range(2):
                    u3 = u3s[i]
                    for j in range(2):
                        u4 = u4s[j]
                        print(i, j, k, l)
                        M[:,i,j] = M_uP(u1, u2, u3, u4, P1, P2, P3, P4)
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

    # Compute R matrix and concurrence.
    rho = rho_P(P1, P2, P3, P4)
    R = rho @ np.kron(sigma[1], sigma[1]) @ rho.conj() @ np.kron(sigma[1], sigma[1])
    eigenvalues = np.linalg.eigvals(R).real
    eigenvalues *= (np.abs(eigenvalues) >= 1e-10)
    eigenvalues = np.sort(np.sqrt(eigenvalues), axis=1)
    concurrence = eigenvalues[:,-1] - np.sum(eigenvalues[:,:-1], axis=1)
    concurrence = np.maximum(concurrence, 0)
    print('Positron energy:', positron_energy)
    print('Efficiency:', np.mean(concurrence > 0))
    print('Efficiency [0.05, 0.10]:', np.mean(np.logical_and(np.logical_and(theta_p >= 0.05, theta_p <= 0.10), concurrence > 0)))
    print('Max concurrence:', np.max(concurrence))
    print('Min theta_p:', np.min(theta_p[concurrence > 0]))
    print('Min theta_e:', np.min(theta_e[concurrence > 0]))
    print('Min E_p:', np.min(E_p[concurrence > 0]))
    print('Min E_e:', np.min(E_e[concurrence > 0]))
    print('Max concurrence [0.05, 0.10]:', np.max(concurrence[np.logical_and(theta_p >= 0.05, theta_p <= 0.10)]))
    print('Min theta_p [0.05, 0.10]:', np.min(theta_p[np.logical_and(np.logical_and(theta_p >= 0.05, theta_p <= 0.10), concurrence > 0)]))
    print('Min theta_e [0.05, 0.10]:', np.min(theta_e[np.logical_and(np.logical_and(theta_p >= 0.05, theta_p <= 0.10), concurrence > 0)]))
    print('Min E_p [0.05, 0.10]:', np.min(E_p[np.logical_and(np.logical_and(theta_p >= 0.05, theta_p <= 0.10), concurrence > 0)]))
    print('Min E_e [0.05, 0.10]:', np.min(E_e[np.logical_and(np.logical_and(theta_p >= 0.05, theta_p <= 0.10), concurrence > 0)]))

    if BELL:
        corr = np.sum(sigma_kron.reshape(1, 3, 3, 16) * rho.transpose(0, 2, 1).reshape(-1, 1, 1, 16), axis=3)
        print(corr[0])
        eigenvalues = np.linalg.eigvals(corr.conjugate().transpose(0, 2, 1) @ corr).real
        eigenvalues *= (np.abs(eigenvalues) >= 1e-10)
        eigenvalues = np.sort(np.sqrt(eigenvalues), axis=1)
        CHSH = 2 * np.sqrt(np.sum(eigenvalues[:,-2:], axis=1))
        print('Efficiency:', np.mean(CHSH[concurrence > 0] > 2))
        print('Max CHSH:', np.max(CHSH))
        print('Min CHSH [0.05, 0.10]:', np.min(CHSH[np.logical_and(theta_p >= 0.05, theta_p <= 0.10)]))

    # Plot concurrence as a function of theta_p and theta_e.
    data = np.array([theta_p, theta_e, CHSH if BELL else concurrence]).T
    data = data[data[:,0].argsort()]
    data = np.concatenate([data[-1:], data[:-1]])
    if COM_FRAME:
        #data = data[data[:,0] >= np.pi / 2]
        pass
    if BELL:
        colors = cmap(mcolors.Normalize(vmin=2, vmax=2.8)(data[:,2]))
        colors[data[:,2] <= 2] = [0.8, 0.8, 0.8, 1]

    if STACK:
        if BELL:
            plt.scatter(data[:,0], data[:,1] + offset, c=colors,
                 label=r'$E_\text{beam}$ = %.0f GeV  $\eta \geq$%.2f  ($\theta^\prime_e + %g$)' % (positron_energy, lepton_eta, offset))
        else:
            plt.scatter(data[:,0], data[:,1] + offset, c=data[:,2], cmap=cmap, norm=mcolors.LogNorm(vmin=1e-2, vmax=1),
                 label=r'$E_\text{beam}$ = %.0f GeV  $\eta \geq$%.2f  ($\theta^\prime_e + %g$)' % (positron_energy, lepton_eta, offset))
        if COM_FRAME:
            offset += 1
        else:
            offset += 0.1
    else:
        if BELL:
            plt.scatter(data[:,0], data[:,1], c=colors,
                 label=r'$E_\text{beam}$ = %.0f GeV  $\eta \geq$%.2f' % (positron_energy, lepton_eta))
        else:
            plt.scatter(data[:,0], data[:,1], c=data[:,2], cmap=cmap, norm=mcolors.LogNorm(vmin=1e-2, vmax=1),
                 label=r'$E_\text{beam}$ = %.0f GeV  $\eta \geq$%.2f' % (positron_energy, lepton_eta))
        if COM_FRAME:
            plt.xlabel(r'$\theta^\prime_{e^+}$ [rad] (center of mass frame)')
            plt.ylabel(r'$\theta^\prime_{e^-}$ [rad]')
        else:
            plt.xlabel(r'$\theta_{e^+}$ [rad] (lab frame)')
            plt.ylabel(r'$\theta_{e^-}$ [rad]')
        plt.legend()
        plt.grid()
        cbar = plt.colorbar()
        cbar.set_label(r'Optimal CHSH value $I_2$' if BELL else r'Concurrence $\mathcal{C}(\rho_f)$')
        if BELL:
            cbar.set_ticks(np.linspace(0, 1, 9))
            cbar.set_ticklabels(map(lambda x: '%.2f' % x, np.linspace(2, 2.8, 9)))
        plt.tight_layout()
        plt.savefig(path.replace('.txt', ('_CHSH' if BELL else '') + ('_com' if COM_FRAME else '_lab') + '.png'))
        plt.clf()

if STACK:
    if COM_FRAME:
        plt.xlabel(r'$\theta^\prime_{e^+}$ [rad] (center of mass frame)')
        plt.ylabel(r'$\theta^\prime_{e^-}$ [rad]')
    else:
        plt.xlabel(r'$\theta_{e^+}$ [rad] (lab frame)')
        plt.ylabel(r'$\theta_{e^-}$ [rad]')
    plt.legend()
    plt.grid()
    cbar = plt.colorbar()
    if BELL:
        cbar.set_ticks(np.linspace(0, 1, 9))
        cbar.set_ticklabels(map(lambda x: '%.2f' % x, np.linspace(2, 2.8, 9)))
    cbar.set_label(r'Optimal CHSH value $I_2$' if BELL else r'Concurrence $\mathcal{C}(\rho_f)$')
    plt.tight_layout()
    plt.savefig(__file__.replace('.py', ('_CHSH' if BELL else '') + ('_com' if COM_FRAME else '_lab') + '.png'))
