from get_p import get_p
import numpy as np

data = np.load('../epem_example_1.0GeV_pT_0.00e+00GeV_eta_1.70e+00.npz')
points, weights = data['points'], data['weights']
p_ps, p_es = np.array([get_p(theta_p) for theta_p in points]).transpose(1, 0, 2)

# Construct 4-momentum from (E, m, theta, phi).
def get_P(E, m, theta, phi):
    p = np.sqrt(E*E - m*m)
    return np.array([
        E,
        p * np.sin(theta) * np.cos(phi),
        p * np.sin(theta) * np.sin(phi),
        p * np.cos(theta),
    ]).T

for P3, P4 in zip(p_ps, p_es):
    P3 = P3.reshape(1, 4)
    P4 = P4.reshape(1, 4)
    m3 = np.mean(np.sqrt(P3[:,0]**2 - np.sum(P3[:,1:]**2, axis=1)))
    m4 = np.mean(np.sqrt(P4[:,0]**2 - np.sum(P4[:,1:]**2, axis=1)))
    P2 = get_P(m4, m4, 0, 0).reshape(1, 4)
    P1 = P3 + P4 - P2

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
                        #print(i, j, k, l)
                        M[:,i,j] = M_uP(u1, u2, u3, u4, P1, P2, P3, P4)
                # Add "Kronecker product of M and M*" from the current initial state to rho.
                M = M.reshape(-1, 4)
                for i in range(4):
                    for j in range(4):
                        #rho[:,i,j] += M[:,i] * M[:,j].conjugate() / 4
                        rho[:,i,j] += M[:,i] * M[:,j].conjugate()
        # Normalize rho.
        rho /= np.trace(rho, axis1=1, axis2=2).reshape(-1, 1, 1)
        #print(rho[:1])
        return rho

    # Compute concurrence.
    rho = rho_P(P1, P2, P3, P4)
    print(rho.reshape(1, 2, 2, 2, 2).trace(axis1=2, axis2=4))
