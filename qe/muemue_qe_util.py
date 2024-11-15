import numpy as np
from scipy.optimize import minimize
from multiprocessing import Pool

class MuEQEUtil:

    m_e = 0.511e-3
    m_mu = 106e-3

    def __init__(self, muon_energy):
        m_e, m_mu = self.m_e, self.m_mu

        gamma = (muon_energy + m_e) / np.sqrt(m_e**2 + m_mu**2 + 2*m_e*muon_energy)
        beta = np.sqrt(1 - 1 / gamma**2)
        m_com = np.sqrt(m_mu**2 + m_e**2 + 2*m_e*muon_energy)
        p_com = np.sqrt((
            m_mu**4 + m_e**4 + m_com**4 - 2 * m_mu**2 * m_e**2 - 2 * m_e**2 * m_com**2 - 2 * m_mu**2 * m_com**2
        ) / (4 * m_com**2))
        E_mu_com = np.hypot(p_com, m_mu)
        E_e_com = np.hypot(p_com, m_e)

        self.muon_energy = muon_energy
        self.gamma, self.beta = gamma, beta
        self.m_com, self.p_com, self.E_mu_com, self.E_e_com = m_com, p_com, E_mu_com, E_e_com

    def get_P(self, E, m, theta, phi):
        p = np.sqrt(E*E - m*m)
        return np.array([
            E,
            p * np.sin(theta) * np.cos(phi),
            p * np.sin(theta) * np.sin(phi),
            p * np.cos(theta),
        ]).T

    def lab_to_com(self, P):
        gamma, beta = self.gamma, self.beta
        P0 = gamma * (P[:,0] - beta * P[:,3])
        P3 = gamma * (P[:,3] - beta * P[:,0])
        P[:,0] = P0
        P[:,3] = P3

    def com_to_lab(self, P):
        gamma, beta = self.gamma, self.beta
        P0 = gamma * (P[:,0] + beta * P[:,3])
        P3 = gamma * (P[:,3] + beta * P[:,0])
        P[:,0] = P0
        P[:,3] = P3

    def get_observable(self, theta_mu_com):
        gamma, beta = self.gamma, self.beta
        m_com, p_com, E_mu_com, E_e_com = self.m_com, self.p_com, self.E_mu_com, self.E_e_com
        P3 = np.array([
            E_mu_com * np.ones_like(theta_mu_com),
            p_com * np.sin(theta_mu_com),
            np.zeros_like(theta_mu_com),
            p_com * np.cos(theta_mu_com),
        ]).T
        P4 = np.array([
            E_e_com * np.ones_like(theta_mu_com),
            -p_com * np.sin(theta_mu_com),
            np.zeros_like(theta_mu_com),
            -p_com * np.cos(theta_mu_com),
        ]).T
        self.com_to_lab(P3)
        self.com_to_lab(P4)
        return np.array([
            np.arctan2(np.hypot(P3[:,1], P3[:,2]), P3[:,3]),  # theta_mu
            np.arctan2(np.hypot(P4[:,1], P4[:,2]), P4[:,3]),  # theta_e
            P3[:,0],  # E_mu
            P4[:,0],  # E_e
        ]).T

    def get_chi2(self, theta_mu_com, obs, obs_unc):
        obs_theta = self.get_observable(theta_mu_com)
        return np.sum(np.square((obs_theta - obs) / obs_unc), axis=-1)

    def do_fit(self, i, theta_mu_com, args):
        r = minimize(self.get_chi2, theta_mu_com, args)
        if i % 10000 == 0: print('%8d\t%.3e\t%.3e\t%.3e' % (i, theta_mu_com, r.x[0], r.fun))
        return r

    def fit(self, obs, obs_unc):
        pool = Pool()
        theta_mu, theta_e, E_mu, E_e = obs.T
        P3 = self.get_P(E_mu, self.m_mu, theta_mu, 0)
        self.lab_to_com(P3)
        theta_mu_com = np.arctan2(np.hypot(P3[:,1], P3[:,2]), P3[:,3])
        chi2 = np.empty_like(theta_mu_com)
        args = []
        for i in range(len(theta_mu_com)):
            args.append((i, theta_mu_com[i], (obs[i], obs_unc[i])))
        for i, r in enumerate(pool.starmap(self.do_fit, args)):
            theta_mu_com[i] = r.x[0]
            chi2[i] = r.fun
        return theta_mu_com, chi2
