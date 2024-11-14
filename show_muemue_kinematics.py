#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

m_mu = 0.106
m_e = 0.000511
E_mu = 160
E = E_mu + m_e
m_com = np.sqrt(m_mu**2 + m_e**2 + 2*m_e*E_mu)
gamma = E / m_com
beta = np.sqrt(1 - 1/gamma**2)
p_com = np.sqrt((
    m_mu**4 + m_e**4 + m_com**4 - 2*m_mu**2*m_e**2 - 2*m_e**2*m_com**2 - 2*m_mu**2*m_com**2
) / (4*m_com**2))

def com_to_lab(p):
    return np.array([
        gamma * (p[0] + beta * p[3]),
        p[1],
        p[2],
        gamma * (p[3] + beta * p[0]),
    ])

def compute_p(theta_mu):
    return com_to_lab(np.array([
        np.ones_like(theta_mu) * np.sqrt(p_com**2 + m_mu**2),
        p_com * np.sin(theta_mu),
        np.zeros_like(theta_mu),
        p_com * np.cos(theta_mu),
    ])), com_to_lab(np.array([
        np.ones_like(theta_mu) * np.sqrt(p_com**2 + m_e**2),
        -p_com * np.sin(theta_mu),
        np.zeros_like(theta_mu),
        -p_com * np.cos(theta_mu),
    ]))

theta_mu = np.linspace(0.01 * np.pi, np.pi, 100)
p_mu, p_e = compute_p(theta_mu)

plt.plot(theta_mu, p_mu[0], label='muon')
plt.plot(theta_mu, p_e[0], label='electron')
plt.xlabel(r'$\theta^\prime_{\mu}$')
plt.ylabel(r'$E$')
plt.yscale('log')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('E-theta_mu.pdf')
plt.close()

eta_mu = -np.log(np.tan(0.5 * np.arctan2(np.hypot(p_mu[1], p_mu[2]), p_mu[3])))
eta_e = -np.log(np.tan(0.5 * np.arctan2(np.hypot(p_e[1], p_e[2]), p_e[3])))
plt.plot(theta_mu, eta_mu, label='muon')
plt.plot(theta_mu, eta_e, label='electron')
print(np.min([eta_mu, eta_e]))
plt.xlabel(r'$\theta^\prime_{\mu}$')
plt.ylabel(r'$\eta$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('eta-theta_mu.pdf')
plt.close()

plt.plot(theta_mu, np.hypot(p_mu[1], p_mu[2]), '.', label='muon')
plt.plot(theta_mu, np.hypot(p_e[1], p_e[2]), '--', label='electron')
plt.xlabel(r'$\theta^\prime_{\mu}$')
plt.ylabel(r'$p_\mathrm{T}$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('pT-theta_mu.pdf')
plt.close()
