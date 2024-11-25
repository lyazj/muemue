#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

m_p = 0.000511
m_e = 0.000511
E_p = 1
E = E_p + m_e
m_com = np.sqrt(m_p**2 + m_e**2 + 2*m_e*E_p)
gamma = E / m_com
beta = np.sqrt(1 - 1/gamma**2)
p_com = np.sqrt((
    m_p**4 + m_e**4 + m_com**4 - 2*m_p**2*m_e**2 - 2*m_e**2*m_com**2 - 2*m_p**2*m_com**2
) / (4*m_com**2))

def com_to_lab(p):
    return np.array([
        gamma * (p[0] + beta * p[3]),
        p[1],
        p[2],
        gamma * (p[3] + beta * p[0]),
    ])

def compute_p(theta_p):
    return com_to_lab(np.array([
        np.ones_like(theta_p) * np.sqrt(p_com**2 + m_p**2),
        p_com * np.sin(theta_p),
        np.zeros_like(theta_p),
        p_com * np.cos(theta_p),
    ])), com_to_lab(np.array([
        np.ones_like(theta_p) * np.sqrt(p_com**2 + m_e**2),
        -p_com * np.sin(theta_p),
        np.zeros_like(theta_p),
        -p_com * np.cos(theta_p),
    ]))

theta_p = np.linspace(0.01 * np.pi, np.pi, 100)
p_p, p_e = compute_p(theta_p)

plt.plot(theta_p, p_p[0], label='positron')
plt.plot(theta_p, p_e[0], label='electron')
plt.xlabel(r'$\theta^\prime_{e^+}$')
plt.ylabel(r'$E$')
plt.yscale('log')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('E-theta_p.pdf')
plt.close()

eta_p = -np.log(np.tan(0.5 * np.arctan2(np.hypot(p_p[1], p_p[2]), p_p[3])))
eta_e = -np.log(np.tan(0.5 * np.arctan2(np.hypot(p_e[1], p_e[2]), p_e[3])))
plt.plot(theta_p, eta_p, label='positron')
plt.plot(theta_p, eta_e, label='electron')
plt.xlabel(r'$\theta^\prime_{e^+}$')
plt.ylabel(r'$\eta$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('eta-theta_p.pdf')
plt.close()

plt.plot(theta_p, np.hypot(p_p[1], p_p[2]), '.', label='positron')
plt.plot(theta_p, np.hypot(p_e[1], p_e[2]), '--', label='electron')
plt.xlabel(r'$\theta^\prime_{e^+}$')
plt.ylabel(r'$p_\mathrm{T}$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('pT-theta_p.pdf')
plt.close()
