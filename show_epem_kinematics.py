#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(4.2, 3.15), dpi=300)
color = plt.rcParams['axes.prop_cycle'].by_key()['color']

#for iE_p, E_p in enumerate([2, 5, 8]):
for iE_p, E_p in enumerate([1, 3, 10]):
    m_p = 0.000511
    m_e = 0.000511
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
    
    
    theta_p = np.linspace(1e-5 * np.pi, (1 - 1e-5) * np.pi, int(1e5) - 1)
    p_p, p_e = compute_p(theta_p)
    
    #plt.plot(theta_p, p_p[0], '--', color=color[iE_p], label=r'$E_{e^+},\ E_\text{beam} = %d$ GeV' % E_p)
    #plt.plot(theta_p, p_e[0], '-',  color=color[iE_p], label=r'$E_{e^-},\ E_\text{beam} = %d$ GeV' % E_p)

    #eta_p = -np.log(np.tan(0.5 * np.arctan2(np.hypot(p_p[1], p_p[2]), p_p[3])))
    #eta_e = -np.log(np.tan(0.5 * np.arctan2(np.hypot(p_e[1], p_e[2]), p_e[3])))
    #plt.plot(theta_p, eta_p, '--', color=color[iE_p], label=r'$\eta_{e^+},\ E_\text{beam} = %d$ GeV' % E_p)
    #plt.plot(theta_p, eta_e, '-',  color=color[iE_p], label=r'$\eta_{e^-},\ E_\text{beam} = %d$ GeV' % E_p)

    #plt.plot(theta_p, np.hypot(p_p[1], p_p[2]), '.', label='positron')
    #plt.plot(theta_p, np.hypot(p_e[1], p_e[2]), '--', label='electron')
    #plt.xlabel(r'$\theta^\prime_{e^+}$')
    #plt.ylabel(r'$p_\mathrm{T}$')
    #plt.grid()
    #plt.legend()
    #plt.tight_layout()
    #plt.savefig('pT-theta_p.png')
    #plt.clf()
    
    theta_p_lab = np.arctan2(np.hypot(p_p[1], p_p[2]), p_p[0])
    theta_e_lab = np.arctan2(np.hypot(p_e[1], p_e[2]), p_e[0])
    plt.plot(theta_p, theta_p_lab, '--', color=color[iE_p], label=r'$\theta_{e^+},\ E_\mathrm{beam} = %d$ GeV' % E_p)
    plt.plot(theta_p, theta_e_lab, '-',  color=color[iE_p], label=r'$\theta_{e^-},\ E_\mathrm{beam} = %d$ GeV' % E_p)

#plt.xlabel(r'$\theta^\prime_{e^+}$ [rad]')
#plt.ylabel(r'$E$')
#plt.yscale('log')
#plt.grid()
#plt.legend()
#plt.tight_layout()
#plt.savefig('E-theta_p.png')
#plt.clf()

#plt.xlabel(r'$\theta^\prime_{e^+}$ [rad]')
#plt.ylabel(r'$\eta$')
#plt.grid()
#plt.legend()
#plt.tight_layout()
#plt.savefig('eta-theta_p.png')
#plt.clf()

plt.xlabel(r'$\theta^\prime_{e^+}$ [rad]')
plt.ylabel(r'$\theta$ [rad]')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('theta-theta_p.png')
plt.clf()
