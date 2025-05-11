#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(4.2, 3.15), dpi=300)
color = plt.rcParams['axes.prop_cycle'].by_key()['color']

m_p = 0.000511
m_e = 0.000511
E_p = np.linspace(0, 10, 10001)
m_com = np.sqrt(m_p**2 + m_e**2 + 2*m_e*E_p)
plt.plot(E_p, m_com, label=r'$e^+e^- \to e^+e^-$')
plt.xlabel(r'$E_\mathrm{beam}$ [GeV]')
plt.ylabel(r'$E_\mathrm{COM}$ [GeV]')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('E_COM-E_beam.png')
plt.clf()
