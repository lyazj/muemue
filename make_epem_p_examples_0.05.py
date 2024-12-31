#!/usr/bin/env python3

import numpy as np
import multiprocessing
import MG5Card
from get_p import get_p

epem_p_card = MG5Card.MG5Card('cards/epem_p.dat')
pool = multiprocessing.Pool(6)
p_p, p_e = get_p(0.05)  # GeV

args = []
positron_energy = p_p[0]
for min_lepton_pt in np.array([  # GeV
    0.0,
]):
    for min_lepton_eta in np.array([
        #0.1,
        1.0,
        #2.0,
    ]):
        r = 0
        for positron_polarization in 'LR':
            for electron_polarization in 'LR':
                args.append({
                    'workdir': f'epem_{positron_polarization}{electron_polarization}_example_{positron_energy:.3f}GeV_pT_{min_lepton_pt:.2e}GeV_eta_{min_lepton_eta:.2e}_{r}',
                    'nevent': 100000, 'seed': r, 'electron_energy': 0.000511,  # GeV
                    'positron_energy': positron_energy,
                    'min_lepton_pt': min_lepton_pt, 'min_lepton_com_energy': -1.0, 'min_lepton_eta': min_lepton_eta,
                    'positron_polarization': positron_polarization, 'electron_polarization': electron_polarization,
                })
pool.map(epem_p_card.run, args)
