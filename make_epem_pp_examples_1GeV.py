#!/usr/bin/env python3

import numpy as np
import multiprocessing
import MG5Card

epem_pp_card = MG5Card.MG5Card('cards/epem_pp.dat')
pool = multiprocessing.Pool(6)

args = []
positron_energy = 1.0  # GeV
for min_lepton_pt in np.array([  # GeV
    0.0,
]):
    for min_lepton_eta in np.array([
        #0.1,
        #1.0,
        2.0,
        #3.0,
    ]):
        r = 0
        for positron_polarization in 'LR':
            for electron_polarization in 'LR':
                args.append({
                    'workdir': f'epem_{positron_polarization}{electron_polarization}_example_{positron_energy}GeV_pT_{min_lepton_pt:.2e}GeV_eta_{min_lepton_eta:.2e}_{r}',
                    'nevent': 100000, 'seed': r, 'electron_energy': 0.000511,  # GeV
                    'positron_energy': positron_energy,
                    'min_lepton_pt': min_lepton_pt, 'min_lepton_com_energy': -1.0, 'min_lepton_eta': min_lepton_eta,
                    'positron_polarization': positron_polarization, 'electron_polarization': electron_polarization,
                })
pool.map(epem_pp_card.run, args)
