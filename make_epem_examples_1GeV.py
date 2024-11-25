#!/usr/bin/env python3

import numpy as np
import multiprocessing
import MG5Card

epem_card = MG5Card.MG5Card('cards/epem.dat')
pool = multiprocessing.Pool(6)

args = []
positron_energy = 1.0  # GeV
for min_lepton_pt in np.array([  # GeV
    0.0,
]):
    for min_lepton_eta in np.array([
        0.1,
        1.0,
        2.0,
        3.0,
    ]):
        repeat = 1
        for r in range(repeat):
            args.append({
                'workdir': f'epem_example_{positron_energy}GeV_pT_{min_lepton_pt:.2e}GeV_eta_{min_lepton_eta:.2e}_{r}',
                'nevent': 100000, 'seed': r, 'electron_energy': 0.000511,  # GeV
                'positron_energy': positron_energy,
                'min_lepton_pt': min_lepton_pt, 'min_lepton_com_energy': -1.0, 'min_lepton_eta': min_lepton_eta,
            })
pool.map(epem_card.run, args)
