#!/usr/bin/env python3

import numpy as np
import multiprocessing
from make_muemue_point import muemue_card

pool = multiprocessing.Pool(6)

args = []
muon_energy = 1.0  # GeV
for min_lepton_pt in np.array([  # GeV
    0.0,
]):
    for min_lepton_eta in np.array([
        #2.8,
        2.85,
        #2.89,  # No QE.
    ]):
        repeat = 1
        for r in range(repeat):
            args.append({
                'workdir': f'muemue_example_{muon_energy}GeV_pT_{min_lepton_pt:.2e}GeV_eta_{min_lepton_eta:.2e}_{r}',
                'nevent': 1000000, 'seed': r, 'electron_energy': 0.000511,  # GeV
                'muon_energy': muon_energy,
                'min_lepton_pt': min_lepton_pt, 'min_lepton_com_energy': -1.0, 'min_lepton_eta': min_lepton_eta,
            })
pool.map(muemue_card.run, args)
