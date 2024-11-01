#!/usr/bin/env python3

import numpy as np
import multiprocessing
from make_muemue_point import muemue_card

pool = multiprocessing.Pool(6)

args = []
muon_energy = 1.0  # GeV
for min_lepton_pt in np.array([  # GeV
    0.000100,
    0.001000,
    0.004000,
]):
    for min_lepton_energy in np.array([  # GeV
        #0.000100,
        #0.001000,
        0.004000,
    ]):
        if min_lepton_energy < min_lepton_pt: continue
        repeat = 1
        if min_lepton_pt == 0.004000 and min_lepton_energy == 0.004000: repeat = 15
        for r in range(repeat):
            args.append({
                'workdir': f'muemue_example_{muon_energy}GeV_pT_{min_lepton_pt:.0e}GeV_E_{min_lepton_energy:.0e}GeV_{r}',
                'nevent': 100000, 'seed': r, 'electron_energy': 0.000511,  # GeV
                'muon_energy': muon_energy, 'min_lepton_pt': min_lepton_pt, 'min_lepton_energy': min_lepton_energy,
            })
pool.map(muemue_card.run, args)
