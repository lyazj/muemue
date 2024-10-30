#!/usr/bin/env python3

import numpy as np
from MG5Run import pool
from make_muemue_point import muemue_card

args = []
muon_energy = 1.0  # GeV
for min_lepton_pt in np.array([  # GeV
    0.000001,
    0.000010,
    0.000100,
    0.001000,
    0.002000,
    0.003000,
    0.004000,
    #0.005000,
]):
    args.append({
        'workdir': f'muemue_example_{muon_energy}GeV_{min_lepton_pt:.0e}GeV',
        'nevent': 100000, 'electron_energy': 0.000511,  # GeV
        'muon_energy': muon_energy, 'min_lepton_pt': min_lepton_pt,
    })
pool.map(muemue_card.run, args)
