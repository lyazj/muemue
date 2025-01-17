#!/usr/bin/env python3

import numpy as np
import multiprocessing
import MG5Card
from get_p import get_p
import numpy as np

emem_p_card = MG5Card.MG5Card('cards/emem_p.dat')
pool = multiprocessing.Pool(6)
points = np.load('epem_example_1.0GeV_pT_0.00e+00GeV_eta_1.70e+00.npz')['points']
p_ps, p_es = np.array([get_p(point) for point in points]).transpose(1, 0, 2)

args = []
for p_e in p_es:
    incoming_energy = p_e[0]
    for min_lepton_pt in np.array([  # GeV
        0.0,
    ]):
        for min_lepton_eta in np.array([
            #0.1,
            1.0,
            #2.0,
        ]):
            r = 0
            for incoming_polarization in 'LR':
                for electron_polarization in 'L':
                args.append({
                    'workdir': f'emem_{incoming_polarization}{electron_polarization}_example_{incoming_energy:.3f}GeV_pT_{min_lepton_pt:.2e}GeV_eta_{min_lepton_eta:.2e}_{r}',
                    'nevent': 1000000, 'seed': r, 'electron_energy': 0.000511,  # GeV
                    'incoming_energy': incoming_energy,
                    'min_lepton_pt': min_lepton_pt, 'min_lepton_com_energy': -1.0, 'min_lepton_eta': min_lepton_eta,
                    'incoming_polarization': incoming_polarization, 'electron_polarization': electron_polarization,
                })
pool.map(emem_p_card.run, args)
