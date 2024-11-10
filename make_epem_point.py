#!/usr/bin/env python3

import MG5Card
import MG5Units

epem_card = MG5Card.MG5Card('cards/epem.dat')

def make_epem_point(e1_energy, min_lepton_pt, nevent=100000):
    print('make_epem_point:', e1_energy, sep='\t', flush=True)
    xs, xs_unc, unit, nevent = epem_card.run_xs({
        'nevent': nevent, 'e2_energy': 0.511e-3,
        'e1_energy': e1_energy, 'min_lepton_pt': min_lepton_pt,
    })
    unit = MG5Units.units[unit]
    xs *= unit
    xs_unc *= unit
    print('make_epem_point:', e1_energy, xs, xs_unc, sep='\t', flush=True)
    return xs, xs_unc
