#!/usr/bin/env python3

import MG5Card
import MG5Units

muemue_card = MG5Card.MG5Card('cards/muemue.dat')

def make_muemue_point(muon_energy, min_lepton_pt, min_lepton_eta, nevent=100000):
    print('make_muemue_point:', muon_energy, sep='\t', flush=True)
    xs, xs_unc, unit, nevent = muemue_card.run_xs({
        'nevent': nevent, 'electron_energy': 0.511e-3,
        'muon_energy': muon_energy, 'min_lepton_pt': min_lepton_pt, 'min_lepton_eta': min_lepton_eta
    })
    unit = MG5Units.units[unit]
    xs *= unit
    xs_unc *= unit
    print('make_muemue_point:', muon_energy, xs, xs_unc, sep='\t', flush=True)
    return xs, xs_unc
