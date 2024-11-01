#!/usr/bin/env python3

import os
import re
import glob
import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

fields = [
    'Event.Weight',  # XS in pb
    'Particle.PID',
    'Particle.Status',  # 1 for final
    'Particle.Px',
    'Particle.Py',
    'Particle.Pz',
    'Particle.E',
]

params = []
data = {'xs': {}, 'E': {}, 'theta': {} }  # [key][(muon_energy, min_lepton_pt)]

for muemue_dirname in sorted(glob.glob('muemue_example_*GeV_*GeV_0/')):
    r = re.search(r'^muemue_example_([0-9.eE+-]+)GeV_pT_([0-9.eE+-]+)GeV_E_([0-9.eE+-]+)GeV_0/$', muemue_dirname)
    if not r: continue
    muon_energy, min_lepton_pt, min_lepton_energy = map(float, r.groups())
    params.append((muon_energy, min_lepton_pt, min_lepton_energy))

    # Load events.
    rootpath = os.path.join(muemue_dirname, 'Events', 'run_01', 'unweighted_events.root')
    print(rootpath)
    tree = uproot.concatenate(rootpath + ':LHEF', fields)

    # Select final state mu- e-.
    mask = tree['Particle.Status'] == 1
    for field in tree.fields:
        if field.startswith('Particle.'): tree[field] = tree[field][mask]
    mask = tree['Particle.PID'] == [[13, 11]]  # mu- e-
    assert ak.all(mask)

    # Extract mu- e- momenta.
    E = np.array(tree['Particle.E'])  # [N, 2]
    p = np.array([
        tree['Particle.Px'],
        tree['Particle.Py'],
        tree['Particle.Pz'],
    ])  # [3, N, 2]
    p = np.transpose(p, [1, 2, 0])  # [N, 2, 3]
    dp = p / np.sqrt(np.sum(p*p, axis=2, keepdims=True))
    theta = np.arccos(np.minimum(1.0, dp[:,:,2]))

    data['xs'][(muon_energy, min_lepton_pt, min_lepton_energy)] = ak.mean(tree['Event.Weight'])
    data['E'][(muon_energy, min_lepton_pt, min_lepton_energy)] = E
    data['theta'][(muon_energy, min_lepton_pt, min_lepton_energy)] = theta

for muon_energy_expected in sorted(set(param[0] for param in params)):
    for min_lepton_energy_expected in sorted(set(param[2] for param in params)):
        for key, data_value in data.items():
            plt.figure(dpi=300)
            if key == 'xs':
                for (muon_energy, min_lepton_pt, min_lepton_energy), value in sorted(data_value.items()):
                    if muon_energy != muon_energy_expected: continue
                    if min_lepton_energy != min_lepton_energy_expected: continue
                    print(key, muon_energy, min_lepton_pt)
                    label = f'$E_\\mu = {muon_energy}\\ \\mathrm{{GeV}},\\ p_{{\\mathrm{{T, min}}}} = \\text{{{min_lepton_pt:.0e}}}\\ \\mathrm{{GeV}}$'
                    plt.scatter(0, value, label=label)
                plt.xticks()
                plt.ylabel(f'Cross section [pb]')
                plt.yscale('log')
            else:
                for (muon_energy, min_lepton_pt, min_lepton_energy), value in sorted(data_value.items()):
                    if muon_energy != muon_energy_expected: continue
                    if min_lepton_energy != min_lepton_energy_expected: continue
                    print(key, muon_energy, min_lepton_pt, min_lepton_energy)
                    label = f'$E_\\mu = {muon_energy}\\ \\mathrm{{GeV}},\\ p_{{\\mathrm{{T, min}}}} = \\text{{{min_lepton_pt:.0e}}}\\ \\mathrm{{GeV}},\\ E_{{\\mathrm{{min}}}} = \\text{{{min_lepton_energy:.0e}}}\\ \\mathrm{{GeV}}$'
                    plt.scatter(value[:,1], value[:,0], label=label)
                plt.xlabel(f'electron {key}')
                plt.ylabel(f'muon {key}')
            plt.legend(loc='upper right')
            plt.grid()
            plt.tight_layout()
            plt.savefig(f'muemue_{muon_energy_expected}GeV_{key}_E_{min_lepton_energy_expected:.0e}GeV.png')
            plt.close()

for param in params:
    muon_energy, min_lepton_pt, min_lepton_energy = param
    with open(f'muemue_example_{muon_energy}GeV_pT_{min_lepton_pt:.0e}GeV_E_{min_lepton_energy:.0e}GeV.txt', 'w') as file:
        for theta, E in zip(data['theta'][param], data['E'][param]):
            print(' '.join(['%.18e'] * 4) % (*theta, *E), file=file)
