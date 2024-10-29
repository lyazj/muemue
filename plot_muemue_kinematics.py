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

data = {'xs': {}, 'E': {}, 'theta': {} }  # [key][(muon_energy, min_lepton_pt)]

for muemue_dirname in sorted(glob.glob('muemue_example_*GeV_*GeV/')):
    r = re.search(r'^muemue_example_([0-9.eE+-]+)GeV_([0-9.eE+-]+)GeV/$', muemue_dirname)
    if not r: continue
    muon_energy, min_lepton_pt = map(float, r.groups())

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

    data['xs'][(muon_energy, min_lepton_pt)] = ak.mean(tree['Event.Weight'])
    data['E'][(muon_energy, min_lepton_pt)] = E
    data['theta'][(muon_energy, min_lepton_pt)] = theta

for key, data_value in data.items():
    plt.figure(dpi=300)
    if key == 'xs':
        for (muon_energy, min_lepton_pt), value in data_value.items():
            print(key, muon_energy, min_lepton_pt)
            label = f'$E_\\mu = {muon_energy}\\ \\mathrm{{GeV}},\\ p_{{\\mathrm{{T, min}}}} = \\text{{{min_lepton_pt:.0e}}}\\ \\mathrm{{GeV}}$'
            plt.scatter(0, value, label=label)
        plt.xticks()
        plt.ylabel(f'Cross section [pb]')
        plt.yscale('log')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'muemue_{key}.png')
        plt.close()
    else:
        for (muon_energy, min_lepton_pt), value in data_value.items():
            print(key, muon_energy, min_lepton_pt)
            label = f'$E_\\mu = {muon_energy}\\ \\mathrm{{GeV}},\\ p_{{\\mathrm{{T, min}}}} = \\text{{{min_lepton_pt:.0e}}}\\ \\mathrm{{GeV}}$'
            plt.scatter(value[:,1], value[:,0], label=label)
        plt.xlabel(f'electron {key}')
        plt.ylabel(f'muon {key}')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'muemue_{key}.png')
        plt.close()
