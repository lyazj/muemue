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
data = {'xs': {}, 'E': {}, 'theta': {} }  # [key][(e1_energy, min_lepton_pt)]

for emem_dirname in sorted(glob.glob('emem_example_*GeV_*GeV_0/')):
    r = re.search(r'^emem_example_([0-9.eE+-]+)GeV_pT_([0-9.eE+-]+)GeV_E_([0-9.eE+-]+)GeV_0/$', emem_dirname)
    if not r: continue
    e1_energy, min_lepton_pt, min_lepton_energy = map(float, r.groups())
    params.append((e1_energy, min_lepton_pt, min_lepton_energy))

    # Load events.
    rootpath = os.path.join(emem_dirname, 'Events', 'run_01', 'unweighted_events.root')
    print(rootpath)
    tree = uproot.concatenate(rootpath + ':LHEF', fields)

    # Select final state e- e-.
    mask = tree['Particle.Status'] == 1
    for field in tree.fields:
        if field.startswith('Particle.'): tree[field] = tree[field][mask]
    mask = tree['Particle.PID'] == [[11, 11]]  # e- e-
    assert ak.all(mask)

    # Extract e- e- momenta.
    E = np.array(tree['Particle.E'])  # [N, 2]
    p = np.array([
        tree['Particle.Px'],
        tree['Particle.Py'],
        tree['Particle.Pz'],
    ])  # [3, N, 2]
    p = np.transpose(p, [1, 2, 0])  # [N, 2, 3]
    dp = p / np.sqrt(np.sum(p*p, axis=2, keepdims=True))
    theta = np.arccos(np.minimum(1.0, dp[:,:,2]))

    data['xs'][(e1_energy, min_lepton_pt, min_lepton_energy)] = ak.mean(tree['Event.Weight'])
    data['E'][(e1_energy, min_lepton_pt, min_lepton_energy)] = E
    data['theta'][(e1_energy, min_lepton_pt, min_lepton_energy)] = theta

for e1_energy_expected in sorted(set(param[0] for param in params)):
    for min_lepton_energy_expected in sorted(set(param[2] for param in params)):
        for key, data_value in data.items():
            plt.figure(dpi=300)
            if key == 'xs':
                for (e1_energy, min_lepton_pt, min_lepton_energy), value in sorted(data_value.items()):
                    if e1_energy != e1_energy_expected: continue
                    if min_lepton_energy != min_lepton_energy_expected: continue
                    print(key, e1_energy, min_lepton_pt)
                    label = f'$E_{{e_1}} = {e1_energy}\\ \\mathrm{{GeV}},\\ p_{{\\mathrm{{T, min}}}} = ${min_lepton_pt:.0e}$\\ \\mathrm{{GeV}}$'
                    plt.scatter(0, value, label=label)
                plt.xticks()
                plt.ylabel(f'Cross section [pb]')
                plt.yscale('log')
            else:
                for (e1_energy, min_lepton_pt, min_lepton_energy), value in sorted(data_value.items()):
                    if e1_energy != e1_energy_expected: continue
                    if min_lepton_energy != min_lepton_energy_expected: continue
                    print(key, e1_energy, min_lepton_pt, min_lepton_energy)
                    label = f'$E_{{e_1}} = {e1_energy}\\ \\mathrm{{GeV}},\\ p_{{\\mathrm{{T, min}}}} = ${min_lepton_pt:.0e}$\\ \\mathrm{{GeV}},\\ E_{{\\mathrm{{min}}}} = ${min_lepton_energy:.0e}$\\ \\mathrm{{GeV}}$'
                    plt.scatter(value[:,1], value[:,0], label=label)
                plt.xlabel(f'electron 2 {key}')
                plt.ylabel(f'electron 1 {key}')
            plt.legend(loc='upper right')
            plt.grid()
            plt.tight_layout()
            plt.savefig(f'emem_{e1_energy_expected}GeV_{key}_E_{min_lepton_energy_expected:.0e}GeV.png')
            plt.close()

for param in params:
    e1_energy, min_lepton_pt, min_lepton_energy = param
    with open(f'emem_example_{e1_energy}GeV_pT_{min_lepton_pt:.0e}GeV_E_{min_lepton_energy:.0e}GeV.txt', 'w') as file:
        for theta, E in zip(data['theta'][param], data['E'][param]):
            print(' '.join(['%.18e'] * 4) % (*theta, *E), file=file)
