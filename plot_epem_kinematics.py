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
data = {'xs': {}, 'E': {}, 'theta': {} }  # [key][(positron_eta, min_lepton_pt)]

for epem_dirname in sorted(glob.glob('epem_example_*GeV_*_0/')):
    r = re.search(r'^epem_example_([0-9.eE+-]+)GeV_pT_([0-9.eE+-]+)GeV_eta_([0-9.eE+-]+)_[0-9]+/$', epem_dirname)
    if not r: continue
    positron_energy, min_lepton_pt, min_lepton_eta = map(float, r.groups())
    params.append((positron_energy, min_lepton_pt, min_lepton_eta))

    # Load events.
    rootpath = os.path.join(epem_dirname, 'Events', 'run_01', 'unweighted_events.root')
    print(rootpath)
    tree = uproot.concatenate(rootpath + ':LHEF', fields)

    # Select final state e+ e-.
    mask = tree['Particle.Status'] == 1
    for field in tree.fields:
        if field.startswith('Particle.'): tree[field] = tree[field][mask]
    mask = tree['Particle.PID'] == [[-11, 11]]  # e+ e-
    assert ak.all(mask)

    # Extract e+ e- momenta.
    E = np.array(tree['Particle.E'])  # [N, 2]
    p = np.array([
        tree['Particle.Px'],
        tree['Particle.Py'],
        tree['Particle.Pz'],
    ])  # [3, N, 2]
    p = np.transpose(p, [1, 2, 0])  # [N, 2, 3]
    dp = p / np.sqrt(np.sum(p*p, axis=2, keepdims=True))
    theta = np.arccos(np.minimum(1.0, dp[:,:,2]))

    data['xs'][(positron_energy, min_lepton_pt, min_lepton_eta)] = ak.mean(tree['Event.Weight'])
    data['E'][(positron_energy, min_lepton_pt, min_lepton_eta)] = E
    data['theta'][(positron_energy, min_lepton_pt, min_lepton_eta)] = theta

for positron_energy_expected in sorted(set(param[0] for param in params)):
    for key, data_value in data.items():
        plotted = False
        plt.figure(dpi=300)
        if key == 'xs':
            for (positron_energy, min_lepton_pt, min_lepton_eta), value in sorted(data_value.items()):
                if positron_energy != positron_energy_expected: continue
                plotted = True
                print(key, positron_energy, min_lepton_pt)
                label = f'$E_{{e^+}} = {positron_energy}\\ \\mathrm{{GeV}},\\ p_{{\\mathrm{{T, min}}}} = ${min_lepton_pt:.2e}$\\ \\mathrm{{GeV}},\\ \\eta_{{\\mathrm{{min}}}} = ${min_lepton_eta:.2e}'
                plt.scatter(0, value, label=label)
            plt.xticks()
            plt.ylabel(f'Cross section [pb]')
            plt.yscale('log')
        else:
            for (positron_energy, min_lepton_pt, min_lepton_eta), value in sorted(data_value.items()):
                if positron_energy != positron_energy_expected: continue
                plotted = True
                print(key, positron_energy, min_lepton_pt, min_lepton_eta)
                label = f'$E_{{e^+}} = {positron_energy}\\ \\mathrm{{GeV}},\\ p_{{\\mathrm{{T, min}}}} = ${min_lepton_pt:.2e}$\\ \\mathrm{{GeV}},\\ \\eta_{{\\mathrm{{min}}}} = ${min_lepton_eta:.2e}'
                plt.scatter(value[:,1], value[:,0], label=label)
            plt.xlabel(f'electron {key}')
            plt.ylabel(f'positron {key}')
        if not plotted: continue
        plt.legend(loc='upper right')
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'epem_{positron_energy_expected}GeV_{key}.png')
        plt.close()

for param in params:
    positron_energy, min_lepton_pt, min_lepton_eta = param
    with open(f'epem_example_{positron_energy}GeV_pT_{min_lepton_pt:.2e}GeV_eta_{min_lepton_eta:.2e}.txt', 'w') as file:
        for theta, E in zip(data['theta'][param], data['E'][param]):
            print(' '.join(['%.18e'] * 4) % (*theta, *E), file=file)