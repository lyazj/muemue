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
data = {'xs': {}, 'E': {}, 'theta': {} }  # [key][(incoming_eta, min_lepton_pt)]

for emem_dirname in sorted(glob.glob('emem_??_example_*GeV_*_0/')):
    r = re.search(r'^emem_([a-zA-Z])([a-zA-Z])_example_([0-9.eE+-]+)GeV_pT_([0-9.eE+-]+)GeV_eta_([0-9.eE+-]+)_[0-9]+/$', emem_dirname)
    if not r: continue
    incoming_polarization, electron_polarization = r.groups()[:2]
    incoming_energy, min_lepton_pt, min_lepton_eta = map(float, r.groups()[2:])
    params.append((incoming_polarization, electron_polarization, incoming_energy, min_lepton_pt, min_lepton_eta))

    # Load events.
    rootpath = os.path.join(emem_dirname, 'Events', 'run_01', 'unweighted_events.root')
    print(rootpath)
    tree = uproot.concatenate(rootpath + ':LHEF', fields)

    # Select final state e+ e-.
    mask = tree['Particle.Status'] == 1
    for field in tree.fields:
        if field.startswith('Particle.'): tree[field] = tree[field][mask]
    mask = tree['Particle.PID'] == [[11, 11]]  # e- e-
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

    data['xs'][(incoming_polarization, electron_polarization, incoming_energy, min_lepton_pt, min_lepton_eta)] = ak.mean(tree['Event.Weight'])
    data['E'][(incoming_polarization, electron_polarization, incoming_energy, min_lepton_pt, min_lepton_eta)] = E
    data['theta'][(incoming_polarization, electron_polarization, incoming_energy, min_lepton_pt, min_lepton_eta)] = theta

for incoming_energy_expected in sorted(set(param[2] for param in params)):
    for key, data_value in data.items():
        plotted = False
        plt.figure(dpi=300)
        if key == 'xs':
            factor = 1
            for (incoming_polarization, electron_polarization, incoming_energy, min_lepton_pt, min_lepton_eta), value in sorted(data_value.items()):
                if incoming_energy != incoming_energy_expected: continue
                plotted = True
                print(key, incoming_energy, min_lepton_pt)
                label = f'{incoming_polarization}{electron_polarization}, $E_{{e^+}} = {incoming_energy}\\ \\mathrm{{GeV}},\\ p_{{\\mathrm{{T, min}}}} = ${min_lepton_pt:.2e}$\\ \\mathrm{{GeV}},\\ \\eta_{{\\mathrm{{min}}}} = ${min_lepton_eta:.2e} ($\\times${factor})'
                plt.scatter(0, value * factor, label=label)
                factor += 1
            plt.xticks()
            plt.ylabel(f'Cross section [pb]')
            plt.yscale('log')
        else:
            factor = 1
            for (incoming_polarization, electron_polarization, incoming_energy, min_lepton_pt, min_lepton_eta), value in sorted(data_value.items()):
                if incoming_energy != incoming_energy_expected: continue
                plotted = True
                print(key, incoming_energy, min_lepton_pt, min_lepton_eta)
                label = f'{incoming_polarization}{electron_polarization}, $E_{{e^+}} = {incoming_energy}\\ \\mathrm{{GeV}},\\ p_{{\\mathrm{{T, min}}}} = ${min_lepton_pt:.2e}$\\ \\mathrm{{GeV}},\\ \\eta_{{\\mathrm{{min}}}} = ${min_lepton_eta:.2e} ($\\times${factor})'
                plt.scatter(value[:,1], value[:,0] * factor, label=label)
                factor += 1
            plt.xlabel(f'electron {key}')
            plt.ylabel(f'incoming {key}')
        if not plotted: continue
        plt.legend(loc='upper right')
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'emem_pp_{incoming_energy_expected}GeV_{key}.png')
        plt.close()

for param in params:
    incoming_polarization, electron_polarization, incoming_energy, min_lepton_pt, min_lepton_eta = param
    with open(f'emem_example_{incoming_polarization}{electron_polarization}_{incoming_energy:.3f}GeV_pT_{min_lepton_pt:.2e}GeV_eta_{min_lepton_eta:.2e}.txt', 'w') as file:
        for theta, E in zip(data['theta'][param], data['E'][param]):
            print(' '.join(['%.18e'] * 4) % (*theta, *E), file=file)
