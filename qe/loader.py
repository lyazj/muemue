import uproot
import awkward as ak
import numpy as np

fields = [
    'Event.Weight',  # XS in pb
    'Particle.PID',
    'Particle.Status',  # 1 for final
    'Particle.Px',
    'Particle.Py',
    'Particle.Pz',
    'Particle.E',
]

def load(path, pid, nevent=None, weight=False):  # expect: pid=[11, 11] pid=[-11, 11]
    tree = uproot.concatenate(path + ':LHEF', fields)
    if nevent: tree = tree[:nevent]

    # Select final state.
    mask = tree['Particle.Status'] == 1
    for field in tree.fields:
        if field.startswith('Particle.'): tree[field] = tree[field][mask]
    mask = tree['Particle.PID'] == [pid]
    assert ak.all(mask)

    # Extract e+ e- momenta.
    p = np.array([
        tree['Particle.E'],
        tree['Particle.Px'],
        tree['Particle.Py'],
        tree['Particle.Pz'],
    ])  # [4, N, len(pid)]
    p = np.transpose(p, [2, 1, 0])  # [len(pid), N, 4]

    if weight: return p, tree['Event.Weight']
    return p
