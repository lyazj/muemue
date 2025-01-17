import uproot
import numpy as np
import awkward as ak

fields = [
    'Particle.PID',
    'Particle.Status',  # 1 for final
    'Particle.Px',
    'Particle.Py',
    'Particle.Pz',
]

tree = uproot.concatenate('epem_example_1.0GeV_pT_0.00e+00GeV_eta_1.70e+00_0/Events/run_01/unweighted_events.root:LHEF', fields)

# Select final state e+ e-.
mask = tree['Particle.Status'] == 1
for field in tree.fields:
    if field.startswith('Particle.'): tree[field] = tree[field][mask]
mask = tree['Particle.PID'] == [[-11, 11]]  # e+ e-
assert ak.all(mask)

# Extract e+ momenta.
theta = np.arctan2(np.hypot(tree['Particle.Px'], tree['Particle.Py']), tree['Particle.Pz'])
theta = theta[:,0]
weights, points = np.histogram(theta, bins=np.linspace(0.05, 0.1, 101))
weights = np.array([*weights])
points = np.mean([points[:-1], points[1:]], axis=0)
np.savez('epem_example_1.0GeV_pT_0.00e+00GeV_eta_1.70e+00.npz', points=points, weights=weights)
