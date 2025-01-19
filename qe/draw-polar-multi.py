import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import loader

# Optional maximum number of events to process.
NEVENT_MAX = None
COM_FRAME = True

# Construct 4-momentum from (E, m, theta, phi).
def get_P(E, m, theta, phi):
    p = np.sqrt(E*E - m*m)
    return np.array([
        E,
        p * np.sin(theta) * np.cos(phi),
        p * np.sin(theta) * np.sin(phi),
        p * np.cos(theta),
    ]).T

def load(path, pid):
    # Outgoing beams.
    (P3, P4), weight = loader.load(path, pid, NEVENT_MAX, weight=True)
    m3 = np.mean(np.sqrt(P3[:,0]**2 - np.sum(P3[:,1:]**2, axis=1)))
    m4 = np.mean(np.sqrt(P4[:,0]**2 - np.sum(P4[:,1:]**2, axis=1)))

    # Target particle.
    P2 = np.repeat(get_P(m4, m4, 0, 0).reshape(1, 4), P4.shape[0], axis=0)

    # Incoming beam.
    P1 = P3 + P4 - P2

    if COM_FRAME:
        incoming_energy = np.mean(P1[:,0])
        gamma = (incoming_energy + m4) / np.sqrt(m3**2 + m4**2 + 2*m4*incoming_energy)
        beta = np.sqrt(1 - 1 / gamma**2)
        def lab_to_com(P):
            P0 = gamma * (P[:,0] - beta * P[:,3])
            P3 = gamma * (P[:,3] - beta * P[:,0])
            P[:,0] = P0
            P[:,3] = P3
            return P
        P1, P2, P3, P4 = map(lab_to_com, (P1, P2, P3, P4))

    return P1, P2, P3, P4, weight

def load_pair(epem_path, emem_path):
    P3, P5, P7, P8, W3 = load(epem_path, [-11, 11])
    P4, P6, P9, P10, W4 = load(emem_path, [11, 11])
    return P3, P4, P5, P6, P7, P8, P9, P10, W3 * W4

data = np.load('../epem_example_1.0GeV_pT_0.00e+00GeV_eta_1.70e+00.npz')
points, weights = data['points'], data['weights']

plt.figure(figsize=(4, 3), dpi=300)
for pp in 'LR':
    for pm in 'LR':
        theta_7s, theta_9s, Ws = [], [], []
        for theta_3, W12 in zip(points, weights):
            P3, P4, P5, P6, P7, P8, P9, P10, W34 = load_pair(*[
                f'../epem_{pp}L_example_{theta_3:.4f}rad_pT_0.00e+00GeV_eta_1.00e+00_0/Events/run_01/unweighted_events.root',
                f'../emem_{pm}L_example_{theta_3:.4f}rad_pT_0.00e+00GeV_eta_1.00e+00_0/Events/run_01/unweighted_events.root',
            ])
            print('%.4f' % theta_3, W12, W34[0,0])
            theta_7 = np.arctan2(np.hypot(P7[:,1], P7[:,2]), P7[:,3])
            theta_9 = np.arctan2(np.hypot(P9[:,1], P9[:,2]), P9[:,3])
            W = W12 * W34[0,0] * np.ones_like(theta_7)
            theta_7s.append(theta_7); theta_9s.append(theta_9); Ws.append(W)
        theta_7, theta_9, W = map(np.concatenate, (theta_7s, theta_9s, Ws))
        plt.hist2d(theta_7, theta_9, bins=100, weights=W, density=True, norm=mcolors.LogNorm())
        plt.xlabel(r'$\theta_7$ [rad]')
        plt.ylabel(r'$\theta_9$ [rad]')
        cbar = plt.colorbar()
        cbar.set_label(f'{pp}{pm} Events')
        plt.tight_layout()
        plt.savefig(f'{pp}{pm}.png')
        plt.clf()
