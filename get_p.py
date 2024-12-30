import numpy as np

def get_p(theta_p_lab_exp):
    E_p = 1
    m_p = 0.000511
    m_e = 0.000511
    E = E_p + m_e
    m_com = np.sqrt(m_p**2 + m_e**2 + 2*m_e*E_p)
    gamma = E / m_com
    beta = np.sqrt(1 - 1/gamma**2)
    p_com = np.sqrt((
        m_p**4 + m_e**4 + m_com**4 - 2*m_p**2*m_e**2 - 2*m_e**2*m_com**2 - 2*m_p**2*m_com**2
    ) / (4*m_com**2))
    
    def com_to_lab(p):
        return np.array([
            gamma * (p[0] + beta * p[3]),
            p[1],
            p[2],
            gamma * (p[3] + beta * p[0]),
        ])
    
    def compute_p(theta_p):
        return com_to_lab(np.array([
            np.ones_like(theta_p) * np.sqrt(p_com**2 + m_p**2),
            p_com * np.sin(theta_p),
            np.zeros_like(theta_p),
            p_com * np.cos(theta_p),
        ])), com_to_lab(np.array([
            np.ones_like(theta_p) * np.sqrt(p_com**2 + m_e**2),
            -p_com * np.sin(theta_p),
            np.zeros_like(theta_p),
            -p_com * np.cos(theta_p),
        ]))
    
    theta_p = np.linspace(1e-5 * np.pi, (1 - 1e-5) * np.pi, int(1e5) - 1)
    p_p, p_e = compute_p(theta_p)
    theta_p_lab = np.arctan2(np.hypot(p_p[1], p_p[2]), p_p[0])
    theta_e_lab = np.arctan2(np.hypot(p_e[1], p_e[2]), p_e[0])
    i = np.argmax(theta_p_lab >= theta_p_lab_exp)
    return p_p[:,i], p_e[:,i]
