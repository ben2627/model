import numpy as np
from scipy import integrate


def model(t, Z):
    x, y = Z

    p_vin = 3
    p_Kout = 16.5
    p_vM2 = 12.5
    p_vM3 = 1562.5
    p_K2 = 0.1
    p_kCaA = 0.9
    p_kCaI = 1.2
    p_kIP3 = 1.65
    p_KF = 0.025
    p_Z0 = 0.6

    xSq = pow(x, 2)

    SX = p_vM2 * (xSq / (pow(p_K2, 2) + xSq))

    p_Z0Sq = pow(p_Z0, 2)
    p_kCaASq = pow(p_kCaA, 2)

    CX = 4 * p_vM3 * (p_Z0Sq / (pow(p_kIP3, 2) + p_Z0Sq)) * \
        ((p_kCaASq * xSq) / ((p_kCaASq + xSq) * (pow(p_kCaI, 2) + xSq)))

    CXmul = CX * (y - x)
    p_KFmul = p_KF * (y - x)

    fxy = p_vin - (p_Kout * x) - SX + CXmul + p_KFmul
    gxy = SX - CXmul - p_KFmul

    return fxy, gxy


X0 = 0.1
Y0 = 1
Z0 = (X0, Y0)

t_span = (0, 10)
t_eval = np.linspace(0, 10, 20000)

results = integrate.solve_ivp(
    model, t_span=t_span, t_eval=t_eval, y0=Z0)