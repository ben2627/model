from scipy import integrate


def model(t, S):
    x, y = S

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


def solve_from(x_curr, y_curr, t_elapsed):
    t_span = (0, 0 + t_elapsed)
    t_eval = (0, 0 + t_elapsed)
    result = integrate.solve_ivp(
        model, t_span=t_span, t_eval=t_eval, y0=(x_curr, y_curr))
    return (result.t[1], result.y[0][1], result.y[1][1])
