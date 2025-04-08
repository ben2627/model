import numpy as np
from scipy import integrate


# t = time
# S = model values
# i = number of coupled oscillators
# boundary_matrix = normalised neighbour boundary pixel matrix


def model(t, S, i, boundary_matrix):
    x_arr = S[:i]
    y_arr = S[i:]

    fx_arr = []
    gx_arr = []

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

    coupling_terms = []

    for m in range(i):
        coupling_terms.append(sum(
            boundary_matrix[m, j] * (x_arr[j] - x_arr[m]) for j in range(i) if j != m))

    for n in range(i):
        xi = x_arr[n]
        yi = y_arr[n]
        coupling_term = coupling_terms[n]

        xiSq = pow(xi, 2)

        SX = p_vM2 * (xiSq / (pow(p_K2, 2) + xiSq))

        p_Z0Sq = pow(p_Z0, 2)
        p_kCaASq = pow(p_kCaA, 2)

        CX = 4 * p_vM3 * (p_Z0Sq / (pow(p_kIP3, 2) + p_Z0Sq)) * \
            ((p_kCaASq * xiSq) / ((p_kCaASq + xiSq) * (pow(p_kCaI, 2) + xiSq)))

        CXmul = CX * (yi - xi)
        p_KFmul = p_KF * (yi - xi)

        fxyi = p_vin - (p_Kout * xi) - SX + CXmul + p_KFmul + coupling_term
        gxyi = SX - CXmul - p_KFmul

        fx_arr.append(fxyi)
        gx_arr.append(gxyi)

    return np.concatenate((fx_arr, gx_arr))


def solve_from(x_arr_curr, y_arr_curr, t_elapsed_seconds, i, boundary_matrix):
    t_span = (0, 0 + t_elapsed_seconds)
    t_eval = (0, 0 + t_elapsed_seconds)

    S0 = np.concatenate((x_arr_curr, y_arr_curr))

    result = integrate.solve_ivp(
        model, t_span=t_span, t_eval=t_eval, y0=S0, args=(i, boundary_matrix))

    x_arr = result.y[:i, -1]
    y_arr = result.y[i:, -1]

    return (x_arr, y_arr)
