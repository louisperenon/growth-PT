import numpy as np


def factor_D(redshift, cosmology):
    """Computes the -1*coefficient in front of the
    function in the differential equations"""
    Omega_m = cosmology.Omega_m(redshift)
    res = 3 / 2 * Omega_m / np.square(1 + redshift)
    return res


def factor_D_prime(redshift, cosmology):
    """Computes the -1*coefficient in front of the
    first derivative in the differential equations"""
    epsilon = cosmology.epsilon(redshift)
    res = (1 - epsilon) / (1 + redshift)
    return res


def growth_coupled_equations_generic(factor_D, factor_D_prime, source, point):
    """
    Computes the coupled differential
    equation system corresponding to the linear equation:
    D'' = source - factor_D' * D' + factor_D*D_ini
    D'  = D'
    """
    res = [point[1], source + factor_D_prime * point[1] + factor_D * point[0]]
    return res


def growth_coupled_equations_linear(factor_D, factor_D_prime, redshift, point):
    res = growth_coupled_equations_generic(
        factor_D(redshift),
        factor_D_prime(redshift),
        0,
        point,
    )
    return res


def growth_coupled_equations_quadratic_A(
    factor_D, factor_D_prime, linear_sol, linear_sol_prime, redshift, point
):
    source = factor_D(redshift) * np.square(linear_sol(redshift))
    # even though linear_sol_prime is not used I am putting it in case it could be used in the case of other models
    res = growth_coupled_equations_generic(
        factor_D(redshift),
        factor_D_prime(redshift),
        source,
        point,
    )
    return res


def growth_coupled_equations_quadratic_B(
    factor_D, factor_D_prime, linear_sol, linear_sol_prime, redshift, point
):
    source = np.square(linear_sol_prime(redshift))
    # even though linear_sol is not used I am putting it in case it could be used in the case of other models
    res = growth_coupled_equations_generic(
        factor_D(redshift),
        factor_D_prime(redshift),
        source,
        point,
    )
    return res
