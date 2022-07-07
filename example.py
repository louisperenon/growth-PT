import numpy as np

from background import Background
from growth import Growth


### General inputs
z = np.linspace(0, 5, 4)

###################################
### Testing the cosmology class ###
###################################
inputs_cosmology = {
    "H0": 67,
    "Om0": 0.3,
}
bg = Background(inputs_cosmology)
print("> Omega_m(z) \t =", bg.Omega_m(z))
print("> epsilon(z) \t =", bg.epsilon(z))

################################
### Testing the growth class ###
################################
growth = Growth(bg, order="quadratic")
growth.compute()
print("> f lin(z) \t =",  growth.linear_growth_factor(z))
print("> f Linder(z) \t =", growth.linear_growth_factor_linder(z, gamma=0.55))
print("> D lin(z) \t =",  growth.linear_solution(z))
print("> Quad A (z) \t =", growth.quadratic_solution_A(z))
print("> Quad B (z) \t =", growth.quadratic_solution_B(z))