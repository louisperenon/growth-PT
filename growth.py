import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from background import Background as background_class


class Growth:
    # """ Growth function for Dark Energy and Modified Gravity.

    #         This class solves the differential equations for the growth functions in
    #         a dark energy or modified gravity theory.

    #         Parameters
    #         ----------
    #         cosmo_dic: dictionary
    #             Cosmology dictionary with cosmological parameters and other
    #             background quantities. Required: Hubble parameter
    #             and the matter density as a function of redshift :math:`$\{ H, ombh2, omch2 \}$`.
    #         redshift: (nz,) ndarray_like
    #             Redshift at which evaluate the growth functions.
    #         parameter_de: dictionary
    #             Dictionary with the modified or dark energy parameter.
    #             Default is {'linear': {'mu_phi':1}}.
    #         solve_ivp_method: Tuple ??
    #             Method to solve the differential equations, initial value problem:
    #             solver (function), initial conditions (float), redshift_resolution (integer).
    #             Default is (None).

    #         Returns
    #         -------
    #         linear_growth: (nz,) ndarray_like
    #             Linear growth function for the corresponding theory at
    #             a given redshift.

    #     """

    def __init__(
        self,
        background,
        model="lcdm",
        order="linear",
        initial_conditions="EdS",
        solving_redshifts=None,
        solving_methods=None,
        interpolation_method="linear",
        interpolation_fill_value="extrapolate",
    ):

        self._set_background(background)
        self._set_order(order)
        self._set_model(model)

        # Setting solving redshifts for solve_ivp
        self.redshift_initial = 100
        self.redshift_final = 0
        self.redshift_resolution = 1000
        self.redshift_span = np.linspace(
            self.redshift_initial, self.redshift_final, self.redshift_resolution
        )
        self._set_solving_redshifts(solving_redshifts)

        # Setting solving methods for solve_ivp
        self.solve_ivp_method_linear = "RK45"
        self.solve_ivp_method_quadratic_A = "RK45"
        self.solve_ivp_method_quadratic_B = "RK45"
        self._set_solving_methods(solving_methods)

        # Setting initial conditions
        self.D_ini = 1.0
        self.D_prime_ini = -1.0 / (1 + self.redshift_initial)
        self.D_A_ini = 0
        self.D_A_prime_ini = 0
        self.D_B_ini = 0
        self.D_B_prime_ini = 0
        self._set_initial_conditions(initial_conditions)

        # Setting the interpolation method for the solutions
        self.interpolation_method = interpolation_method

        # Setting whether extrapolations are allowed
        self.interpolation_fill_value = interpolation_fill_value

        # For good practice: defining default growth solutions attributes
        self.interpolated_factor_D = None
        self.interpolated_factor_D_prime = None
        self.interpolated_linear = None
        self.interpolated_linear_prime = None
        self.interpolated_quadratic_A = None
        self.interpolated_quadratic_A_prime = None
        self.interpolated_quadratic_B = None
        self.interpolated_quadratic_B_prime = None

    def _set_background(self, background):
        if isinstance(background, background_class):
            self.background = background
        else:
            raise ValueError(
                "The 'background' parameter must be a instance \
                            of the background.Background class"
            )

    def _set_model(self, model):
        if model == "lcdm":
            import lcdm as model
        else:
            raise ValueError("Model can only be 'lcdm'")
        self.model = model

    def _set_order(self, order):
        if order in ["linear", "quadratic", "cubic"]:
            self.order = order
        else:
            raise ValueError(
                "The 'order' parameter must be either the string: \n \
                            'linear' to compute the linear growth (default) \n \
                            'quadratic' to compute the quadratic growth \n \
                            'cubic' to compute the cubic growth"
            )

    def _set_solving_redshifts(self, dictionary):
        if not dictionary == None and isinstance(dictionary, dict):
            if dictionary.keys() in [
                "redshift_initial",
                "redshift_final",
                "redshift_resolution",
                "redshift_span",
            ]:

                if "redshift_initial" in dictionary.keys():
                    self.redshift_initial = dictionary["redshift_initial"]
                if "redshift_final" in dictionary.keys():
                    self.redshift_final = dictionary["redshift_final"]
                if "redshift_resolution" in dictionary.keys():
                    self.redshift_resolution = dictionary["redshift_resolution"]
                self.redshift_span = np.linspace(
                    self.redshift_initial,
                    self.redshift_final,
                    self.redshift_resolution,
                )
                if "redshift_span" in dictionary.keys():
                    self.redshift_span = dictionary["redshift_span"]
                    self.redshift_initial = self.redshift_span[0]
                    self.redshift_final = self.redshift_span[-1]
            else:
                raise ValueError(
                    "solving_redshifts must be a dictionary with keys among \
                    'redshift_initial', 'redshift_final', 'redshift_resolution'\
                    and 'redshift_span' \
                    if 'redshift_span' is not given then the redshift vector to solve on\
                    will be linear between redshift_initial to redshift_final divided into \
                    steps equal to 'redshift_resolution' \
                    if 'redshift_span' is given then 'redshift_initial' and 'redshift_final' \
                    will be deduced automatically "
                )

    def _set_solving_methods(self, dictionary):
        if not dictionary == None and isinstance(dictionary, dict):
            if dictionary.keys() in ["linear", "quadratic_A", "quadratic_B"]:
                self.solve_ivp_method_linear = dictionary["linear"]
                self.solve_ivp_method_quadratic_A = dictionary["quadratic_A"]
                self.solve_ivp_method_quadratic_B = dictionary["quadratic_B"]
            else:
                raise ValueError(
                    "solve_ivp_methods must be a dictionary with \
                    keys among 'linear', 'quadratic_A', 'quadratic_B' and their values \
                    must be those supported by the function \
                    scipy.integrate.solve_ivp"
                )

    def _set_initial_conditions(self, initial_conditions):
        if isinstance(initial_conditions, dict):
            # Case tests given the order chosen
            if (
                self.order == "quadratic"
                and "D_A" not in initial_conditions.keys()
            ):
                raise ValueError("Missing the D_A initial condition")
            if (
                self.order == "quadratic"
                and "D_A_prime" not in initial_conditions.keys()
            ):
                raise ValueError("Missing the D_A_prime initial condition")
            if (
                self.order == "quadratic"
                and "D_B" not in initial_conditions.keys()
            ):
                raise ValueError("Missing the D_B initial condition")
            if (
                self.order == "quadratic"
                and "D_B_prime" not in initial_conditions.keys()
            ):
                raise ValueError("Missing the D_B_prime initial condition")

            # Updating accordingly
            if "D" in initial_conditions.keys():
                self.D_ini = initial_conditions["D"]
            if "D_prime" in initial_conditions.keys():
                self.D_prime_ini = initial_conditions["D_prime"]
            if "D_A" in initial_conditions.keys():
                self.D_A_ini = initial_conditions["D_A"]
            if "D_A_prime" in initial_conditions.keys():
                self.D_A_prime_ini = initial_conditions["D_A_prime"]
            if "D_B" in initial_conditions.keys():
                self.D_A_ini = initial_conditions["D_B"]
            if "D_B_prime" in initial_conditions.keys():
                self.D_A_prime_ini = initial_conditions["D_B_prime"]

        elif initial_conditions == "EdS":
            # To set these IC, the linear needs to have been computed, see _run_computations
            self.do_EdS = True

        else:
            raise ValueError(
                "The initial conditions input must either be \
                             ''EdS'' or a dictionary containing the values \
                             of the initial condition you are requiring"
            )

    def _set_initial_condition_EdS(
        self, redshift, growth_function, growth_function_prime
    ):
        gz0 = growth_function(0)
        g = growth_function(redshift) / gz0
        dg = growth_function_prime(redshift)

        self.D_A_ini = 3 * np.square(g) / 7.0
        self.D_A_prime_ini = 6 * g * dg / 7.0
        self.D_B_ini = 2 * np.square(g) / 7.0
        self.D_B_prime_ini = 4 * g * dg / 7.0
        self.D_D_ini = 2 * np.power(g, 3) / 21.0
        self.D_D_prime_ini = 6 * np.square(g) * dg / 21.0
        self.D_E_ini = 4 * np.power(g, 3) / 63.0
        self.D_E_prime_ini = 12 * np.square(g) * dg / 63.0
        self.D_F_ini = np.power(g, 3) / 14.0
        self.D_F_prime_ini = 3 * np.square(g) * dg / 14.0
        self.D_G_ini = np.power(g, 3) / 21.0
        self.D_G_prime_ini = 3 * np.square(g) * dg / 21.0
        self.D_J_ini = np.power(g, 3) / 9.0
        self.D_J_prime_ini = 3 * np.square(g) * dg / 9.0

    def _compute_factor_D(self):
        self.interpolated_factor_D = interp1d(
            self.redshift_span,
            self.model.factor_D(self.redshift_span, self.background),
            kind=self.interpolation_method,
            fill_value=self.interpolation_fill_value,
        )

    def _compute_factor_D_prime(self):
        self.interpolated_factor_D_prime = interp1d(
            self.redshift_span,
            self.model.factor_D_prime(self.redshift_span, self.background),
            kind=self.interpolation_method,
            fill_value=self.interpolation_fill_value,
        )

    def _solve_ivp(self, D_ini, D_prime_ini, coupled_eq, method):
        # generic function to run solve_ivp given a coupled_eq function
        sol = solve_ivp(
            coupled_eq,
            [self.redshift_initial, self.redshift_final],
            [D_ini, D_prime_ini],
            t_eval=self.redshift_span,
            method=method,
        )
        return sol

    def _interpolate_solution(self, sol):
        vsol = interp1d(
            sol.t,
            sol.y[0],
            kind=self.interpolation_method,
            fill_value=self.interpolation_fill_value,
        )
        vsol_prime = interp1d(
            sol.t,
            sol.y[1],
            kind=self.interpolation_method,
            fill_value=self.interpolation_fill_value,
        )
        return vsol, vsol_prime

    # Computes the linear growth solutions
    def _compute_linear(self):

        # Coupled differential equation to feed to solve_ivp (has to be of inputs (x,y))
        def coupled_eqs(redshift, point):
            res = self.model.growth_coupled_equations_linear(
                self.interpolated_factor_D,
                self.interpolated_factor_D_prime,
                redshift,
                point,
            )
            return res

        # Running solve_ivp and processing the results
        solution = self._solve_ivp(
            self.D_ini,
            self.D_prime_ini,
            coupled_eqs,
            method=self.solve_ivp_method_linear,
        )

        # Storing the interpolated solutions
        solution = self._interpolate_solution(solution)
        self.interpolated_linear = solution[0]
        self.interpolated_linear_prime = solution[1]

    def _compute_quadratic(self):

        # Equation A
        def coupled_eqs(redshift, point):
            res = self.model.growth_coupled_equations_quadratic_A(
                self.interpolated_factor_D,
                self.interpolated_factor_D_prime,
                self.interpolated_linear,
                self.interpolated_linear_prime,
                redshift,
                point,
            )
            return res

        solution = self._solve_ivp(
            self.D_A_ini,
            self.D_A_prime_ini,
            coupled_eqs,
            method=self.solve_ivp_method_quadratic_A,
        )
        solution = self._interpolate_solution(solution)
        self.interpolated_quadratic_A = solution[0]
        self.interpolated_quadratic_A_prime = solution[1]

        # Equation B
        def coupled_eqs(redshift, point):
            res = self.model.growth_coupled_equations_quadratic_B(
                self.interpolated_factor_D,
                self.interpolated_factor_D_prime,
                self.interpolated_linear,
                self.interpolated_linear_prime,
                redshift,
                point,
            )
            return res

        solution = self._solve_ivp(
            self.D_B_ini,
            self.D_B_prime_ini,
            coupled_eqs,
            method=self.solve_ivp_method_quadratic_B,
        )
        solution = self._interpolate_solution(solution)
        self.interpolated_quadratic_B = solution[0]
        self.interpolated_quadratic_B_prime = solution[1]

    def _compute_cubic(self):
        # To be implemented
        pass

    def compute(self):

        # Computing the interpolations of the factors for the growth equations
        self._compute_factor_D()
        self._compute_factor_D_prime()

        # Computing the linear order (default)
        self._compute_linear()

        # If the user chose EdS initial conditions they get updated
        if self.do_EdS:
            self._set_initial_condition_EdS(
                self.redshift_final,
                self.interpolated_linear,
                self.interpolated_linear_prime,
            )

        # Running methods if higher order is requested
        if self.order == "quadratic":
            self._compute_quadratic()

        elif self.order == "cubic":
            self._compute_quadratic()
            self._compute_cubic()

    def linear_growth_factor(self, redshift):
        res = self.interpolated_linear_prime(redshift)
        res /= self.interpolated_linear(redshift)
        res *= -(1 + redshift)
        return res

    def linear_growth_factor_linder(self, redshift, gamma=0.55):
        return np.power(self.background.Omega_m(redshift), gamma)

    def linear_solution(self, redshift):
        return self.interpolated_linear(
            redshift
        ), self.interpolated_linear_prime(redshift)

    def quadratic_solution_A(self, redshift):
        if self.order in ["quadratic", "cubic"]:
            return self.interpolated_quadratic_A(
                redshift
            ), self.interpolated_quadratic_A_prime(redshift)
        else:
            raise ValueError("The 'order' chosen must be at least quadratic.")

    def quadratic_solution_B(self, redshift):
        if self.order in ["quadratic", "cubic"]:
            return self.interpolated_quadratic_B(
                redshift
            ), self.interpolated_quadratic_B_prime(redshift)
        else:
            raise ValueError("The 'order' chosen must be at least quadratic.")
