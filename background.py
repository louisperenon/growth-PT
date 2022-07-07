from astropy.cosmology import FlatLambdaCDM


class Background:
    """
    The class used to set the background cosmology

    So far it only uses the FlatLambdaCDM model from Astropy

    Attributes
    ----------
    inputs : dict
        cosmological parameters required

    Methods
    -------
    Omega_m(redshift)
        Computes the value of Omega_m for redshift

    epsilon(redshift)
        Computes the value of epsilon for redshift
    """

    def __init__(self, inputs):
        """
        Parameters
        ----------
        inputs : dict
            Cosmological parameters required for the model FlatLambdaCDM of
            astropy.cosmology: :math:`$\{ H_0,  \Omega_{m,0} \}$` ,

        """
        self._cosmology = FlatLambdaCDM(H0=inputs["H0"], Om0=inputs["Om0"])

    def Omega_m(self, redshift):
        """Computes the values of Omega_m for the inputs redshifts

        Parameters
        ----------
        redshift: (nz,) ndarray_like
            redshifts at which to evaluate Omega_m

        Returns
        -------
        Omega_m: (nz,) ndarray_like
            Omega_m at the chosen redshifts
        """

        return self._cosmology.Om(redshift)

    def epsilon(self, redshift):
        """Computes the values of epsilon for the inputs redshifts

        Parameters
        ----------
        redshift: (nz,) ndarray_like
            redshifts at which to evaluate epsilon

        Returns
        -------
        epsilon: (nz,) ndarray_like
            epsilon at the chosen redshifts
        """

        return 3 / 2 * self.Omega_m(redshift)
