# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""Methods defining the SEIR model."""
import datetime
import numpy as np
from scipy.integrate import solve_ivp


def exp_beta(t, beta_0, kappa, tau):
    """Evaluate the transmission rate as a function of time.

    Here, the transmission rate changes exponatially, due to control
    measures introduced.

    Parameters
    ----------
    t : float
        The point in time we are evaluating for.
    beta_0 : float
        Transmission rate.
    kappa : float
        Exponential factor for time evolution of beta.
    tau : float
        The time at which control measures are introduced.

    Returns
    -------
    beta : float
        The beta factor evaulated at the given time.

    """
    if t <= tau:
        return beta_0
    return beta_0 * np.exp(-kappa * (t - tau))


def seir_model(t, y, gamma, sigma, beta_0, fatality, kappa, tau,
               beta_model=exp_beta):
    """Calculate the derivative for the SEIR model.

    The possible states are susceptible (S), exposed (E),
    infected (I), and resistant (R). The cumulative number of cases
    (C) and deaths (D) are also calculated.

    Parameters
    ----------
    t : float
        The point in time we are evaluating for.
    y : float
        The current values,
        `y(t) = [S(t), E(t), I(t), R(t), D(t), C(t)]`.
    gamma : float
        Inverse average duration of infectiousness.
    sigma : float
        Inverse average duration of incubation.
    beta_0 : float
        Transmission rate.
    fatality : float
        The fatality rate.
    kappa : float
        Exponential factor for time evolution of beta.
    tau : float
        The time at which control measures are introduced.
    beta_model : callable, optional
        The model for the time evolution of the transmission rate.

    Returns
    -------
    dydt : list of floats
        The derivatives for the model.

    """
    s_t = y[0]
    e_t = y[1]
    i_t = y[2]
    r_t = y[3]
    n_t = s_t + e_t + i_t + r_t

    beta = beta_model(t, beta_0, kappa, tau)

    ds_dt = - beta * s_t * i_t / n_t
    de_dt = (beta * s_t * i_t / n_t) - sigma * e_t
    di_dt = sigma * e_t - gamma * i_t
    dr_dt = (1.0 - fatality) * gamma * i_t
    dd_dt = fatality * gamma * i_t
    dc_dt = sigma * e_t
    dy_dt = [ds_dt, de_dt, di_dt, dr_dt, dd_dt, dc_dt]
    return dy_dt


def run_model(time_span, parameters, initial_values, npoints=100):
    """Run the time integration of the model.

    Parameters
    ----------
    time_span : list/tuple of numbers
        The minimum and maximum time to run the integration for.
    parameters : dict
        The parameters for the model.
    initial_values : list of numbers
        The initial values for the variables (S, E, I, R, D, C)
    npoints : integer, optional
        The number of time points to evaluate the model for.

    """
    t_eval = np.linspace(min(time_span), max(time_span), npoints)
    # Convert between input parameters and the parameters for the model:
    arguments = [
        1.0 / parameters['infectiousness'],
        1.0 / parameters['incubation'],
        parameters['beta_0'],
        parameters['fatality'],
        parameters['kappa'],
        parameters['tau'],
    ]

    result = solve_ivp(seir_model, time_span, initial_values, args=arguments,
                       t_eval=t_eval)

    variables = ['susceptible', 'exposed', 'infected',
                 'resistant', 'deaths', 'cases']
    time = result['t']
    model = {}
    for i, key in enumerate(variables):
        model[key] = result['y'][i, :]

    zero = parameters['time_zero']
    time_date = [
        zero + datetime.timedelta(seconds=int(i * 3600 * 24)) for i in time
    ]
    return result, time, time_date, model
