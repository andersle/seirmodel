# Copyright (c) 2020, Anders Lervik.
# Distributed under the MIT License. See LICENSE for more info.
"""Methods for plotting."""
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates


def plot_model_and_raw_data(time_date, model, raw, max_model_date=None):
    """Plot the modelled data and the raw data."""
    fig, ax1 = plt.subplots(constrained_layout=True)
    date_fmt = mdates.DateFormatter('%d %b, %Y')
    ax1.xaxis.set_major_formatter(date_fmt)
    if max_model_date is not None:
        time_date = np.array(time_date)
        idx1 = np.where(time_date < max_model_date)[0]
        date = time_date[idx1]
        cases = model['cases'][idx1]
        deaths = model['deaths'][idx1]
    else:
        date = time_date
        cases = model['cases']
        deaths = model['deaths']


    ax1.scatter(raw['Date'], raw['Cases'], s=100, label='Cumulative cases reported')
    ax1.scatter(raw['Date'], raw['Deaths'], s=100, label='Cumulative deaths reported')
    ax1.plot(date, cases, label='Modelled cumulative cases')
    ax1.plot(date, deaths, label='Modelled cumulative deaths')

    ax1.set(ylabel='Number of persons')
    ax1.xaxis.set_tick_params(rotation=30)
    ax1.legend()
    return fig, ax1


def plot_model_evolution(time_date, model):
    """Plot the time evolution of the model."""
    fig, ax1 = plt.subplots(constrained_layout=True)
    ax1.plot(time_date, model['infected'], label='Infected persons')
    ax1.plot(time_date, model['resistant'], label='Resistant persons')
    ax1.plot(time_date, model['deaths'], label='Cumulative deaths')
    ax1.set(ylabel='Number of persons')
    date_fmt = mdates.DateFormatter('%d %b, %Y')
    ax1.xaxis.set_major_formatter(date_fmt)
    ax1.xaxis.set_tick_params(rotation=30)
    ax1.legend()
    return fig, ax1


def plot_model_evolution_item(time_date, model, item):
    """Plot the time evolution of the model."""
    fig, ax1 = plt.subplots(constrained_layout=True)
    ax1.plot(time_date, model[item])
    ax1.set(ylabel=item)
    date_fmt = mdates.DateFormatter('%d %b, %Y')
    ax1.xaxis.set_major_formatter(date_fmt)
    ax1.xaxis.set_tick_params(rotation=30)
    return fig, ax1
