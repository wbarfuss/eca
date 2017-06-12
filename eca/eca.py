import math
from scipy import misc
from statsmodels.distributions.empirical_distribution import ECDF
import pandas as pd
import numpy as np


def binarize(x, thres=0.9, inverted=False):
    upper = 1 if not inverted else 0
    lower = 0 if not inverted else 1
    binarized = x.copy()
    binarized.loc[x > thres] = upper
    binarized.loc[x < thres] = lower
    binarized.loc[x == thres] = upper if inverted else lower
    return binarized


def binarize_quantile(x, quantile=0.9, **kwargs):
    q = x.dropna().quantile(quantile)
    return binarize(x, thres=q, **kwargs)


def precursor_coincidence_rate(seriesA, seriesB, deltaT=0, tau=0, sym=False, **kwagrs):
    ts_end = seriesA.index[-1]
    ts_start = seriesA.index[0]
    a_events = seriesA[seriesA == 1].count()
    precursor_coincidences = 0  # precursor_coincidences
    for date, event in seriesA.iteritems():
        if event == 1:
            # check of occurrence of another event in [i-tau-deltaT;i-tau(+deltaT)]
            start = date - tau - deltaT
            end = date - tau  # don't use this with time series ... + 1
            if not sym:
                end += deltaT
            end = min(ts_end, end)
            start = max(ts_start, start)
            assert (
            deltaT > 0 or len(seriesB[start:end]) == 1), " deltaT > 0 or Series with delta 0 should have only one date"
            if seriesB[start:end].max() == 1:
                precursor_coincidences += 1
    return {'precursor_coincidence_rate': precursor_coincidences / a_events,
            'a_events': a_events,
            'precursor_coincidences': precursor_coincidences}


def trigger_coincidence_rate(seriesA, seriesB, deltaT=0, tau=0, sym=False, **kwagrs):
    b_events = seriesB[seriesB == 1].count()
    trigger_coincidences = 0  # trigger_coincidences
    ts_end = seriesA.index[-1]
    ts_start = seriesA.index[0]
    for date, event in seriesB.iteritems():
        if event == 1:
            start = date + tau
            end = date + tau + deltaT  # + 1
            if sym:
                start -= deltaT
            end = min(ts_end, end)
            start = max(ts_start, start)
            if seriesA[start:end].max() == 1:
                trigger_coincidences += 1
    return {'trigger_coincidence_rate': trigger_coincidences / b_events,
            'b_events': b_events,
            'trigger_coincidences': trigger_coincidences}


def binomial_of(K_p, N_a, N_b, TOL, T, tau):
    return (misc.comb(N=N_a, k=K_p, exact=True)
            * math.pow(1 - math.pow(1 - (TOL / (T - tau)), N_b), K_p)
            * math.pow(math.pow(1 - (TOL / (T - tau)), N_b), N_a - K_p))

def waiting_times_distribution_index(series):
    event_date = series[series == 1].index.to_series()
    next_event_date = event_date.shift(1)
    time_to_next_event = (event_date - next_event_date).shift(-1)
    return time_to_next_event.value_counts() / time_to_next_event.value_counts().sum()

def sample_distr(waiting_times,waiting_times_distr):
    return waiting_times.sample(1, weights=waiting_times_distr.values).values[0]

def create_wt_surrogate(series,waiting_times_distr,waiting_times):
    surrogate = pd.Series(data=np.zeros(len(series)), index=series.index)
    cursor = surrogate.index.min() + sample_distr(waiting_times, waiting_times_distr)
    while cursor < surrogate.index.max():
        surrogate[cursor] = 1
        cursor += sample_distr(waiting_times, waiting_times_distr)
        print(cursor)
    return surrogate


def surrogate_waiting_time_distr_test(seriesA,seriesB,precursor_coincidences, trigger_coincidences,alpha,repititions=10,**kwargs):
    waiting_times_distr_A = waiting_times_distribution_index(seriesA)
    waiting_times_A = pd.Series(waiting_times_distr_A.index)
    waiting_times_distr_B = waiting_times_distribution_index(seriesB)
    waiting_times_B = pd.Series(waiting_times_distr_B.index)
    surrogate_trigger_p = repititions * [0]
    surrogate_precursor_p = repititions * [0]
    for i in range(repititions):
        surrogateA = create_wt_surrogate(seriesA,waiting_times_distr_A,waiting_times_A)
        surrogateB = create_wt_surrogate(seriesB, waiting_times_distr_B, waiting_times_B)
        surrogate_trigger_p[i] = trigger_coincidence_rate(surrogateA,surrogateB,**kwargs)['trigger_coincidences']
        surrogate_precursor_p[i] = precursor_coincidence_rate(surrogateA, surrogateB, **kwargs)['precursor_coincidences']
    p_value_trigger = 1 - ECDF(surrogate_trigger_p)(trigger_coincidences)
    p_value_precursor = 1 - ECDF(surrogate_precursor_p)(precursor_coincidences)
    return  {'waiting_time_precursor_coincidence': p_value_precursor,
            'waiting_time_trigger_coincidence': p_value_trigger,
            'Null_Hypoth_precursor': p_value_precursor >= alpha,
            'Null_Hypoth_trigger': p_value_trigger >= alpha}



def surrogate_shuffle_test(seriesA,seriesB,alpha,repititions=1000):
    wating_times = seriesA[seriesA == 1] - seriesA[seriesA==1].shift(1)

    pass


def poisson_test(a_events, b_events, tau, deltaT, precursor_coincidences, trigger_coincidences, sym,
                 alpha, no_nan_length,
                 **kwgars):
    if not sym:
        # Note: nan values are not considered in analytical significance test
        T = no_nan_length
        TOL = deltaT + 1
    else:
        T = no_nan_length
        TOL = 2 * deltaT + 1
    # Calculation for Precursor Coincidence
    analytic_precursor_coincidence = 0
    analytic_trigger_coincidence = 0
    for k_p in range(precursor_coincidences, a_events + 1):
        analytic_precursor_coincidence += binomial_of(k_p, a_events, b_events, TOL, T, tau)

    for k_p in range(trigger_coincidences, b_events + 1):
        analytic_trigger_coincidence += binomial_of(k_p, b_events, a_events, TOL, T, tau)

    """
    Null hypothesis of this test is: SeriesA and SeriesB represent independent random events.
    """
    return {'analytic_precursor_coincidence': analytic_precursor_coincidence,
            'analytic_trigger_coincidence': analytic_trigger_coincidence,
            'Null_Hypoth_precursor': analytic_precursor_coincidence >= alpha,
            'Null_Hypoth_trigger': analytic_trigger_coincidence >= alpha}


def eca_analysis(seriesA, seriesB, tau=0, deltaT=0, alpha=0.05, sym=False):
    # TODO error_checking
    # if len(seriesA) != len(seriesB):
    #   raise ValueError("Series not of same length")
    input = {'seriesA': seriesA,
             'seriesB': seriesB,
             'tau': tau,
             'deltaT': deltaT,
             'sym': sym,
             'alpha': alpha,
             'no_nan_length': len(seriesA) - seriesA.isnull().sum()}
    p_out = precursor_coincidence_rate(**input)
    t_out = trigger_coincidence_rate(**input)
    test_out = poisson_test(**t_out, **p_out, **input)
    return {**p_out, **t_out, **test_out}

def eca_analysis_suffle(seriesA, seriesB, tau=0, deltaT=0, alpha=0.05, sym=False):
    # TODO error_checking
    # if len(seriesA) != len(seriesB):
    #   raise ValueError("Series not of same length")
    input = {'seriesA': seriesA,
             'seriesB': seriesB,
             'tau': tau,
             'deltaT': deltaT,
             'sym': sym,
             'alpha': alpha,
             'no_nan_length': len(seriesA) - seriesA.isnull().sum()}
    p_out = precursor_coincidence_rate(**input)
    t_out = trigger_coincidence_rate(**input)
    test_out = surrogate_waiting_time_distr_test(**t_out, **p_out, **input)
    return {**p_out, **t_out, **test_out}
