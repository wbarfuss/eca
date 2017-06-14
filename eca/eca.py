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
    precursor_coincidences = 0
    for date, event in seriesA.iteritems():
        if event == 1:
            # check occurrence of another event in [i-tau-deltaT;i-tau(+deltaT)]
            start = date - tau - deltaT
            end = date - tau
            if not sym:
                end += deltaT
            end = min(ts_end, end)
            start = max(ts_start, start)
            if seriesB[start:end].max() == 1:
                precursor_coincidences += 1
    pcr = 0 if a_events == 0 else precursor_coincidences / a_events
    return {'precursor_coincidence_rate': pcr,
            'a_events': a_events,
            'precursor_coincidences': precursor_coincidences}


def aggregated_precursor_coincidence_rate(seriesAs, seriesBs, deltaT=0, tau=0, sym=False, **kwagrs):
    individual_rates = [precursor_coincidence_rate(seriesA, seriesB, deltaT, tau, sym)
                            for (seriesA, seriesB) in zip(seriesAs, seriesBs)]
    precursor_coincidences = sum([r["precursor_coincidences"] for r in individual_rates])
    a_events = sum([r["a_events"] for r in individual_rates])
    pcr = 0 if a_events == 0 else precursor_coincidences / a_events
    return {'precursor_coincidence_rate': pcr,
            'a_events': a_events,
            'precursor_coincidences': precursor_coincidences}


def roll_in(array, shift):
    return array if shift == 0 else np.concatenate((np.zeros(shift), array[:-shift]))


def trigger_coincidence_rate(seriesA, seriesB, deltaT=0, tau=0, sym=False, **kwagrs):
    b_events = seriesB[seriesB == 1].count()
    trigger_coincidences = 0  # trigger_coincidences
    shifted_values = [(seriesB.values == roll_in(seriesA, shift)) for shift in range(0 + tau, tau + deltaT + 1)]
    trigger_coincidences = len(seriesB[np.all([seriesB.values == 1, np.any(shifted_values, axis=0)], axis=0)])
    tcr = 0 if b_events == 0 else trigger_coincidences / b_events
    return {'trigger_coincidence_rate': tcr,
            'b_events': b_events,
            'trigger_coincidences': trigger_coincidences}


def aggregated_trigger_coincidence_rate(seriesAs, seriesBs, deltaT=0, tau=0, sym=False, **kwagrs):
    individual_rates = [trigger_coincidence_rate(seriesA, seriesB, deltaT, tau, sym)
                            for (seriesA, seriesB) in zip(seriesAs, seriesBs)]
    trigger_coincidences = sum([r["trigger_coincidences"] for r in individual_rates])
    b_events = sum([r["b_events"] for r in individual_rates])
    tcr = 0 if b_events == 0 else trigger_coincidences / b_events
    return {'trigger_coincidence_rate': tcr,
            'b_events': b_events,
            'trigger_coincidences': trigger_coincidences}


def binomial_of(K_p, N_a, N_b, TOL, T, tau):
    return (misc.comb(N=N_a, k=K_p, exact=True)
            * math.pow(1 - math.pow(1 - (TOL / (T - tau)), N_b), K_p)
            * math.pow(math.pow(1 - (TOL / (T - tau)), N_b), N_a - K_p))


def waiting_time_distribution_of(series):
    event_date = series[series == 1].index.to_series()
    next_event_date = event_date.shift(1)
    time_to_next_event = (event_date - next_event_date).shift(-1)
    return time_to_next_event.value_counts() / time_to_next_event.value_counts().sum()


def sample_wating_time(waiting_times, waiting_times_distr):
    return waiting_times.sample(1, weights=waiting_times_distr.values).values[0]


def create_wt_surrogate(series, waiting_times_distr, waiting_times):
    surrogate = pd.Series(data=np.zeros(len(series)), index=series.index)
    cursor = surrogate.index.min() + sample_wating_time(waiting_times, waiting_times_distr)
    while cursor < surrogate.index.max():
        surrogate[cursor] = 1
        cursor += sample_wating_time(waiting_times, waiting_times_distr)
    return surrogate


def surrogate_waiting_time_distr_test(seriesA, seriesB, precursor_coincidences, trigger_coincidences, alpha,
                                      repititions=100, **kwargs):
    waiting_times_distr_A = waiting_time_distribution_of(seriesA)
    waiting_times_A = pd.Series(waiting_times_distr_A.index)
    waiting_times_distr_B = waiting_time_distribution_of(seriesB)
    waiting_times_B = pd.Series(waiting_times_distr_B.index)
    surrogate_trigger_p = repititions * [0]
    surrogate_precursor_p = repititions * [0]
    for i in range(repititions):
        surrogateA = create_wt_surrogate(seriesA, waiting_times_distr_A, waiting_times_A)
        surrogateB = create_wt_surrogate(seriesB, waiting_times_distr_B, waiting_times_B)
        surrogate_trigger_p[i] = trigger_coincidence_rate(surrogateA, surrogateB, **kwargs)['trigger_coincidences']
        surrogate_precursor_p[i] = precursor_coincidence_rate(surrogateA, surrogateB, **kwargs)[
            'precursor_coincidences']
    p_value_trigger = 1 - ECDF(surrogate_trigger_p)(trigger_coincidences)
    p_value_precursor = 1 - ECDF(surrogate_precursor_p)(precursor_coincidences)
    return {'waiting_time_precursor_coincidence': p_value_precursor,
            'waiting_time_trigger_coincidence': p_value_trigger,
            'Null_Hypoth_precursor': p_value_precursor >= alpha,
            'Null_Hypoth_trigger': p_value_trigger >= alpha}


def create_shuffled_surrogate(series):
    shuffled = np.random.choice(series, len(series))
    return pd.Series(shuffled, index=series.index)


def surrogate_shuffle_test(seriesA, seriesB, precursor_coincidences, trigger_coincidences, alpha, repititions=1000,
                           **kwargs):
    surrogate_trigger_p = repititions * [0]
    surrogate_precursor_p = repititions * [0]
    for i in range(repititions):
        surrogateA = create_shuffled_surrogate(seriesA)
        surrogateB = create_shuffled_surrogate(seriesB)
        surrogate_trigger_p[i] = trigger_coincidence_rate(surrogateA, surrogateB, **kwargs)['trigger_coincidences']
        surrogate_precursor_p[i] = precursor_coincidence_rate(surrogateA, surrogateB, **kwargs)[
            'precursor_coincidences']
    p_value_trigger = 1 - ECDF(surrogate_trigger_p)(trigger_coincidences)
    p_value_precursor = 1 - ECDF(surrogate_precursor_p)(precursor_coincidences)
    return {'shuffeled_precursor_coincidence': p_value_precursor,
            'suhffeled_trigger_coincidence': p_value_trigger,
            'Null_Hypoth_precursor': p_value_precursor >= alpha,
            'Null_Hypoth_trigger': p_value_trigger >= alpha}


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


test_methods = {'poisson': poisson_test,
                'shuffle': surrogate_shuffle_test,
                'waiting_time': surrogate_waiting_time_distr_test}


def eca_analysis(seriesA, seriesB, tau=0, deltaT=0, alpha=0.05, sym=False, test_method='poisson'):
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
    test_out = test_methods[test_method](**t_out, **p_out, **input)
    return {**p_out, **t_out, **test_out}


def aggregated_eca_analysis(seriesAs, seriesBs, tau=0, deltaT=0, alpha=0.05, sym=False, test_method='poisson'):
    input = {'seriesAs': seriesAs,
             'seriesBs': seriesBs,
             'tau': tau,
             'deltaT': deltaT,
             'sym': sym,
             'alpha': alpha,
             }
             #'no_nan_length': len(seriesA) - seriesA.isnull().sum()}
    p_out = aggregated_precursor_coincidence_rate(**input)
    t_out = aggregated_trigger_coincidence_rate(**input)
    #test_out = test_methods[test_method](**t_out, **p_out, **input)
    return {**p_out, **t_out} #, **test_out}
