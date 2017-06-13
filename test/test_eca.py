from unittest import TestCase
import pandas as pd

from eca import eca


class TestBinarize(TestCase):
    """ Test eca to conform with examples from ``CoinCalc -- A new R package for quantifying simultaneities of event series``
    
    https://arxiv.org/abs/1603.05038 by Jonathan F. Siegmund et al.
    
    """

    def setUp(self):
        flower_data_file = "sample_data/CC.Example.Data1.txt"
        self.df = pd.read_csv(flower_data_file, sep=' ', skipinitialspace=True, dtype={0: int})
        self.df = self.df.set_index(pd.PeriodIndex(pd.to_datetime(self.df['YEAR'], format='%Y'), freq='A'))
        del self.df['YEAR']

        self.flowering_day = eca.binarize_quantile(self.df['DOY_Lilac_Flowering'].copy(), quantile=0.1, inverted=True)
        self.temperature = eca.binarize_quantile(self.df['April_Tmean'].copy(), quantile=0.9)

    def test_binarize(self):
        thres = 0.5
        a_expected = pd.Series([0.0, 0.0, 0.0, 0.0, 1.0])
        a_actual = pd.Series([0.1, 0.3, 0.5, 0.5, 0.6])
        self.assertTrue(eca.binarize(a_actual, thres).equals(a_expected))

    def test_binarize_quantile(self):
        temp_event_count_expected = 6
        day_event_count_expected = 6

        temperature = self.temperature
        flowering_day = self.flowering_day

        self.assertEqual(temp_event_count_expected, temperature[temperature == 1].count())
        self.assertEqual(flowering_day[flowering_day == 1].count(), day_event_count_expected)
        self.assertEqual(60, len(temperature))
        self.assertEqual(60, len(flowering_day))

    def test_coincidence_rate(self):
        pcr_expected = 0.5
        tcr_expected = 0.5

        temperature = self.temperature
        flowering_day = self.flowering_day
        pcr_actual = eca.precursor_coincidence_rate(temperature, flowering_day)[
            'precursor_coincidence_rate']
        tgr_actual = eca.trigger_coincidence_rate(temperature, flowering_day)[
            'trigger_coincidence_rate']

        self.assertEqual(pcr_actual, pcr_expected)
        self.assertEqual(tgr_actual, tcr_expected)

    def test_eca_analysis(self):
        out_expected = {'analytic_precursor_coincidence': 0.017775577,
                        'analytic_trigger_coincidence': 0.017775577,
                        'Null_Hypoth_precursor': False,
                        'Null_Hypoth_trigger': False}
        out = eca.eca_analysis(self.flowering_day, self.temperature)
        self.assertEqual(out['Null_Hypoth_precursor'], out_expected['Null_Hypoth_precursor'])
        self.assertEqual(out['Null_Hypoth_trigger'], out_expected['Null_Hypoth_trigger'])
        self.assertAlmostEqual(out['analytic_precursor_coincidence'], out_expected['analytic_precursor_coincidence'])
        self.assertAlmostEqual(out['analytic_trigger_coincidence'], out_expected['analytic_trigger_coincidence'])

    def test_eca_analysis_shuffle(self):
        out = eca.eca_analysis(self.flowering_day, self.temperature,test_method='waiting_time')
        print(out)
        out = eca.eca_analysis(self.flowering_day, self.temperature, test_method='shuffle')
        print(out)
        # We just want to check that this is not throwing any errors
