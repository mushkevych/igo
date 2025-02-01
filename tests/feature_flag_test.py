import os
import unittest

from core.feature_flag import FeatureFlagMgr


class FeatureFlagMgrTest(unittest.TestCase):
    def tearDown(self):
        os.environ.pop('feature_flag_alpha_feature', None)
        os.environ.pop('feature_flag_beta_feature', None)
        os.environ.pop('feature_flag_experiment', None)

    def setUp(self):
        os.environ['feature_flag_alpha_feature'] = 'True'
        os.environ['feature_flag_beta_feature'] = 'false'
        os.environ['feature_flag_experiment'] = '1'

    def test_setter_getter(self):
        feature_flag = FeatureFlagMgr()
        feature_flag.start()

        # consider that "feature_flag_" prefix is discarded by the FeatureFlagMgr
        self.assertEqual(feature_flag.is_enabled('alpha_feature'), True)
        self.assertEqual(feature_flag.is_enabled('beta_feature'), False)
        self.assertEqual(feature_flag.is_enabled('experiment'), True)
        self.assertEqual(feature_flag.is_enabled('non_existent'), False)


if __name__ == '__main__':
    unittest.main()
