import unittest
import tensorflow as tf

from estimators.layers.dca_layer import DCALayer
from estimators.configs.t7_dca_config import DCA_CONFIG

from estimators.layers.dcl_layer import DCLLayer
from estimators.configs.t9_dcl_config import DCL_CONFIG

from estimators.layers.oibd_layer import OIBDLayer
from estimators.configs.t12_oibd_config import OIBD_CONFIG

from estimators.layers.pallo_layer import PALLOLayer
from estimators.configs.t23_pallo_config import PALLO_CONFIG

from estimators.layers.dll_layer import DLLLayer
from estimators.configs.t8_dll_config import DLL_CONFIG
from estimators.layers.drr_layer import DRRLayer
from estimators.configs.t11_drr_config import DRR_CONFIG
from estimators.layers.edepbu_layer import EDEPBULayer
from estimators.configs.t4_edepbu_config import EDEPBU_CONFIG
from estimators.layers.edepma_layer import EDEPMALayer
from estimators.configs.t1_edepma_config import EDEPMA_CONFIG
from estimators.layers.fe_layer import FELayer
from estimators.configs.t14_fe_config import FE_CONFIG
from estimators.layers.fi_layer import FILayer
from estimators.configs.t13_fi_config import FI_CONFIG
from estimators.layers.ibu_layer import IBULayer
from estimators.configs.t5_ibu_config import IBU_CONFIG
from estimators.layers.rot_layer import ROTLayer
from estimators.configs.t24_rot_config import ROT_CONFIG
from estimators.layers.tl_layer import TLLayer
from estimators.configs.t20_tl_config import TL_CONFIG
from estimators.layers.zpf_layer import ZPFLayer
from estimators.configs.t16_zpf_config import ZPF_CONFIG

from estimators.layers.ima_layer import IMALayer
from estimators.configs.t3_ima_config import IMA_CONFIG
from estimators.layers.tdepbu_layer import TDEPBULayer
from estimators.configs.t22_tdepbu_config import TDEPBU_CONFIG
from estimators.layers.tdepma_layer import TDEPMALayer
from estimators.configs.t15_tdepma_config import TDEPMA_CONFIG

from estimators.layers.dour_layer import DOURLayer
from estimators.configs.t17_dour_config import DOUR_CONFIG
from estimators.layers.oa_layer import OALayer
from estimators.configs.t19_oa_config import OA_CONFIG
from estimators.layers.ota_layer import OTALayer
from estimators.configs.t21_ota_config import OTA_CONFIG
from estimators.layers.sma_layer import SMALayer
from estimators.configs.t2_sma_config import SMA_CONFIG

from estimators.layers.dofa_layer import DOFALayer
from estimators.configs.t6_dofa_config import DOFA_CONFIG
from estimators.layers.dsc_layer import DSCLayer
from estimators.configs.t10_dsc_config import DSC_CONFIG
from estimators.layers.gc_layer import GCLayer
from estimators.configs.t18_gc_config import GC_CONFIG

class TestAllLayers(unittest.TestCase):
    
    def _test_single_hs_layer(self, layer_class, config):
        """Tests a single layer that uses HSLayer."""
        layer = layer_class()
        dummy_input = {name: tf.zeros((13, 1)) for name in layer.feature_names}
        _ = layer(dummy_input)
        layer.load_weights_from_cfg(config)

        # Test output shape
        prediction = layer(dummy_input)
        self.assertEqual(prediction.shape, (13, 1))

        # Test weights loading
        loaded_weights = layer.get_weights()
        self.assertAlmostEqual(loaded_weights[1][0], config['steps'][0]['coefficients']['Intercept'], places=4)
        feature_name = layer.feature_names[0]
        expected_weight = config['steps'][0]['coefficients'][feature_name]
        self.assertAlmostEqual(loaded_weights[0][0][0], expected_weight, places=4)

    def _test_logistic_hs_layer(self, layer_class, config):
        """Tests a layer that uses a LogisticLayer and an HSLayer."""
        layer = layer_class()
        dummy_input = {name: tf.zeros((13, 1)) for name in layer.feature_names}
        _ = layer(dummy_input)
        layer.load_weights_from_cfg(config)

        # Test output shape
        prediction = layer(dummy_input)
        self.assertEqual(prediction.shape, (13, 1))

        # Test weights loading for prob_layer
        prob_weights = layer.prob_layer.get_weights()
        prob_coeffs = config['steps'][0]['coefficients']
        self.assertAlmostEqual(prob_weights[1][0], prob_coeffs['Intercept'], places=4)
        prob_feature_name = layer.prob_features[0]
        expected_prob_weight = prob_coeffs[prob_feature_name]
        self.assertAlmostEqual(prob_weights[0][0][0], expected_prob_weight, places=4)

        # Test weights loading for level_layer
        level_weights = layer.level_layer.get_weights()
        level_coeffs = config['steps'][1]['coefficients']
        self.assertAlmostEqual(level_weights[1][0], level_coeffs['Intercept'], places=4)
        level_feature_name = layer.level_features[0]
        expected_level_weight = level_coeffs[level_feature_name]
        self.assertAlmostEqual(level_weights[0][0][0], expected_level_weight, places=4)

    def _test_tobit_layer(self, layer_class, config):
        """Tests a layer that uses a TobitLayer."""
        layer = layer_class()
        dummy_input = {name: tf.zeros((13, 1)) for name in layer.feature_names}
        _ = layer(dummy_input)
        layer.load_weights_from_cfg(config)

        # Test output shape
        prediction = layer(dummy_input)
        self.assertEqual(prediction.shape, (13, 1))

        # Test weights loading
        loaded_weights = layer.get_weights()
        self.assertAlmostEqual(loaded_weights[1][0], config['steps'][0]['coefficients']['Intercept'], places=4)
        feature_name = layer.feature_names[0]
        expected_weight = config['steps'][0]['coefficients'][feature_name]
        self.assertAlmostEqual(loaded_weights[0][0][0], expected_weight, places=4)
        self.assertAlmostEqual(loaded_weights[2][0], config['scale'], places=4)

    def _test_multinomial_hs_layer(self, layer_class, config):
        """Tests a layer that uses a MultinomialLayer and two HSLayers."""
        layer = layer_class()
        dummy_input = {name: tf.zeros((13, 1)) for name in layer.feature_names}
        _ = layer(dummy_input)
        layer.load_weights_from_cfg(config)

        # Test output shape
        prediction = layer(dummy_input)
        self.assertEqual(prediction.shape, (13, 1))

        # Test weights loading for prob_layer
        prob_weights = layer.prob_layer.get_weights()
        prob_coeffs = config['steps'][0]['coefficients']
        
        # Test intercepts
        for i, intercept in enumerate(prob_coeffs['Intercept']):
            self.assertAlmostEqual(prob_weights[1][i], intercept, places=4)

        # Test weights
        for i, feature_name in enumerate(layer.prob_features):
            expected_weight = prob_coeffs[feature_name]
            if isinstance(expected_weight, list):
                for j, weight in enumerate(expected_weight):
                    self.assertAlmostEqual(prob_weights[0][i][j], weight, places=4)
            else: # It's a float
                self.assertAlmostEqual(prob_weights[0][i][0], expected_weight, places=4)

        # Test weights loading for pos_level_layer
        pos_level_weights = layer.pos_level_layer.get_weights()
        pos_level_coeffs = config['steps'][1]['coefficients']
        self.assertAlmostEqual(pos_level_weights[1][0], pos_level_coeffs['Intercept'], places=4)
        pos_level_feature_name = layer.pos_level_features[0]
        expected_pos_level_weight = pos_level_coeffs[pos_level_feature_name]
        self.assertAlmostEqual(pos_level_weights[0][0][0], expected_pos_level_weight, places=4)

        # Test weights loading for neg_level_layer
        neg_level_weights = layer.neg_level_layer.get_weights()
        neg_level_coeffs = config['steps'][2]['coefficients']
        self.assertAlmostEqual(neg_level_weights[1][0], neg_level_coeffs['Intercept'], places=4)
        neg_level_feature_name = layer.neg_level_features[0]
        expected_neg_level_weight = neg_level_coeffs[neg_level_feature_name]
        self.assertAlmostEqual(neg_level_weights[0][0][0], expected_neg_level_weight, places=4)

    def _test_dual_logistic_hs_layer(self, layer_class, config):
        """Tests a layer that uses two LogisticLayers and two HSLayers."""
        layer = layer_class()
        dummy_input = {name: tf.zeros((13, 1)) for name in layer.feature_names}
        _ = layer(dummy_input)
        layer.load_weights_from_cfg(config)

        # Test output shape
        prediction = layer(dummy_input)
        self.assertEqual(prediction.shape, (13, 1))

        # Test weights loading for pos_prob_layer
        pos_prob_weights = layer.pos_prob_layer.get_weights()
        pos_prob_coeffs = config['steps'][0]['coefficients']
        self.assertAlmostEqual(pos_prob_weights[1][0], pos_prob_coeffs['Intercept'], places=4)
        pos_prob_feature_name = layer.pos_prob_features[0]
        expected_pos_prob_weight = pos_prob_coeffs[pos_prob_feature_name]
        self.assertAlmostEqual(pos_prob_weights[0][0][0], expected_pos_prob_weight, places=4)

        # Test weights loading for neg_prob_layer
        neg_prob_weights = layer.neg_prob_layer.get_weights()
        neg_prob_coeffs = config['steps'][1]['coefficients']
        self.assertAlmostEqual(neg_prob_weights[1][0], neg_prob_coeffs['Intercept'], places=4)
        neg_prob_feature_name = layer.neg_prob_features[0]
        expected_neg_prob_weight = neg_prob_coeffs[neg_prob_feature_name]
        self.assertAlmostEqual(neg_prob_weights[0][0][0], expected_neg_prob_weight, places=4)

        # Test weights loading for pos_level_layer
        pos_level_weights = layer.pos_level_layer.get_weights()
        pos_level_coeffs = config['steps'][2]['coefficients']
        self.assertAlmostEqual(pos_level_weights[1][0], pos_level_coeffs['Intercept'], places=4)
        pos_level_feature_name = layer.pos_level_features[0]
        expected_pos_level_weight = pos_level_coeffs[pos_level_feature_name]
        self.assertAlmostEqual(pos_level_weights[0][0][0], expected_pos_level_weight, places=4)

        # Test weights loading for neg_level_layer
        neg_level_weights = layer.neg_level_layer.get_weights()
        neg_level_coeffs = config['steps'][3]['coefficients']
        self.assertAlmostEqual(neg_level_weights[1][0], neg_level_coeffs['Intercept'], places=4)
        neg_level_feature_name = layer.neg_level_features[0]
        expected_neg_level_weight = neg_level_coeffs[neg_level_feature_name]
        self.assertAlmostEqual(neg_level_weights[0][0][0], expected_neg_level_weight, places=4)

    def test_dca_layer(self):
        self._test_single_hs_layer(DCALayer, DCA_CONFIG)

    def test_dcl_layer(self):
        self._test_single_hs_layer(DCLLayer, DCL_CONFIG)

    def test_oibd_layer(self):
        self._test_single_hs_layer(OIBDLayer, OIBD_CONFIG)

    def test_pallo_layer(self):
        self._test_single_hs_layer(PALLOLayer, PALLO_CONFIG)

    def test_dll_layer(self):
        self._test_logistic_hs_layer(DLLLayer, DLL_CONFIG)

    def test_drr_layer(self):
        self._test_logistic_hs_layer(DRRLayer, DRR_CONFIG)

    def test_edepbu_layer(self): 
        self._test_logistic_hs_layer(EDEPBULayer, EDEPBU_CONFIG)

    def test_edepma_layer(self):
        self._test_logistic_hs_layer(EDEPMALayer, EDEPMA_CONFIG)

    def test_fe_layer(self):
        self._test_logistic_hs_layer(FELayer, FE_CONFIG)

    def test_fi_layer(self):
        self._test_logistic_hs_layer(FILayer, FI_CONFIG)

    def test_ibu_layer(self):
        self._test_logistic_hs_layer(IBULayer, IBU_CONFIG)

    def test_rot_layer(self):
        self._test_logistic_hs_layer(ROTLayer, ROT_CONFIG)

    def test_tl_layer(self):
        self._test_logistic_hs_layer(TLLayer, TL_CONFIG)

    def test_zpf_layer(self):
        self._test_logistic_hs_layer(ZPFLayer, ZPF_CONFIG)

    def test_ima_layer(self):
        self._test_tobit_layer(IMALayer, IMA_CONFIG)

    def test_tdepbu_layer(self):
        self._test_tobit_layer(TDEPBULayer, TDEPBU_CONFIG)

    def test_tdepma_layer(self):
        self._test_tobit_layer(TDEPMALayer, TDEPMA_CONFIG)

    def test_dour_layer(self):
        self._test_multinomial_hs_layer(DOURLayer, DOUR_CONFIG)

    def test_oa_layer(self):
        self._test_multinomial_hs_layer(OALayer, OA_CONFIG)

    def test_ota_layer(self):
        self._test_multinomial_hs_layer(OTALayer, OTA_CONFIG)

    def test_sma_layer(self):
        self._test_multinomial_hs_layer(SMALayer, SMA_CONFIG)

    def test_dofa_layer(self):
        self._test_dual_logistic_hs_layer(DOFALayer, DOFA_CONFIG)

    def test_dsc_layer(self):
        self._test_dual_logistic_hs_layer(DSCLayer, DSC_CONFIG)

    def test_gc_layer(self):
        self._test_dual_logistic_hs_layer(GCLayer, GC_CONFIG)

if __name__ == '__main__':
    unittest.main()