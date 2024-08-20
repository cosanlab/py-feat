import pytest
import sys
from feat.au_detectors.StatLearning.SL_test import XGBClassifier

# Explicitly add XGBClassifier to __main__ namespace
sys.modules['__main__'].__dict__['XGBClassifier'] = XGBClassifier

def test_xgbclassifier_in_main():
    # Check if XGBClassifier is in __main__
    assert 'XGBClassifier' in sys.modules['__main__'].__dict__, "XGBClassifier not found in __main__"
    print("XGBClassifier found in __main__")
    print(sys.modules['__main__'].__dict__)

if __name__ == "__main__":
    pytest.main([__file__])