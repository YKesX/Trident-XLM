import pytest, re, sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from report_llm.summarizer_sync import _guard_no_angle_brackets, _guard_non_operational

def test_angle_guard():
    with pytest.raises(ValueError):
        _guard_no_angle_brackets("this has <system> inside")

def test_ops_guard():
    with pytest.raises(ValueError):
        _guard_non_operational("hedefe ateş edin")
