#!/usr/bin/env python3
"""
Comprehensive tests for the Trident-XLM report_llm package.
These test the core functionality without requiring model downloads.
"""
import os
import sys
import json
import tempfile
from unittest.mock import Mock, patch

# Add the project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from report_llm.types import TelemetryNLIn, Contribution, StylePreset
from report_llm.prompt_builder import build_inputs_for_llm

class TestTypes:
    """Test the data types and structures."""
    
    def test_contribution_creation(self):
        """Test creating a Contribution."""
        contrib = Contribution(
            name="Test RCS",
            modality="RADAR (Ka-band)",
            sign="pos",
            value_pct=25.5,
            note="test signal"
        )
        assert contrib.name == "Test RCS"
        assert contrib.modality == "RADAR (Ka-band)"
        assert contrib.sign == "pos"
        assert contrib.value_pct == 25.5
        assert contrib.note == "test signal"
    
    def test_telemetry_creation(self):
        """Test creating a TelemetryNLIn."""
        contrib = Contribution("Test", "RADAR", "pos", 10.0, "note")
        telem = TelemetryNLIn(
            p_hit_calib=0.8, p_kill_calib=0.7,
            p_hit_masked=0.75, p_kill_masked=0.65,
            spoof_risk=0.1,
            flags={"test": True},
            exp={"test_exp": "value"},
            meta={"test_meta": 42},
            contributions=[contrib]
        )
        assert telem.p_hit_calib == 0.8
        assert len(telem.contributions) == 1
        assert telem.contributions[0].name == "Test"

class TestPromptBuilder:
    """Test the prompt building functionality."""
    
    def setup_method(self):
        """Set up test data."""
        self.contrib_pos = Contribution("RCS", "RADAR", "pos", 25.5, "good signal")
        self.contrib_neg = Contribution("Noise", "EO", "neg", -10.2, "bad noise")
        self.telem = TelemetryNLIn(
            p_hit_calib=0.8, p_kill_calib=0.7,
            p_hit_masked=0.75, p_kill_masked=0.65,
            spoof_risk=0.1,
            flags={"test": True},
            exp={"test_exp": "value"},
            meta={"test_meta": 42},
            contributions=[self.contrib_pos, self.contrib_neg]
        )
    
    def test_prompt_building_resmi(self):
        """Test prompt building with resmi style."""
        prompt = build_inputs_for_llm(self.telem, style="resmi")
        assert "resmi" in prompt
        assert "Görev:" in prompt
        assert "p_hit_calib=0.80" in prompt
        assert "RCS | RADAR | +25.50" in prompt
        assert "Noise | EO | -10.20" in prompt
    
    def test_prompt_building_madde(self):
        """Test prompt building with madde style."""
        prompt = build_inputs_for_llm(self.telem, style="madde")
        assert "madde" in prompt
        assert "Görev:" in prompt
    
    def test_prompt_building_anlatimci(self):
        """Test prompt building with anlatımcı style."""
        prompt = build_inputs_for_llm(self.telem, style="anlatımcı")
        assert "anlatımcı" in prompt
        assert "Görev:" in prompt
    
    def test_no_contributions(self):
        """Test prompt building with no contributions."""
        telem_empty = TelemetryNLIn(
            p_hit_calib=0.8, p_kill_calib=0.7,
            p_hit_masked=0.75, p_kill_masked=0.65,
            spoof_risk=0.1,
            flags={}, exp={}, meta={},
            contributions=[]
        )
        prompt = build_inputs_for_llm(telem_empty, style="resmi")
        assert "(yok)" in prompt

class TestGuards:
    """Test the guard functionality."""
    
    def test_angle_guard_pattern(self):
        """Test angle bracket detection pattern."""
        import re
        pattern = re.compile(r"<[^>]+>")
        
        # Should detect angle brackets
        assert pattern.search("text with <system> token")
        assert pattern.search("multiple <user> and <gun> tokens")
        assert pattern.search("<single>")
        
        # Should not detect
        assert not pattern.search("normal text")
        assert not pattern.search("less than < but no close")
        assert not pattern.search("greater than > but no open")
    
    def test_operational_guard_pattern(self):
        """Test operational word detection pattern."""
        import re
        pattern = re.compile(r"\b(ateş|ateşle|nişan|vur|angaje ol|hedefe ateş)\b", re.I)
        
        # Should detect operational words
        assert pattern.search("hedefe ateş et")
        assert pattern.search("Ateş edin")  # case insensitive
        assert pattern.search("nişan al")
        assert pattern.search("vur hedefi")
        
        # Should not detect
        assert not pattern.search("ateşci değil")  # partial word
        assert not pattern.search("normal teknik metin")
        assert not pattern.search("RCS ve Doppler")

class TestDatasetBuilding:
    """Test dataset building functionality."""
    
    def test_load_telemetry_format(self):
        """Test loading telemetry data format."""
        # Create a temporary telemetry file
        sample_data = {
            "inputs": {
                "p_hit_calib": 0.8, "p_kill_calib": 0.7,
                "p_hit_masked": 0.75, "p_kill_masked": 0.65,
                "spoof_risk": 0.1,
                "flags": {"test": True},
                "exp": {"test_exp": "value"},
                "meta": {"test_meta": 42},
                "contributions": [
                    {"name": "Test", "modality": "RADAR", "sign": "pos", "value_pct": 10.0, "note": "test"}
                ]
            },
            "targets": {
                "one_liner": "Test one liner",
                "report": "Test report text"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps(sample_data, ensure_ascii=False) + '\n')
            temp_path = f.name
        
        try:
            # Test the loading function
            from report_llm.build_dataset import load_telemetry, to_struct
            
            data_generator = load_telemetry(temp_path)
            loaded_data = next(data_generator)
            
            assert "inputs" in loaded_data
            assert "targets" in loaded_data
            
            # Test conversion to struct
            telem_struct = to_struct(loaded_data["inputs"])
            assert isinstance(telem_struct, TelemetryNLIn)
            assert telem_struct.p_hit_calib == 0.8
            assert len(telem_struct.contributions) == 1
            
        finally:
            os.unlink(temp_path)

class TestInferenceLogic:
    """Test inference-related logic without actual models."""
    
    def test_mock_summarizer_logic(self):
        """Test the logic flow of summarizers."""
        # This tests the structure without actual model inference
        
        # Mock the model loading and generation
        with patch('report_llm.summarizer_sync.load_model') as mock_load:
            with patch('report_llm.summarizer_sync.AutoTokenizer') as mock_tok:
                with patch('report_llm.summarizer_sync.AutoModelForSeq2SeqLM') as mock_model:
                    # Setup mocks
                    mock_tokenizer = Mock()
                    mock_tokenizer.return_value = {"input_ids": [1, 2, 3]}
                    mock_tokenizer.decode.return_value = "Valid Turkish text"
                    
                    mock_mdl = Mock()
                    mock_mdl.generate.return_value = [[1, 2, 3]]
                    
                    mock_load.return_value = (mock_tokenizer, mock_mdl)
                    
                    # Test the function structure
                    from report_llm.summarizer_sync import make_one_liner
                    
                    result = make_one_liner("mock_model_dir", "test prompt")
                    assert result == "Valid Turkish text"
                    
                    # Verify the flow
                    mock_load.assert_called_once_with("mock_model_dir")
                    mock_tokenizer.assert_called_once()
                    mock_mdl.generate.assert_called_once()

def run_tests():
    """Run all tests manually without pytest."""
    print("🧪 Running Trident-XLM Unit Tests")
    print("=" * 50)
    
    test_classes = [TestTypes, TestPromptBuilder, TestGuards, TestDatasetBuilding]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n📋 {test_class.__name__}")
        instance = test_class()
        
        # Setup if available
        if hasattr(instance, 'setup_method'):
            instance.setup_method()
        
        # Run test methods
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                total_tests += 1
                try:
                    method = getattr(instance, method_name)
                    method()
                    print(f"  ✅ {method_name}")
                    passed_tests += 1
                except Exception as e:
                    print(f"  ❌ {method_name}: {e}")
    
    print(f"\n📊 Results: {passed_tests}/{total_tests} tests passed")
    return passed_tests == total_tests

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)