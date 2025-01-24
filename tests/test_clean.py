import pytest
from ogrre_data_cleaning import string_to_float, string_to_int, llm_clean

def test_string_to_float():
    assert string_to_float("123.45") == 123.45
    assert string_to_float("$123.45") == 123.45
    assert string_to_float("not a number") is None
    assert string_to_float(None) is None

def test_string_to_int():
    assert string_to_int("123") == 123
    assert string_to_int("$123") == 123
    assert string_to_int("not a number") is None
    assert string_to_int(None) is None

def test_llm_clean():
    # Test basic functionality
    result = llm_clean("6 inch")
    assert isinstance(result, (float, int))
    
    # Test with invalid model
    with pytest.raises(Exception):
        llm_clean("test", model_name="nonexistent") 