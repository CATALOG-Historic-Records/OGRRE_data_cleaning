import pytest
from datetime import datetime
from ogrre_data_cleaning.clean import string_to_float, string_to_int, llm_clean, clean_date, clean_bool, convert_hole_size_to_decimal

@pytest.mark.unit
@pytest.mark.parametrize("input_value, expected", [
    ("123.45", 123.45),
    ("$123.45", 123.45),
    (123.45, 123.45),
    ("not a number", None),
    (None, None),
])
def test_string_to_float(input_value, expected):
    assert string_to_float(input_value) == expected

@pytest.mark.unit
def test_string_to_int():
    assert string_to_int("123") == 123
    assert string_to_int("$123") == 123
    assert string_to_int(42) == 42
    assert string_to_int("not a number") is None
    assert string_to_int(None) is None

@pytest.mark.unit
@pytest.mark.parametrize("input_value, expected", [
    ("6/25/1971", "06/25/1971"),
    ("25/10/1971", "10/25/1971"),
    ("2020/8/1", "08/01/2020"),
    ("April 28,1958", "04/28/1958"),
    ("5-27-66", "05/27/1966"),
    ("11/29/54", "11/29/1954"),
    ("3-15-22", "03/15/2022"),
    ("7/4/99", "07/04/1999"),
    (None, None),
    ("", None),
])
def test_clean_date(input_value, expected):
    assert clean_date(input_value) == expected

# ## TODO: should this raise an error?
@pytest.mark.unit
@pytest.mark.parametrize("invalid_input", [
    "13/45/1995"
])
def test_clean_date_invalid(invalid_input):
    with pytest.raises(ValueError):
        clean_date(invalid_input)

@pytest.mark.unit
@pytest.mark.unit
@pytest.mark.parametrize("input_value, expected", [
    (' yes ', True),
    ('true', True),
    ('t', True),
    ('y', True),
    ('1', True),
    ('no', False),
    (None, False),
    ('', False),
    (True, True),
    (False, False),
    ('test', False)
])
def test_clean_bool(input_value, expected):
    assert clean_bool(input_value) == expected

@pytest.mark.unit
@pytest.mark.parametrize("input_value, expected", [
    ("8 3/4", 8.75),
    ("7-7/8", 7.875),
    ("13 3/8", 13.375),
    (None, None),
    ("", None),
    ("8-3/4\u2033", 8.75), # unicode double prime
    ("None", None),
    ("N/A", None),
    (8.75, 8.75),
    ("5\u00bd", 5.5), # unicode ½
    ("85/8", 8.625),
    ("95/8", 9.625),
    ("133/8", 13.375),
    ("8 3/4\" OD", 8.75),
])
def test_convert_hole_size_to_decimal(input_value, expected):
    assert convert_hole_size_to_decimal(input_value) == expected

## TODO: should these produce errors?
@pytest.mark.unit
@pytest.mark.parametrize("invalid_input", [
    "17 1/2, 12 1/4, 7-7/8",
    "8 3/4 4265",
])
def test_convert_hole_size_to_decimal_invalid(invalid_input):
    with pytest.raises(ValueError):
        convert_hole_size_to_decimal(invalid_input)

@pytest.mark.unit
def test_llm_clean():
    # Test basic functionality
    result = llm_clean("6 inch")
    assert isinstance(result, (float, int))
    
    # Test with invalid model
    with pytest.raises(Exception):
        llm_clean("test", model_name="nonexistent")

# if __name__ == '__main__':
#     test_clean_date()
#     test_clean_bool()
#     test_convert_hole_size_to_decimal()
#     test_string_to_int()
#     test_string_to_float()
#     test_llm_clean()

#     test_convert_hole_size_to_decimal_invalid()
#     test_clean_date_invalid()