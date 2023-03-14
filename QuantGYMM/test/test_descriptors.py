from QuantGYMM.descriptors import *
import pytest
import pandas as pd
import numpy as np


# TODO: test of float number

@pytest.fixture
def instance_class():
    obj = type("TestClass", (), {"date": Date(),
                                 "positive_number": PositiveNumber(),
                                 "non_negative_integer": NonNegativeInteger(),
                                 "positive_integer": PositiveInteger(),
                                 "business_convention": BusinessConvention(),
                                 "float_number": FloatNumber(),
                                 "string": String(),
                                 "compounding_convention": CompoundingConvention(),
                                 "bool": Boolean(),
                                 "day_count_convention": DayCountConvention(),
                                 "pandas": DataFrame()})
    return obj()


@pytest.fixture
def instance_class_with_none():
    obj = type("TestClass", (), {"date": Date(),
                                 "positive_number": PositiveNumber(none_accepted=True),
                                 "non_negative_integer": NonNegativeInteger(none_accepted=True),
                                 "positive_integer": PositiveInteger(none_accepted=True),
                                 "float_number": FloatNumber(none_accepted=True)})
    return obj()


@pytest.fixture
def instance_class_with_none_and_nan():
    obj = type("TestClass", (), {"date": Date(),
                                 "positive_number": PositiveNumber(none_accepted=True, return_if_none=np.nan),
                                 "non_negative_integer": NonNegativeInteger(none_accepted=True, return_if_none=np.nan),
                                 "positive_integer": PositiveInteger(none_accepted=True, return_if_none=np.nan),
                                 "float_number": FloatNumber(none_accepted=True, return_if_none=np.nan)})
    return obj()


@pytest.fixture
def instance_class_dataframe():
    obj = type("TestClass", (), {"pandas": DataFrame(index_type=pd.DatetimeIndex)})
    return obj()


@pytest.mark.parametrize("date", ["2023-01-01", pd.to_datetime("2023-01-01")])
def test_valid_date(instance_class, date):
    instance_class.date = date
    assert instance_class.date == pd.to_datetime(date)


@pytest.mark.parametrize("date", [[1, 2, 3], "2023", "20230101", "hello", 2, False])
def test_invalid_date(instance_class, date):
    with pytest.raises(TypeError):
        instance_class.date = date


@pytest.mark.parametrize("positive_number", [0.1, 1])
def test_valid_positive_number(instance_class, positive_number):
    instance_class.positive_number = positive_number
    assert instance_class.positive_number == positive_number


@pytest.mark.parametrize("positive_number", [[1, 2, 3], "2023", -2, False, None])
def test_invalid_positive_number(instance_class, positive_number):
    with pytest.raises(TypeError):
        instance_class.positive_number = positive_number


@pytest.mark.parametrize("positive_number", [0.1, 1, None])
def test_valid_positive_number_with_none(instance_class_with_none, positive_number):
    instance_class_with_none.positive_number = positive_number
    assert instance_class_with_none.positive_number == positive_number


@pytest.mark.parametrize("positive_number", [[1, 2, 3], "2023", -2, False])
def test_invalid_positive_number_with_none(instance_class_with_none, positive_number):
    with pytest.raises(TypeError):
        instance_class_with_none.positive_number = positive_number


@pytest.mark.parametrize("positive_number", [0.1, 1, None])
def test_valid_positive_number_with_none_and_nan(instance_class_with_none_and_nan, positive_number):
    instance_class_with_none_and_nan.positive_number = positive_number
    if positive_number is None:
        assert instance_class_with_none_and_nan.positive_number is np.nan
    else:
        assert instance_class_with_none_and_nan.positive_number == positive_number


@pytest.mark.parametrize("non_negative_integer", [0, 1])
def test_valid_non_negative_integer(instance_class, non_negative_integer):
    instance_class.non_negative_integer = non_negative_integer
    assert instance_class.non_negative_integer == non_negative_integer


@pytest.mark.parametrize("non_negative_integer", [[1, 2, 3], -2, 0.3, "2", False, True, None])
def test_invalid_non_negative_integer(instance_class, non_negative_integer):
    with pytest.raises(TypeError):
        instance_class.non_negative_integer = non_negative_integer


@pytest.mark.parametrize("non_negative_integer", [0, 1, None])
def test_valid_non_negative_integer_with_none(instance_class_with_none, non_negative_integer):
    instance_class_with_none.non_negative_integer = non_negative_integer
    assert instance_class_with_none.non_negative_integer == non_negative_integer


@pytest.mark.parametrize("non_negative_integer", [0, 1, None])
def test_valid_non_negative_integer_with_none_and_nan(instance_class_with_none_and_nan, non_negative_integer):
    instance_class_with_none_and_nan.non_negative_integer = non_negative_integer
    if non_negative_integer is None:
        assert instance_class_with_none_and_nan.non_negative_integer is np.nan
    else:
        assert instance_class_with_none_and_nan.non_negative_integer == non_negative_integer


@pytest.mark.parametrize("positive_integer", [1])
def test_valid_positive_integer(instance_class, positive_integer):
    instance_class.positive_integer = positive_integer
    assert instance_class.positive_integer == positive_integer


@pytest.mark.parametrize("positive_integer", [-1, True, False, "2", 0.01, None])
def test_invalid_positive_integer(instance_class, positive_integer):
    with pytest.raises(TypeError):
        instance_class.positive_integer = positive_integer


@pytest.mark.parametrize("positive_integer", [1, None])
def test_valid_positive_integer_with_none(instance_class_with_none, positive_integer):
    instance_class_with_none.positive_integer = positive_integer
    assert instance_class_with_none.positive_integer == positive_integer


@pytest.mark.parametrize("positive_integer", [1, None])
def test_valid_positive_integer_with_none_and_nan(instance_class_with_none_and_nan, positive_integer):
    instance_class_with_none_and_nan.positive_integer = positive_integer
    if positive_integer is None:
        assert instance_class_with_none_and_nan.positive_integer is np.nan
    else:
        assert instance_class_with_none_and_nan.positive_integer == positive_integer


@pytest.mark.parametrize("business_convention",
                         ["following", "modified_following", "preceding", "modified_following_bimonthly"])
def test_valid_business_convention(instance_class, business_convention):
    instance_class.business_convention = business_convention
    assert instance_class.business_convention == business_convention


@pytest.mark.parametrize("business_convention", ["hello", "4", 1, False, True, None])
def test_invalid_business_convention(instance_class, business_convention):
    with pytest.raises(ValueError):
        instance_class.business_convention = business_convention


@pytest.mark.parametrize("compounding_convention", ["annually_compounded", "simple", "continuous"])
def test_valid_compounding_convention(instance_class, compounding_convention):
    instance_class.compounding_convention = compounding_convention
    assert instance_class.compounding_convention == compounding_convention


@pytest.mark.parametrize("compounding_convention", ["discounted", 2, "2", False, True, None])
def test_invalid_compounding_convention(instance_class, compounding_convention):
    with pytest.raises(ValueError):
        instance_class.compounding_convention = compounding_convention


@pytest.mark.parametrize("day_count_convention", ["ACT/360", "ACT/365", "30/360"])
def test_valid_dcc_convention(instance_class, day_count_convention):
    instance_class.day_count_convention = day_count_convention
    assert instance_class.day_count_convention == day_count_convention


@pytest.mark.parametrize("day_count_convention", ["ACT/ACT", 1, False, True, None])
def test_invalid_dcc_convention(instance_class, day_count_convention):
    with pytest.raises(ValueError):
        instance_class.day_count_convention = day_count_convention


@pytest.mark.parametrize("string", ["ACT/360", "hello", "False"])
def test_valid_string_convention(instance_class, string):
    instance_class.string = string
    assert instance_class.string == string


@pytest.mark.parametrize("string", [1, False, True, None])
def test_invalid_string_convention(instance_class, string):
    with pytest.raises(TypeError):
        instance_class.string = string


@pytest.mark.parametrize("bool", [True, False])
def test_valid_bool_convention(instance_class, bool):
    instance_class.bool = bool
    assert instance_class.bool == bool


@pytest.mark.parametrize("bool", [0, 1, None, "False"])
def test_invalid_bool_convention(instance_class, bool):
    with pytest.raises(TypeError):
        instance_class.bool = bool


@pytest.mark.parametrize("pandas",
                         [pd.DataFrame({"data": [1, 2, 3, 4]}, index=pd.date_range("2023-01-01", "2023-01-04")),
                          pd.DataFrame({"data": [1, 2, 3, 4]}, index=pd.RangeIndex(4)),
                          pd.DataFrame({"data": [1, 2, 3, 4]})])
def test_valid_dataframe(instance_class, pandas):
    instance_class.pandas = pandas
    assert instance_class.pandas is pandas


@pytest.mark.parametrize("pandas",
                         [{"data": [1, 2, 3, 4], "index": pd.date_range("2023-01-01", "2023-01-04")},
                          np.array([[1, 2, 3], [1, 2, 3]]), "hi", 4])
def test_invalid_dataframe(instance_class, pandas):
    with pytest.raises(TypeError):
        instance_class.pandas = pandas


@pytest.mark.parametrize("pandas",
                         [pd.DataFrame({"data": [1, 2, 3, 4]}, index=pd.date_range("2023-01-01", "2023-01-04"))])
def test_valid_dataframe_index(instance_class_dataframe, pandas):
    instance_class_dataframe.pandas = pandas
    assert instance_class_dataframe.pandas is pandas


@pytest.mark.parametrize("pandas", [pd.DataFrame({"data": [1, 2, 3, 4]}),
                                    pd.DataFrame({"data": [1, 2, 3, 4]}, index=pd.RangeIndex(4))])
def test_invalid_dataframe_index(instance_class_dataframe, pandas):
    with pytest.raises(IndexError):
        instance_class_dataframe.pandas = pandas


@pytest.mark.parametrize("float_number", [0.1, -0.1])
def test_valid_float_number(instance_class, float_number):
    instance_class.float_number = float_number
    assert instance_class.float_number == float_number

@pytest.mark.parametrize("float_number", [0, 1, -1, "0.1", None, False, True])
def test_invalid_float_number(instance_class, float_number):
    with pytest.raises(TypeError):
        instance_class.float_number = float_number

@pytest.mark.parametrize("float_number", [0.1, -0.1, None])
def test_valid_float_number_with_none(instance_class_with_none, float_number):
    instance_class.float_number = float_number
    assert instance_class.float_number == float_number

@pytest.mark.parametrize("float_number", [0.1, -0.1, None])
def test_valid_float_number_with_none_and_nan(instance_class_with_none_and_nan, float_number):
    instance_class_with_none_and_nan.float_number = float_number
    if float_number is None:
        assert instance_class_with_none_and_nan.float_number is np.nan
    else:
        assert instance_class_with_none_and_nan.float_number == float_number