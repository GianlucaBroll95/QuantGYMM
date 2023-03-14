import pandas
import pandas as pd
import datetime


class Date:
    def __init__(self, sterilize_attr=None):
        if sterilize_attr is None:
            sterilize_attr = []
        self.sterilize_attr = sterilize_attr

    def __set_name__(self, owner, name):
        self.property_name = name

    def __set__(self, instance, value):
        if isinstance(value, str):
            try:

                instance.__dict__[self.property_name] = pd.to_datetime(datetime.date.fromisoformat(value))
            except Exception:
                raise TypeError(f"Can't convert '{value}' into datetime.")
        elif isinstance(value, pandas.Timestamp):
            instance.__dict__[self.property_name] = value
        else:
            raise TypeError(f"Wrong type for '{self.property_name}'. Accepted types are string or pandas.Timestamp.")
        if self.sterilize_attr:
            for attr in self.sterilize_attr:
                instance.__dict__[attr] = None

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.property_name, None)


class PositiveNumber:
    def __init__(self, sterilize_attr=None, none_accepted=False, return_if_none=None):
        if sterilize_attr is None:
            sterilize_attr = []
        self.sterilize_attr = sterilize_attr
        self.none_accepted = none_accepted
        self.return_if_none = return_if_none

    def __set_name__(self, owner, name):
        self.property_name = name

    def __set__(self, instance, value):
        if (isinstance(value, (float, int)) and value > 0) or (value is None and self.none_accepted):
            instance.__dict__[self.property_name] = value
        else:
            raise TypeError(f"'{self.property_name}' must be a positive number.")
        if self.sterilize_attr:
            for attr in self.sterilize_attr:
                instance.__dict__[attr] = None

    def __get__(self, instance, owner):
        if instance is None:
            return self
        if instance.__dict__.get(self.property_name, None) is None:
            return self.return_if_none
        return instance.__dict__.get(self.property_name, None)


class PositiveInteger:
    def __init__(self, sterilize_attr=None, none_accepted=False, return_if_none=None):
        if sterilize_attr is None:
            sterilize_attr = []
        self.sterilize_attr = sterilize_attr
        self.none_accepted = none_accepted
        self.return_if_none = return_if_none

    def __set_name__(self, owner, name):
        self.property_name = name

    def __set__(self, instance, value):
        if ((isinstance(value, int) and value > 0) and not isinstance(value, bool)) or (value is None and self.none_accepted):
            instance.__dict__[self.property_name] = value
        else:
            raise TypeError(f"Wrong type for '{value}'. Accepted types are positive integers.")

    def __get__(self, instance, owner):
        if instance is None:
            return self
        if instance.__dict__.get(self.property_name, None) is None:
            return self.return_if_none
        return instance.__dict__.get(self.property_name, None)


class NonNegativeInteger:
    def __init__(self, sterilize_attr=None, none_accepted=False, return_if_none=None):
        if sterilize_attr is None:
            sterilize_attr = []
        self.sterilize_attr = sterilize_attr
        self.none_accepted = none_accepted
        self.return_if_none = return_if_none

    def __set_name__(self, owner, name):
        self.property_name = name

    def __set__(self, instance, value):
        if ((isinstance(value, int) and value >= 0) and not isinstance(value, bool)) or (value is None and self.none_accepted) :
            instance.__dict__[self.property_name] = value
        else:
            raise TypeError(f"Wrong type for '{value}'. Accepted types are non negative integers.")

    def __get__(self, instance, owner):
        if instance is None:
            return self
        if instance.__dict__.get(self.property_name, None) is None:
            return self.return_if_none
        return instance.__dict__.get(self.property_name, None)


class BusinessConvention:
    _BUSINESS_CONVENTIONS = ["following", "modified_following", "preceding", "modified_following_bimonthly"]

    def __init__(self, sterilize_attr=None):
        if sterilize_attr is None:
            sterilize_attr = []
        self.sterilize_attr = sterilize_attr

    def __set_name__(self, owner, name):
        self.property_name = name

    def __set__(self, instance, value):
        if value in self._BUSINESS_CONVENTIONS:
            instance.__dict__[self.property_name] = value
        else:
            raise ValueError(f"'{value}' is not a valid business convention. "
                             f"Allowed choices are: {', '.join(self._BUSINESS_CONVENTIONS)}.")
        if self.sterilize_attr:
            for attr in self.sterilize_attr:
                instance.__dict__[attr] = None

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.property_name, None)


class Boolean:
    def __init__(self, sterilize_attr=None):
        if sterilize_attr is None:
            sterilize_attr = []
        self.sterilize_attr = sterilize_attr

    def __set_name__(self, owner, name):
        self.property_name = name

    def __set__(self, instance, value):
        if isinstance(value, bool):
            instance.__dict__[self.property_name] = value
        else:
            raise TypeError(f"Wrong type for '{self.property_name}'. Allowed choices are booleans.")
        if self.sterilize_attr:
            for attr in self.sterilize_attr:
                instance.__dict__[attr] = None

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.property_name, None)


class DayCountConvention:
    _DAY_COUNT_CONVENTIONS = ["30/360", "ACT/360", "ACT/365"]

    def __init__(self, sterilize_attr=None):
        if sterilize_attr is None:
            sterilize_attr = []
        self.sterilize_attr = sterilize_attr

    def __set_name__(self, owner, name):
        self.property_name = name

    def __set__(self, instance, value):
        if value in self._DAY_COUNT_CONVENTIONS:
            instance.__dict__[self.property_name] = value
        else:
            raise ValueError(f"'{value}' is not a valid day count convention. "
                             f"Allowed choices are: {', '.join(self._DAY_COUNT_CONVENTIONS)}.")
        if self.sterilize_attr:
            for attr in self.sterilize_attr:
                instance.__dict__[attr] = None

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.property_name, None)


class FloatNumber:
    def __init__(self, sterilize_attr=None, none_accepted=False, return_if_none=None):
        if sterilize_attr is None:
            sterilize_attr = []
        self.sterilize_attr = sterilize_attr
        self.none_accepted = none_accepted
        self.return_if_none = return_if_none

    def __set_name__(self, owner, name):
        self.property_name = name

    def __set__(self, instance, value):
        if isinstance(value, float) or (value is None and self.none_accepted):
            instance.__dict__[self.property_name] = value
        else:
            raise TypeError(f"Wrong type for '{self.property_name}'. Allowed choices are float.")
        if self.sterilize_attr:
            for attr in self.sterilize_attr:
                instance.__dict__[attr] = None

    def __get__(self, instance, owner):
        if instance is None:
            return self
        if instance.__dict__.get(self.property_name, None) is None:
            return self.return_if_none
        return instance.__dict__.get(self.property_name, None)


class String:
    def __init__(self, sterilize_attr=None):
        if sterilize_attr is None:
            sterilize_attr = []
        self.sterilize_attr = sterilize_attr

    def __set_name__(self, owner, name):
        self.property_name = name

    def __set__(self, instance, value):
        if isinstance(value, str):
            instance.__dict__[self.property_name] = value
        else:
            raise TypeError(f"Wrong type for '{self.property_name}'. Allowed choices are string.")
        if self.sterilize_attr:
            for attr in self.sterilize_attr:
                instance.__dict__[attr] = None

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.property_name, None)


class CompoundingConvention:
    _COMPOUNDING_CONVENTIONS = ["annually_compounded", "simple", "continuous"]

    def __init__(self, sterilize_attr=None):
        if sterilize_attr is None:
            sterilize_attr = []
        self.sterilize_attr = sterilize_attr

    def __set_name__(self, owner, name):
        self.property_name = name

    def __set__(self, instance, value):
        if value in self._COMPOUNDING_CONVENTIONS:
            instance.__dict__[self.property_name] = value
        else:
            raise ValueError(f"'{value}' is not a valid day compounding convention. "
                             f"Allowed choices are: {', '.join(self._COMPOUNDING_CONVENTIONS)}.")
        if self.sterilize_attr:
            for attr in self.sterilize_attr:
                instance.__dict__[attr] = None

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.property_name, None)


class DataFrame:
    def __init__(self, index_type=None, sterilize_attr=None):
        self.index_type = index_type
        if sterilize_attr is None:
            sterilize_attr = []
        self.sterilize_attr = sterilize_attr

    def __set_name__(self, owner, name):
        self.property_name = name

    def __set__(self, instance, value):
        if isinstance(value, pandas.DataFrame):
            if self.index_type:
                if isinstance(value.index, self.index_type):
                    instance.__dict__[self.property_name] = value
                else:
                    raise IndexError(
                        f"'{value}' has not a {self.index_type}. Please provide a DataFrame with {self.index_type}.")
            else:
                instance.__dict__[self.property_name] = value
        else:
            raise TypeError(f"Expected DataFrame, got {type(value)}.")

        if self.sterilize_attr:
            for attr in self.sterilize_attr:
                instance.__dict__[attr] = None

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.property_name, None)
