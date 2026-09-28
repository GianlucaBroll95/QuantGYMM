from contextlib import ExitStack, contextmanager

import numpy as np
from scipy.linalg import solve_triangular
from .curves import YieldCurve

import pandas as pd


__all__ = ["CreditRisk", "RateRisk"]


class CreditRisk:
    """
    Finite difference credit risk (spread sensitivity).
    """
    bonds: list

    @staticmethod
    @contextmanager
    def _spread_shocked(bonds, size):
        saved = [bond.z_spread for bond in bonds]
        for bond, z in zip(bonds, saved):
            bond.z_spread = z + size
        try:
            yield
        finally:
            for bond, z in zip(bonds, saved):
                bond.z_spread = z

    def cs01(self, shift_size=0.0001, kind="symmetric") -> float:
        """
        Signed price change for a +1bp move of the z-spread of every bond held, curves untouched.
        Args:
            shift_size (float): shock size (default 1bp).
            kind (str): 'symmetric' (centred, more accurate) or 'oneside'.
        Returns:
            float: negative for a long position.
        """
        if kind not in ("symmetric", "oneside"):
            raise ValueError("Admitted kind types are: 'symmetric', 'oneside'.")

        def price(size):
            with self._spread_shocked(self.bonds, size):
                return self._dirty_price()

        if kind == "symmetric":
            return (price(shift_size) - price(-shift_size)) / (2 * shift_size) * 0.0001
        return (price(shift_size) - self._dirty_price()) / shift_size * 0.0001

    def spread_duration(self, shift_size=0.0001, kind="symmetric") -> float:
        """
        Compute the spread duration of the bond via a parallel finite difference.
        Args:
            shift_size (float): parallel shock size (default 1bp).
            kind (str): 'symmetric' (centred, more accurate) or 'oneside'.
        Returns:
            spread duration (years per unit of yield change).
        """
        return float(-self.cs01(shift_size, kind) * 10000 / self._dirty_price())

    def _dirty_price(self) -> float:
        raise NotImplementedError


class RateRisk:
    """
    Finite difference rate risk (rate sensitivity).
    """
    discount_curve: YieldCurve
    projection_curves: list[YieldCurve]

    @staticmethod
    def _shift_shape(shift_type, size, n):
        match shift_type:
            case "parallel":
                return size
            case "slope":
                return np.linspace(size, -size, n)
            case "curvature":
                x = np.linspace(0, 1, n)
                return 8.0 * size * x * (x - 1.0) + size
            case _:
                raise ValueError("Admitted shift types are: 'parallel', 'slope', 'curvature'.")

    @staticmethod
    def _shocked(target, others, bump, sticky, size, node):
        shock = "shocked_quotes" if bump == "quotes" else "shocked_zeros"
        stack = ExitStack()
        for curve in (target, *others):
            curve._ensure_current()
        try:
            if sticky == "zeros":
                for other in others:
                    stack.enter_context(other.frozen())
            stack.enter_context(getattr(target, shock)(size, node))
            if sticky == "spread":
                for other in others:
                    stack.enter_context(getattr(other, shock)(size, node))
        except Exception:
            stack.close()
            raise
        return stack

    def _dirty_price(self) -> float:
        raise NotImplementedError

    def _target(self, curve):
        if isinstance(curve, YieldCurve):
            if curve is not self.discount_curve and curve not in self.projection_curves:
                raise ValueError("This curve does not price anything.")
            return curve
        if curve == "discount":
            return self.discount_curve
        if curve == "projection":
            if not self.projection_curves:
                raise ValueError("This instrument has no projection curve: its cash flows do not depend on an index.")
            if len(self.projection_curves) > 1:
                raise ValueError(
                    "'projection' is ambiguous with more than one projection curve: pass the curve itself.")
            return self.projection_curves[0]
        raise ValueError("Admitted curves are: 'projection', 'discount', or YieldCurve.")

    def sensitivity(self, curve="discount", bump="zeros", sticky="zeros", shift_type="parallel", shift_size=0.0001,
                    kind="symmetric", node=None) -> float:
        """
         Signed price change per basis point, by finite difference: negative for a long position
         (Coleman and Bloomberg report the opposite sign), in currency on the face amount.
         Args:
             curve (str | YieldCurve): 'discount' or 'projection' or YieldCurve object.
             bump (str): 'quotes' moves the market quotes (par rates of the calibration instruments)
                 and solves the curve again, 'zeros' moves the zero rates and leaves the quotes behind.
             sticky (str): what the other curves do — 'quotes' lets them follow, 'zeros' holds their zero
                 rates, 'spread' moves them by the same amount.
             shift_type (str): 'parallel', 'slope' or 'curvature'; ignored when a node is named.
             shift_size (float): size of the move.
             kind (str): 'symmetric' or 'oneside'.
             node (int | str | pandas.Timestamp): [optional] one node instead of the whole curve.
         Returns:
             float: price change for a +1bp move.
         """
        if bump not in ("quotes", "zeros"):
            raise ValueError("Admitted bumps are: 'quotes', 'zeros'.")
        if sticky not in ("quotes", "zeros", "spread"):
            raise ValueError("Admitted sticky kinds are: 'quotes', 'zeros', 'spread'.")
        if kind not in ("symmetric", "oneside"):
            raise ValueError("Admitted kind types are: 'symmetric', 'oneside'.")
        if sticky == "spread" and isinstance(node, int):
            raise ValueError("'spread' needs a tenor or a date, not a node index: two curves do not share a node grid.")
        if sticky == "spread" and node is not None and bump == "quotes":
            raise ValueError("'spread' on a single node needs 'zeros': two curves rarely quote the same"
                             " tenors, so one quote cannot be moved on both.")
        if sticky == "spread" and node is None and shift_type != "parallel":
            raise ValueError("'spread' on the whole curve demands 'parallel': two curves do not share a node grid, "
                             "placing 'slope' or 'curvature' makes no sense.")

        target = self._target(curve)

        if curve == "projection" and target is self.discount_curve:
            raise ValueError(
                "The index projects on the very curve the pricer discounts with, so there is no separate "
                "projection risk: 'discount' already carries it. Build the index on its own curve to split the two.")
        curves = list(dict.fromkeys((self.discount_curve, *self.projection_curves)))
        others = [c for c in curves if c is not target]

        def price(size):
            shift = size if node is not None else self._shift_shape(shift_type, size, len(target.instruments))

            with self._shocked(target, others, bump, sticky, shift, node):
                return self._dirty_price()

        if kind == "symmetric":
            return (price(shift_size) - price(-shift_size)) / (2 * shift_size) * 0.0001
        return (price(shift_size) - self._dirty_price()) / shift_size * 0.0001

    def total_dv01(self, bump="zeros", sticky="zeros", shift_size=0.0001, kind="symmetric") -> float:
        """
        Signed price change for a +1bp parallel move of every curve together: the sum of dv01 over the curves.
        Args:
            bump (str): 'quotes', 'zeros' whether to shock the market quotes or the zero rates
            sticky (str): 'quotes', 'zeros', how to propagate the shock among dependent curves
            shift_size (float): size of the shock
            kind (str): 'oneside' or 'symmetric', how to calculate the finite difference
        Returns:
            float
        """
        return sum(self.dv01(bump, sticky, shift_size, kind).values())

    def dv01(self, bump="zeros", sticky="zeros", shift_size=0.0001, kind="symmetric") -> dict:
        """
        Dv01 with respect to each curve: signed price change for a +1bp parallel move of that curve,
        the others held according to 'sticky'.
        Args:
            bump (str): 'quotes', 'zeros' whether to shock the market quotes or the zero rates
            sticky (str): 'quotes', 'zeros', how to propagate the shock among dependent curves ('spread' not valid here)
            shift_size (float): size of the shock
            kind (str): 'oneside' or 'symmetric', how to calculate the finite difference
        Returns:
            dict keyed by curve name.
        """
        if sticky not in ("quotes", "zeros"):
            raise ValueError("Admitted sticky kinds here are: 'quotes', 'zeros'.")
        curves = list(dict.fromkeys((self.discount_curve, *self.projection_curves)))
        return {curve.name: self.sensitivity(curve, bump, sticky, "parallel", shift_size, kind) for curve in curves}

    def _key_rate_dv01(self, curve="discount", bump="zeros", sticky="zeros",
                       shift_size=0.0001, kind="symmetric", nodes=None, via_jacobian=False) -> dict:
        """
        Price change per basis point at each node, one node at a time.
        Args:
            curve (str | YieldCurve): curve to be shocked
            bump (str): 'quotes', 'zeros' whether to shock the market quotes or the zero rates
            sticky (str): 'quotes', 'zeros' or 'spread', how to propagate the shock among dependent curves
            shift_size (float): size of the shock
            kind (str): 'oneside' or 'symmetric', how to calculate the finite difference
            nodes (Iterable): [optional] tenors or dates to shock, defaults to the curve's own nodes.
            via_jacobian (bool): whether to use the Jacobian in calculating the quotes sensitivities; requires
                bump='quotes' and sticky 'quotes' or 'zeros'. With sticky='quotes' the recalibration of the dependent
                curves is included through their dependency Jacobian, with no bootstrap.
        Returns:
            dict keyed by pillar maturity.
        """
        target = self._target(curve)
        if via_jacobian:
            if bump != "quotes" or nodes is not None or sticky == "spread":
                raise ValueError("The Jacobian method is used to translate zero rate risk into market quote risk. "
                                 "Therefore it requires bump='quotes' and the whole curve")
            J = target.jacobian
            zeros = self._key_rate_dv01(curve=curve, bump="zeros", sticky="zeros", shift_size=shift_size, kind=kind)
            s = np.array(list(zeros.values()))
            if sticky == "quotes":
                for other in list(dict.fromkeys((self.discount_curve, *self.projection_curves))):
                    if target in other.sources:
                        s_other = self._key_rate_dv01(curve=other, bump="zeros", sticky="zeros", shift_size=shift_size,
                                                      kind=kind)
                        s = s + np.array(list(s_other.values())) @ other.dependency_jacobian(target)
            s_q = solve_triangular(J.T, s, lower=False)
            return dict(zip(zeros, map(float, s_q)))

        if nodes is None:

            labels = [instrument.maturity for instrument in target.instruments]
            nodes = range(len(labels))
        else:
            labels = list(nodes)
        return {
            label: self.sensitivity(curve=curve, bump=bump, sticky=sticky, node=node, shift_size=shift_size, kind=kind,
                                    ) for label, node in zip(labels, nodes)}

    def key_rate_dv01(self, bump="zeros", sticky="zeros", shift_size=0.0001, kind="symmetric", nodes=None,
                      via_jacobian=False) -> dict:
        """
        Key rate dv01 of each curve: price change per basis point at each node of that curve, one node at a time.
        Args:
            bump (str): 'quotes', 'zeros' whether to shock the market quotes or the zero rates
            sticky (str): 'quotes', 'zeros' or 'spread', how to propagate the shock among dependent curves
            shift_size (float): size of the shock
            kind (str): 'oneside' or 'symmetric', how to calculate the finite difference
            nodes (Iterable): [optional] tenors or dates to shock, defaults to each curve's own nodes.
            via_jacobian (bool): whether to use the Jacobian in calculating the quotes sensitivities; requires
                bump='quotes' and sticky 'quotes' or 'zeros'.
        Returns:
            dict keyed by curve name, each a dict keyed by node.
        """
        curves = list(dict.fromkeys((self.discount_curve, *self.projection_curves)))
        return {curve.name: self._key_rate_dv01(curve, bump, sticky, shift_size, kind, nodes, via_jacobian)
                for curve in curves}

    def total_key_rate_dv01(self, bump="zeros", sticky="zeros", shift_size=0.0001, kind="symmetric", nodes=None,
                            via_jacobian=False) -> dict:
        """
        Key rate dv01 of every curve summed node by node, on the union of the curves' nodes.
        Args:
            bump (str): 'quotes', 'zeros' whether to shock the market quotes or the zero rates
            sticky (str): 'quotes', 'zeros' or 'spread', how to propagate the shock among dependent curves
            shift_size (float): size of the shock
            kind (str): 'oneside' or 'symmetric', how to calculate the finite difference
            nodes (Iterable): [optional] tenors or dates to shock, defaults to each curve's own nodes.
            via_jacobian (bool): whether to use the Jacobian in calculating the quotes sensitivities; requires
                bump='quotes' and sticky 'quotes' or 'zeros'.
        Returns:
            dict keyed by node, sorted by date when nodes is not given.
        """
        total = pd.DataFrame(self.key_rate_dv01(bump, sticky, shift_size, kind, nodes, via_jacobian)).sum(axis=1)
        return total.to_dict() if nodes is not None else total.sort_index().to_dict()

    def modified_duration(self, bump="zeros", sticky="zeros", shift_size=0.0001, kind="symmetric") -> dict:
        """
        Compute the modified duration via a parallel finite difference on each curve,
        normalised by the dirty price.
        Args:
            bump (str): 'zeros' or 'quotes', whether to shock zero rates or market quotes.
            sticky (str): how to propagate the shock: 'quotes' keeps the other curves' market quotes constant and
                            triggers them to re-bootstrap; 'zeros' keeps the other curves' zero rates constant (no
                            bootstrap)
            shift_size (float): parallel shock size for finite difference (default 1bp).
            kind (str): 'symmetric' (centred, more accurate) or 'oneside'.
        Returns:
            dict keyed by curve name: modified duration (years per unit of rate change), positive for a long position.
        """
        if sticky not in ("quotes", "zeros"):
            raise ValueError("Admitted sticky kinds here are: 'quotes', 'zeros'.")
        price = self._dirty_price()
        return {name: -value * 10000 / price for name, value in self.dv01(bump, sticky, shift_size, kind).items()}

    def total_modified_duration(self, bump="zeros", sticky="zeros", shift_size=0.0001, kind="symmetric") -> float:
        """
        Modified duration for a parallel move of every curve together: total_dv01 normalised by the dirty price.
        Args:
            bump (str): 'quotes', 'zeros' whether to shock the market quotes or the zero rates
            sticky (str): 'quotes', 'zeros', how to propagate the shock among dependent curves
            shift_size (float): size of the shock
            kind (str): 'oneside' or 'symmetric', how to calculate the finite difference
        Returns:
            float: positive for a long position.
        """
        return -self.total_dv01(bump, sticky, shift_size, kind) * 10000 / self._dirty_price()
