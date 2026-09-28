"""
Test per l'albero trinomiale di Hull-White e la valutazione bermudana.

L'albero serve dove la decomposizione di Jamshidian non arriva: più date di esercizio e un
prezzo di call che cambia data per data. Lì non esiste una forma chiusa, quindi l'albero è
inchiodato in due modi: deve riprodurre la forma chiusa dove esiste (una call sola), e deve
rispettare le proprietà strutturali che valgono per qualunque reticolo corretto.
"""
import numpy as np
import pandas as pd
import pytest

from QuantGYMM.calendar import Schedule
from QuantGYMM.fixed_income.bonds import CallableBond, FixedRateBond, FloatingRateBond
from QuantGYMM.fixed_income.curves import Deposit, Swap, YieldCurve
from QuantGYMM.fixed_income.indexes import Euribor6M
from QuantGYMM.fixed_income.models import hw_bermudan_bond_value, hw_trinomial_tree
from QuantGYMM.fixed_income.pricers import Pricer
from QuantGYMM.utils import business_days_before

TRADE_DATE = pd.Timestamp("2026-06-30")
MATURITY = pd.Timestamp("2034-06-30")
OIS_DEPOSITS = {"1W": 2.185, "3M": 2.227, "6M": 2.309, "1Y": 2.394}
OIS_SWAPS = {"2Y": 2.412, "3Y": 2.413, "5Y": 2.465, "10Y": 2.688, "30Y": 2.957}


def _curva(t, level=0.03, slope=0.004):
    """Curva di prova crescente, così niente si nasconde dietro una struttura piatta."""
    return np.exp(-(level + slope * np.sqrt(t)) * t)


@pytest.fixture
def ois():
    return YieldCurve([Deposit(tenor, quote / 100, TRADE_DATE) for tenor, quote in OIS_DEPOSITS.items()]
                      + [Swap(tenor, quote / 100, TRADE_DATE, dcc="ACT/360")
                         for tenor, quote in OIS_SWAPS.items()], "OIS", TRADE_DATE)


@pytest.fixture
def pricer(ois):
    return Pricer(ois)


@pytest.fixture
def bond(pricer):
    def build():
        b = FixedRateBond(
            schedule=Schedule(start_date=TRADE_DATE - pd.DateOffset(months=12),
                              end_date=MATURITY, frequency=2, eom=False),
            dcc="ACT/ACT ICMA", face_amount=100.0, coupon_rate=0.045, redemption=100.0)
        b.set_evaluation_date(TRADE_DATE)
        b.set_pricer(pricer)
        return b
    return build


@pytest.fixture
def call_dates(bond):
    payments = pd.DatetimeIndex(bond().schedule.schedule["paymentDate"])
    return payments[(payments > pd.Timestamp("2030-01-01")) & (payments < MATURITY)]


def _callable(underlying, dates, prices=100.0, method="tree", volatility=0.01):
    schedule = pd.Series(prices if not np.isscalar(prices) else [prices] * len(dates), index=dates)
    return CallableBond(underlying, schedule, volatility=volatility, method=method)


# --- il reticolo in sé ------------------------------------------------------

@pytest.mark.parametrize("a, sigma, dt, n", [
    (0.03, 0.010, 0.25, 40), (0.10, 0.020, 0.50, 60),
    (0.05, 0.015, 0.10, 200), (0.02, 0.030, 0.25, 120),
])
def test_le_probabilita_sono_positive_e_sommano_a_uno(a, sigma, dt, n):
    """Il cambio di ramificazione ai bordi esiste proprio per tenerle positive."""
    tree = hw_trinomial_tree(dt, n, a, sigma, _curva(np.arange(n + 1) * dt))
    for probs in tree["probs"]:
        assert probs.min() > 0
        np.testing.assert_allclose(probs.sum(axis=0), 1.0, atol=1e-12)


@pytest.mark.parametrize("a, sigma, dt, n", [
    (0.03, 0.010, 0.25, 40), (0.10, 0.020, 0.50, 60), (0.05, 0.015, 0.10, 200),
])
def test_l_albero_riprezza_la_curva_in_ingresso(a, sigma, dt, n):
    """Secondo stadio della costruzione: alpha è risolto esattamente per questo."""
    discount_factors = _curva(np.arange(n + 1) * dt)
    tree = hw_trinomial_tree(dt, n, a, sigma, discount_factors)
    zero_coupon = np.zeros(n + 1)
    zero_coupon[-1] = 1.0
    assert hw_bermudan_bond_value(tree, dt, zero_coupon) == pytest.approx(discount_factors[-1], abs=1e-12)


def test_un_titolo_senza_call_e_la_somma_dei_flussi_scontati():
    dt, n, a, sigma = 0.5, 20, 0.05, 0.015
    discount_factors = _curva(np.arange(n + 1) * dt)
    tree = hw_trinomial_tree(dt, n, a, sigma, discount_factors)
    cash_flows = np.zeros(n + 1)
    cash_flows[1:] = 2.0
    cash_flows[-1] += 100.0
    exact = (cash_flows[1:] * discount_factors[1:]).sum()
    assert hw_bermudan_bond_value(tree, dt, cash_flows) == pytest.approx(exact, abs=1e-10)


def test_l_albero_si_allarga_quando_il_passo_si_stringe():
    """j_max = ceil(0.184 / (a dt)): dimezzare il passo raddoppia la larghezza, e il costo
    con lei. È il motivo per cui una griglia giornaliera è cara, non solo lenta."""
    wide = hw_trinomial_tree(0.05, 20, 0.03, 0.01, _curva(np.arange(21) * 0.05))
    narrow = hw_trinomial_tree(0.50, 20, 0.03, 0.01, _curva(np.arange(21) * 0.50))
    assert wide["j_max"] > narrow["j_max"]


@pytest.mark.parametrize("a, sigma", [(0.0, 0.01), (0.03, 0.0), (-0.03, 0.01)])
def test_un_parametro_non_positivo_e_rifiutato(a, sigma):
    with pytest.raises(ValueError, match="strictly positive"):
        hw_trinomial_tree(0.5, 4, a, sigma, _curva(np.arange(5) * 0.5))


def test_una_curva_di_lunghezza_sbagliata_e_rifiutata():
    with pytest.raises(ValueError, match="discount_factors"):
        hw_trinomial_tree(0.5, 4, 0.03, 0.01, _curva(np.arange(3) * 0.5))


# --- contro la forma chiusa -------------------------------------------------

def test_una_call_sola_converge_al_valore_di_jamshidian(bond, call_dates):
    """L'unico caso con una risposta esatta. Riprodurla qui è ciò che autorizza a fidarsi
    dell'albero dove la forma chiusa non c'è."""
    chiusa = _callable(bond(), call_dates[:1], method="hw").prices()["optionValue"]
    albero = _callable(bond(), call_dates[:1])._option_value_tree(step_days=7)
    assert albero == pytest.approx(chiusa, rel=2e-3)


def test_raffinare_la_griglia_avvicina_alla_forma_chiusa(bond, call_dates):
    chiusa = _callable(bond(), call_dates[:1], method="hw").prices()["optionValue"]
    albero = _callable(bond(), call_dates[:1])
    grezzo = abs(albero._option_value_tree(step_days=30) / chiusa - 1)
    fine = abs(albero._option_value_tree(step_days=7) / chiusa - 1)
    assert fine < grezzo


# --- proprietà che devono valere senza forma chiusa -------------------------

def test_piu_date_di_esercizio_valgono_solo_di_piu(bond, call_dates):
    """Una bermudana contiene ogni europea al suo interno: il valore non può scendere quando
    si aggiunge una data. È la proprietà che il modello a call singola viola."""
    valori = [_callable(bond(), call_dates[:n]).prices()["optionValue"] for n in (1, 2, 4, len(call_dates))]
    assert valori == sorted(valori)
    assert valori[-1] > valori[0], "il premio bermudano non può essere zero"


def test_il_callable_vale_meno_del_titolo_nudo(bond, call_dates):
    """Il portatore ha venduto un'opzione: il valore resta limitato."""
    sottostante = bond()
    nudo = sottostante.prices()["dirtyPrice"]
    prezzi = _callable(sottostante, call_dates).prices()
    assert prezzi["dirtyPrice"] < nudo
    assert prezzi["optionValue"] > 0
    assert prezzi["dirtyPrice"] == pytest.approx(nudo - prezzi["optionValue"])


def test_uno_strike_piu_alto_rende_l_opzione_piu_economica(bond, call_dates):
    """Il piano di premi decrescenti dell'high yield, che la forma chiusa non sa esprimere."""
    alla_pari = _callable(bond(), call_dates).prices()["optionValue"]
    premi = np.maximum(100.0, 102.5 - 0.5 * np.arange(len(call_dates)))
    sopra_la_pari = _callable(bond(), call_dates, prices=premi).prices()["optionValue"]
    assert sopra_la_pari < alla_pari


def test_piu_volatilita_rende_l_opzione_piu_cara(bond, call_dates):
    quieta = _callable(bond(), call_dates, volatility=0.005).prices()["optionValue"]
    agitata = _callable(bond(), call_dates, volatility=0.02).prices()["optionValue"]
    assert agitata > quieta


def test_lo_spread_del_sottostante_entra_nell_albero(bond, call_dates):
    liscio = _callable(bond(), call_dates).prices()
    sottostante = bond()
    sottostante.z_spread = 0.02
    con_spread = _callable(sottostante, call_dates).prices()
    assert con_spread["dirtyPrice"] < liscio["dirtyPrice"]
    assert con_spread["optionValue"] < liscio["optionValue"]


# --- perimetro --------------------------------------------------------------

def test_un_sottostante_variabile_e_rifiutato(ois, pricer, call_dates):
    schedule = Schedule(start_date=TRADE_DATE - pd.DateOffset(months=12), end_date=MATURITY, frequency=2, eom=False)
    resets = pd.DatetimeIndex(business_days_before(schedule.schedule["startingDate"], 2))
    index = Euribor6M(projection_curve=ois, fixings=pd.Series(0.024, index=resets))
    floater = FloatingRateBond(schedule=schedule, index=index, dcc="ACT/360", face_amount=100.0,
                               spread=0.005)
    floater.set_evaluation_date(TRADE_DATE)
    floater.set_pricer(pricer)
    with pytest.raises(NotImplementedError, match="fixed rate"):
        _callable(floater, call_dates[:1]).prices()


def test_la_forma_chiusa_rifiuta_una_bermudana_e_indica_l_albero(bond, call_dates):
    """Jamshidian prezza una data di esercizio. Rifiutare è corretto; sbagliato sarebbe
    ripiegare in silenzio su un altro modello."""
    with pytest.raises(NotImplementedError, match="method='tree'"):
        _callable(bond(), call_dates[:3], method="hw").prices()


def test_un_metodo_ignoto_e_rifiutato(bond, call_dates):
    with pytest.raises(ValueError, match="worst"):
        _callable(bond(), call_dates[:1], method="binomial")
