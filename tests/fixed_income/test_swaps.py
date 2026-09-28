"""VanillaSwap: multicurva, coerente col bootstrap e con QuantLib.

I valori attesi di QuantLib 1.43 vengono da ql.VanillaSwap con DiscountingSwapEngine
sulla OIS e Euribor6M proiettato sulla curva Euribor, costruite sugli stessi
pilastri (fattori di sconto log-lineari, ACT/365) delle curve di questo file.
"""

import pandas as pd
import pytest

from QuantGYMM.calendar import Schedule
from QuantGYMM.fixed_income.curves import Deposit, ForwardRateAgreement, Swap, YieldCurve
from QuantGYMM.fixed_income.indexes import Euribor6M
from QuantGYMM.fixed_income.pricers import Pricer
from QuantGYMM.fixed_income.swaps import VanillaSwap
from QuantGYMM.utils import business_days_after

TRADE_DATE = pd.Timestamp("2026-06-30")
SPOT = business_days_after(TRADE_DATE, 2)
NOMINALE = 1_000_000.0
OIS_DEPOSITS = {"1W": 2.185, "3M": 2.227, "6M": 2.309, "1Y": 2.394}
OIS_SWAPS = {"2Y": 2.412, "3Y": 2.413, "5Y": 2.465, "10Y": 2.688, "30Y": 2.957}
EURIBOR_FRAS = {("3M", "9M"): 2.667, ("6M", "12M"): 2.719, ("12M", "18M"): 2.704}
EURIBOR_SWAPS = {"2Y": 2.711, "3Y": 2.704, "5Y": 2.734, "10Y": 2.914, "30Y": 3.106}
STORICO = pd.Series(0.021, index=pd.bdate_range("2025-01-01", TRADE_DATE))


@pytest.fixture
def ois():
    return YieldCurve([Deposit(tenor, quote / 100, TRADE_DATE) for tenor, quote in OIS_DEPOSITS.items()]
                      + [Swap(tenor, quote / 100, TRADE_DATE, dcc="ACT/360") for tenor, quote in OIS_SWAPS.items()],
                      "OIS", TRADE_DATE)


@pytest.fixture
def quotati(ois):
    return [Swap(tenor, quote / 100, TRADE_DATE, frequency=1, float_frequency=2, dcc="30/360", discount_curve=ois)
            for tenor, quote in EURIBOR_SWAPS.items()]


@pytest.fixture
def euribor(ois, quotati):
    instruments = [Deposit("6M", 0.02568, TRADE_DATE)]
    instruments += [ForwardRateAgreement(start, end, quote / 100, TRADE_DATE)
                    for (start, end), quote in EURIBOR_FRAS.items()]
    return YieldCurve(instruments + quotati, "EURIBOR6M", TRADE_DATE)


@pytest.fixture
def swap(ois, euribor):
    def costruisci(scadenza, tasso, inizio=SPOT, fixings=None, spread=0.0, side="payer"):
        indice = Euribor6M(projection_curve=euribor, fixings=fixings)
        contratto = VanillaSwap(Schedule(inizio, scadenza, 1), tasso, Schedule(inizio, scadenza, 2), indice,
                                NOMINALE, spread=spread, side=side)
        contratto.set_evaluation_date(TRADE_DATE)
        contratto.set_pricer(Pricer(ois))
        return contratto
    return costruisci


class TestCoerenzaColBootstrap:

    @pytest.mark.parametrize("posizione", range(len(EURIBOR_SWAPS)), ids=list(EURIBOR_SWAPS))
    def test_lo_swap_della_curva_al_tasso_quotato_vale_zero(self, swap, quotati, posizione):
        quotato = quotati[posizione]
        contratto = swap(quotato.maturity, quotato.quote)
        assert contratto.npv() == pytest.approx(0.0, abs=1e-6)
        assert contratto.par_rate() == pytest.approx(quotato.quote, abs=1e-14)


class TestQuantLib:

    @pytest.mark.parametrize("scadenza, tasso, inizio, fixings, spread, npv, par", [
        ("2031-07-02", 0.0275, SPOT, None, 0.0, -743.80670927951, 0.027339999999999982),
        ("2036-07-02", 0.029, SPOT, None, 0.0015, 14544.780889458169, 0.030672405103143324),
        ("2033-07-04", 0.028, SPOT, STORICO, 0.0, 2327.313506016275, 0.02836681899331036),
        ("2032-09-15", 0.0255, "2025-09-15", STORICO, 0.001, 5690.902506396902, 0.02637926754859093),
    ], ids=["spot 5Y", "spot 10Y con spread", "fixing di oggi pubblicato", "avviato con fixing storici"])
    def test_npv_e_tasso_par(self, swap, scadenza, tasso, inizio, fixings, spread, npv, par):
        contratto = swap(scadenza, tasso, inizio=inizio, fixings=fixings, spread=spread)
        assert contratto.npv() == pytest.approx(npv, abs=1e-8)
        assert contratto.par_rate() == pytest.approx(par, abs=1e-15)


class TestFixing:

    def test_il_fixing_di_oggi_pubblicato_si_usa(self, swap):
        assert swap("2031-07-02", 0.0275, fixings=STORICO).floating_leg.resetRate.iloc[0] == 0.021

    def test_il_fixing_di_oggi_non_pubblicato_si_stima_sul_periodo_della_cedola(self, swap, euribor):
        cedola = swap("2031-07-02", 0.0275).floating_leg.iloc[0]
        df = euribor.discount_factor_at(pd.DatetimeIndex([cedola.accrualStart, cedola.accrualEnd]))
        atteso = (df[0] / df[1] - 1) / ((cedola.accrualEnd - cedola.accrualStart).days / 360)
        assert cedola.resetRate == pytest.approx(atteso, abs=1e-15)

    def test_un_fixing_passato_mancante_solleva(self, swap):
        with pytest.raises(KeyError):
            swap("2032-09-15", 0.0255, inizio="2025-09-15").npv()


class TestRischio:

    def test_il_receiver_e_l_opposto_del_payer(self, swap):
        payer, receiver = swap("2036-07-02", 0.029), swap("2036-07-02", 0.029, side="receiver")
        assert receiver.npv() == -payer.npv()
        assert receiver.dv01() == pytest.approx({curva: -valore for curva, valore in payer.dv01().items()})

    def test_i_key_rate_sommano_al_dv01_di_ogni_curva(self, swap):
        contratto = swap("2036-07-02", 0.029)
        dv01 = contratto.dv01()
        for curva, nodi in contratto.key_rate_dv01().items():
            assert sum(nodi.values()) == pytest.approx(dv01[curva], rel=1e-6)

    def test_un_payer_guadagna_se_la_curva_euribor_sale(self, swap):
        assert swap("2036-07-02", 0.029).dv01()["EURIBOR6M"] > 0


class TestValidazione:

    def test_il_verso_e_payer_o_receiver(self, swap):
        with pytest.raises(ValueError):
            swap("2031-07-02", 0.0275, side="long")

    def test_una_curva_di_un_altra_data_solleva(self, swap):
        contratto = swap("2031-07-02", 0.0275)
        contratto.set_evaluation_date("2026-07-01")
        with pytest.raises(ValueError):
            contratto.npv()
