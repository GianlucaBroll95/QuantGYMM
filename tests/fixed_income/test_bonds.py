"""
Test per il livello multicurva degli strumenti: RateRisk e le quattro classi di bonds.py.
"""
from contextlib import ExitStack

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import solve_triangular

from QuantGYMM.calendar import Schedule
from QuantGYMM.fixed_income.bonds import (Bond, BondPortfolio, CallableBond, FixedRateBond, FloatingRateBond,
                                          RateRisk, ZeroCouponBond)
from QuantGYMM.fixed_income.curves import Deposit, ForwardRateAgreement, Swap, YieldCurve
from QuantGYMM.fixed_income.indexes import Euribor3M, Euribor6M
from QuantGYMM.fixed_income.pricers import (BachelierCouponPricer, BlackCouponPricer, DisplacedBlackCouponPricer,
                                            Pricer)
from QuantGYMM.utils import accrual_factor, business_days_after, business_days_before

TRADE_DATE = pd.Timestamp("2026-06-30")
UN_BP = 0.0001

OIS_DEPOSITS = {"1W": 2.185, "3M": 2.227, "6M": 2.309, "1Y": 2.394}
OIS_SWAPS = {"2Y": 2.412, "3Y": 2.413, "5Y": 2.465, "10Y": 2.688, "30Y": 2.957}
EURIBOR_FRAS = {("3M", "9M"): 2.667, ("6M", "12M"): 2.719, ("12M", "18M"): 2.704}
EURIBOR_SWAPS = {"2Y": 2.711, "3Y": 2.704, "5Y": 2.734, "10Y": 2.914, "30Y": 3.106}


@pytest.fixture
def ois():
    return YieldCurve([Deposit(tenor, quote / 100, TRADE_DATE) for tenor, quote in OIS_DEPOSITS.items()]
                      + [Swap(tenor, quote / 100, TRADE_DATE, dcc="ACT/360")
                         for tenor, quote in OIS_SWAPS.items()], "OIS", TRADE_DATE)


@pytest.fixture
def euribor(ois):
    instruments = [Deposit("6M", 0.02568, TRADE_DATE)]
    instruments += [ForwardRateAgreement(start, end, quote / 100, TRADE_DATE)
                    for (start, end), quote in EURIBOR_FRAS.items()]
    instruments += [Swap(tenor, quote / 100, TRADE_DATE, frequency=1, float_frequency=2,
                         dcc="30/360", discount_curve=ois) for tenor, quote in EURIBOR_SWAPS.items()]
    return YieldCurve(instruments, "EURIBOR6M", TRADE_DATE)


@pytest.fixture
def fixed_bond(ois):
    bond = FixedRateBond(schedule=Schedule("2024-01-15", "2034-01-15", 1), dcc="ACT/ACT ICMA",
                         face_amount=1_000_000, coupon_rate=0.0325)
    bond.set_evaluation_date(TRADE_DATE)
    bond.set_pricer(Pricer(ois))
    return bond


@pytest.fixture
def zero_bond(ois):
    bond = ZeroCouponBond(maturity_date="2034-01-15", face_amount=1_000_000)
    bond.set_evaluation_date(TRADE_DATE)
    bond.set_pricer(Pricer(ois))
    return bond


@pytest.fixture
def floater(ois, euribor):
    schedule = Schedule("2024-01-15", "2032-01-15", 2)
    resets = pd.DatetimeIndex(business_days_before(schedule.schedule["startingDate"], 2))
    index = Euribor6M(projection_curve=euribor, fixings=pd.Series(0.03, index=resets))
    bond = FloatingRateBond(schedule=schedule, index=index, dcc="ACT/360", face_amount=1_000_000,
                            spread=0.0045)
    bond.set_evaluation_date(TRADE_DATE)
    bond.set_pricer(Pricer(ois))
    return bond


@pytest.fixture
def callable_bond(fixed_bond):
    dates = [date for date in fixed_bond.schedule.schedule["paymentDate"]
             if date > pd.Timestamp("2028-01-01")]
    return CallableBond(fixed_bond, pd.Series(100.0, index=pd.DatetimeIndex(dates)))


def dirty(bond):
    return bond._dirty_price()


def prezzo_al_regolamento(bond):
    return bond.prices()["dirtyPrice"]


# ---------------------------------------------------------------------------
# Il motore di rischio: chi lo ospita e cosa rifiuta
# ---------------------------------------------------------------------------

class TestRateRisk:

    def test_e_montato_su_bond_e_su_callable(self):
        assert issubclass(Bond, RateRisk)
        assert issubclass(CallableBond, RateRisk)

    def test_il_callable_non_e_un_bond(self):
        assert not issubclass(CallableBond, Bond)

    @pytest.mark.parametrize("shift_type", ["slope", "curvature"])
    def test_le_forme_non_parallele_hanno_un_valore_per_nodo(self, shift_type):
        assert len(RateRisk._shift_shape(shift_type, UN_BP, 7)) == 7

    def test_il_parallelo_resta_uno_scalare(self):
        assert RateRisk._shift_shape("parallel", UN_BP, 5) == UN_BP

    def test_la_pendenza_parte_positiva_e_finisce_negativa(self):
        shift = RateRisk._shift_shape("slope", UN_BP, 5)
        assert shift[0] == pytest.approx(UN_BP)
        assert shift[-1] == pytest.approx(-UN_BP)

    def test_la_curvatura_e_simmetrica(self):
        shift = RateRisk._shift_shape("curvature", UN_BP, 7)
        assert shift == pytest.approx(shift[::-1])

    def test_una_forma_ignota_solleva(self):
        with pytest.raises(ValueError):
            RateRisk._shift_shape("boh", UN_BP, 5)

    @pytest.mark.parametrize("kwargs", [
        {"bump": "boh"}, {"sticky": "boh"}, {"kind": "boh"},
    ])
    def test_i_valori_ignoti_sollevano(self, fixed_bond, kwargs):
        with pytest.raises(ValueError):
            fixed_bond.sensitivity(**kwargs)

    def test_sticky_spread_con_un_indice_solleva(self, fixed_bond):
        with pytest.raises(ValueError):
            fixed_bond.sensitivity(sticky="spread", node=2)

    def test_sticky_spread_su_un_nodo_vuole_i_tassi_zero(self, floater):
        assert floater.sensitivity(sticky="spread", bump="zeros", node="5Y") != 0.0
        with pytest.raises(ValueError):
            floater.sensitivity(sticky="spread", bump="quotes", node="5Y")

    def test_sticky_spread_parallelo_passa_con_entrambi(self, floater):
        assert floater.sensitivity(sticky="spread", bump="quotes") != 0.0
        assert floater.sensitivity(sticky="spread", bump="zeros") != 0.0

    @pytest.mark.parametrize("shift_type", ["slope", "curvature"])
    def test_sticky_spread_su_tutta_la_curva_vuole_il_parallelo(self, floater, shift_type):
        with pytest.raises(ValueError):
            floater.sensitivity(sticky="spread", shift_type=shift_type)
        assert floater.sensitivity(sticky="quotes", shift_type=shift_type) != 0.0

    @pytest.mark.parametrize("bump", ["quotes", "zeros"])
    def test_sticky_spread_con_curve_di_lunghezza_diversa(self, bump):
        ois_corta = YieldCurve([Deposit(tenor, quote / 100, TRADE_DATE) for tenor, quote in OIS_DEPOSITS.items()]
                               + [Swap(tenor, quote / 100, TRADE_DATE, dcc="ACT/360")
                                  for tenor, quote in list(OIS_SWAPS.items())[:-1]], "OIS corta", TRADE_DATE)
        eur = euribor.__wrapped__(ois_corta)
        bond = floater.__wrapped__(ois_corta, eur)
        assert len(ois_corta.instruments) != len(eur.instruments)

        def shocked(size):
            with ois_corta.shocked_quotes(size) if bump == "quotes" else ois_corta.shocked_zeros(size):
                with eur.shocked_quotes(size) if bump == "quotes" else eur.shocked_zeros(size):
                    return dirty(bond)

        atteso = (shocked(UN_BP) - shocked(-UN_BP)) / (2 * UN_BP) * UN_BP
        assert bond.sensitivity(sticky="spread", bump=bump) == pytest.approx(atteso)

    @pytest.mark.parametrize("sticky, bump", [("zeros", "quotes"), ("spread", "zeros")])
    def test_il_risultato_non_dipende_da_cosa_si_e_calcolato_prima(self, ois, euribor, floater, sticky, bump):
        fresco = floater.sensitivity(sticky=sticky, bump=bump)
        floater.sensitivity(curve="discount", sticky="quotes", node=5)
        assert [source.pillars for source in euribor.sources] != euribor._source_pillars
        assert floater.sensitivity(sticky=sticky, bump=bump) == pytest.approx(fresco, rel=1e-12)

    @pytest.mark.parametrize("alias, curva", [("discount", "ois"), ("projection", "euribor")])
    def test_la_curva_passata_come_oggetto_vale_quanto_l_alias(self, floater, ois, euribor, alias, curva):
        istanza = {"ois": ois, "euribor": euribor}[curva]
        assert floater.sensitivity(curve=istanza) == floater.sensitivity(curve=alias)
        assert floater._key_rate_dv01(curve=istanza, nodes=["2Y", "5Y"], bump="zeros") ==                floater._key_rate_dv01(curve=alias, nodes=["2Y", "5Y"], bump="zeros")

    def test_una_curva_estranea_solleva(self, fixed_bond, euribor):
        with pytest.raises(ValueError):
            fixed_bond.sensitivity(curve=euribor)

    def test_un_nome_di_curva_ignoto_solleva(self, fixed_bond):
        with pytest.raises(ValueError):
            fixed_bond.sensitivity(curve="boh")

    def test_le_altre_curve_sono_tutte_meno_il_bersaglio(self, floater, ois, euribor, monkeypatch):
        visto = {}

        def spia(target, others, *args):
            visto[target] = list(others)
            return ExitStack()

        monkeypatch.setattr(RateRisk, "_shocked", staticmethod(spia))
        floater.sensitivity(curve=ois, kind="oneside")
        floater.sensitivity(curve=euribor, kind="oneside")
        assert visto == {ois: [euribor], euribor: [ois]}

    def test_con_la_stessa_curva_nei_due_ruoli_le_altre_sono_vuote(self, ois, monkeypatch):
        schedule = Schedule("2024-01-15", "2032-01-15", 2)
        resets = pd.DatetimeIndex(business_days_before(schedule.schedule["startingDate"], 2))
        index = Euribor6M(projection_curve=ois, fixings=pd.Series(0.03, index=resets))
        bond = FloatingRateBond(schedule=schedule, index=index, dcc="ACT/360",
                                face_amount=1_000_000)
        bond.set_evaluation_date(TRADE_DATE)
        bond.set_pricer(Pricer(ois))
        visto = {}

        def spia(target, others, *args):
            visto[target] = list(others)
            return ExitStack()

        monkeypatch.setattr(RateRisk, "_shocked", staticmethod(spia))
        bond.sensitivity(curve="discount", kind="oneside")
        assert visto == {ois: []}

    def test_senza_curva_di_proiezione_solleva(self, fixed_bond):
        with pytest.raises(ValueError):
            fixed_bond.sensitivity(curve="projection")

    def test_una_curva_sola_per_entrambi_i_ruoli_solleva(self, ois):
        schedule = Schedule("2024-01-15", "2032-01-15", 2)
        resets = pd.DatetimeIndex(business_days_before(schedule.schedule["startingDate"], 2))
        index = Euribor6M(projection_curve=ois, fixings=pd.Series(0.03, index=resets))
        bond = FloatingRateBond(schedule=schedule, index=index, dcc="ACT/360",
                                face_amount=1_000_000)
        bond.set_evaluation_date(TRADE_DATE)
        bond.set_pricer(Pricer(ois))
        with pytest.raises(ValueError):
            bond.sensitivity(curve="projection")

    def test_lo_shock_non_lascia_traccia_sulla_curva(self, fixed_bond, ois):
        prima = list(ois._log_df), [instrument.quote for instrument in ois.instruments]
        fixed_bond.sensitivity()
        fixed_bond.key_rate_dv01()
        assert (list(ois._log_df), [instrument.quote for instrument in ois.instruments]) == prima


# ---------------------------------------------------------------------------
# La base
# ---------------------------------------------------------------------------

class TestBond:

    def test_senza_pricer_solleva(self):
        bond = ZeroCouponBond(maturity_date="2034-01-15", face_amount=100.0)
        bond.set_evaluation_date(TRADE_DATE)
        with pytest.raises(ValueError):
            bond.prices()

    def test_senza_data_di_valutazione_solleva(self, ois):
        bond = ZeroCouponBond(maturity_date="2034-01-15", face_amount=100.0)
        bond.set_pricer(Pricer(ois))
        with pytest.raises(ValueError):
            bond.cash_flows

    def test_un_titolo_senza_indice_non_ha_curva_di_proiezione(self, fixed_bond):
        assert fixed_bond.projection_curves == []

    def test_il_prezzo_e_la_somma_dei_flussi_scontati(self, fixed_bond, ois):
        flussi = fixed_bond.cash_flows
        atteso = flussi.cashFlow.to_numpy().dot(ois.discount_factor_at(flussi.paymentDate))
        assert dirty(fixed_bond) == pytest.approx(atteso)

    def test_la_cache_dei_flussi_e_riusata(self, fixed_bond):
        assert fixed_bond.cash_flows is fixed_bond.cash_flows

    def test_la_duration_e_la_media_pesata_dei_tempi(self, fixed_bond, ois):
        flussi = fixed_bond.cash_flows
        df = ois.discount_factor_at(flussi.paymentDate)
        t = accrual_factor(ois.dcc, TRADE_DATE, flussi.paymentDate)
        cf = flussi.cashFlow.to_numpy()
        assert fixed_bond.duration() == pytest.approx((t * cf * df).sum() / cf.dot(df))

    def test_la_modified_duration_e_la_dv01_normalizzata_per_curva(self, fixed_bond):
        atteso = -fixed_bond.sensitivity(bump="zeros", sticky="zeros") * 10000 / dirty(fixed_bond)
        assert fixed_bond.modified_duration() == {"OIS": pytest.approx(atteso)}

    def test_la_modified_duration_del_floater_ha_una_voce_per_curva(self, floater):
        assert set(floater.modified_duration()) == {"OIS", "EURIBOR6M"}

    def test_il_prezzo_da_derivare_e_quello_aggiustato_se_c_e_il_cds(self, fixed_bond):
        liscio = dirty(fixed_bond)
        fixed_bond.set_cds_spread(0.01)
        fixed_bond.set_recovery_rate(0.4)
        assert dirty(fixed_bond) == pytest.approx(fixed_bond.pricer.present_value(fixed_bond))
        assert dirty(fixed_bond) < liscio

    def test_il_prezzo_e_il_valore_al_regolamento_dei_flussi_successivi(self, fixed_bond, ois):
        regolamento = business_days_after(TRADE_DATE, 2)
        flussi = fixed_bond.cash_flows[fixed_bond.cash_flows.paymentDate > regolamento]
        df = ois.discount_factor_at(flussi.paymentDate) / ois.discount_factor_at(regolamento)
        assert prezzo_al_regolamento(fixed_bond) == pytest.approx(flussi.cashFlow.to_numpy().dot(df), rel=1e-12)

    def test_una_cedola_fra_valutazione_e_regolamento_sta_nel_valore_ma_non_nel_prezzo(self, ois):
        bond = FixedRateBond(schedule=Schedule("2025-07-01", "2030-07-01", 1), dcc="ACT/ACT ICMA",
                             face_amount=1_000_000, coupon_rate=0.04)
        bond.set_evaluation_date(TRADE_DATE)
        bond.set_pricer(Pricer(ois))
        cedola = bond.cash_flows.iloc[0]
        assert TRADE_DATE < cedola.paymentDate <= bond.settlement_date
        regolamento = bond.settlement_date
        df_cedola = ois.discount_factor_at(cedola.paymentDate)
        df_regolamento = ois.discount_factor_at(regolamento)
        atteso = (dirty(bond) - cedola.cashFlow * df_cedola) / df_regolamento
        assert prezzo_al_regolamento(bond) == pytest.approx(atteso, rel=1e-12)
        prossima = bond.cash_flows.iloc[1]
        rateo = prossima.coupon * (regolamento - prossima.accrualStart).days / (prossima.accrualEnd - prossima.accrualStart).days
        assert bond.accrued_interest() == pytest.approx(rateo)
        assert prezzo_a_yield(bond, bond.ytm()) == pytest.approx(prezzo_al_regolamento(bond), rel=1e-10)

    def test_con_il_cds_il_prezzo_e_condizionato_alla_sopravvivenza_al_regolamento(self, fixed_bond):
        fixed_bond.set_cds_spread(0.01)
        fixed_bond.set_recovery_rate(0.4)
        regolamento = fixed_bond.settlement_date
        df_regolamento = fixed_bond.pricer.discount_factor_at(fixed_bond, [regolamento])[0]
        q_regolamento = fixed_bond._survival_at(regolamento)[0]
        df_primo = fixed_bond.pricer.discount_factor_at(fixed_bond, fixed_bond.cash_flows.paymentDate)[0]
        default_prima_del_regolamento = 0.4 * fixed_bond.face_amount * df_primo * (1 - q_regolamento)
        atteso = (dirty(fixed_bond) - default_prima_del_regolamento) / (df_regolamento * q_regolamento)
        assert prezzo_al_regolamento(fixed_bond) == pytest.approx(atteso, rel=1e-12)

    def test_una_curva_di_un_altra_data_solleva(self, fixed_bond):
        fixed_bond.set_evaluation_date("2026-07-01")
        with pytest.raises(ValueError):
            fixed_bond.prices()

    def test_con_il_cds_ma_senza_recovery_solleva(self, fixed_bond):
        fixed_bond.set_cds_spread(0.01)
        with pytest.raises(ValueError):
            fixed_bond.prices()

    def test_il_rateo_e_la_frazione_di_cedola_maturata(self, fixed_bond):
        flussi = fixed_bond.cash_flows
        inizio, fine = flussi.accrualStart.iloc[0], flussi.accrualEnd.iloc[0]
        regolamento = business_days_after(TRADE_DATE, 2)
        atteso = flussi.coupon.iloc[0] * (regolamento - inizio).days / (fine - inizio).days
        prezzi = fixed_bond.prices()
        assert prezzi["accruedInterest"] == pytest.approx(atteso)
        assert prezzi["cleanPrice"] == pytest.approx(prezzi["dirtyPrice"] - atteso)

    def test_lo_spread_zero_non_cambia_il_prezzo(self, fixed_bond):
        prima = dirty(fixed_bond)
        fixed_bond.z_spread = 0.0
        assert dirty(fixed_bond) == prima

    def test_lo_spread_abbassa_il_prezzo_e_lo_vede_la_duration(self, fixed_bond):
        prezzo, durata = dirty(fixed_bond), fixed_bond.duration()
        fixed_bond.z_spread = 0.01
        assert dirty(fixed_bond) < prezzo
        assert fixed_bond.duration() < durata

    def test_lo_spread_non_si_muove_sotto_uno_shock(self, fixed_bond):
        fixed_bond.z_spread = 0.005
        fixed_bond.key_rate_dv01(nodes=[1, 3])
        fixed_bond.sensitivity(bump="zeros", sticky="spread")
        assert fixed_bond.z_spread == 0.005

    def test_un_pricer_solo_serve_due_titoli_con_spread_diversi(self, fixed_bond, ois):
        gemello = FixedRateBond(schedule=fixed_bond.schedule, dcc="ACT/ACT ICMA", face_amount=1_000_000,
                                coupon_rate=0.0325)
        gemello.z_spread = 0.01
        gemello.set_evaluation_date(TRADE_DATE)
        gemello.set_pricer(fixed_bond.pricer)
        assert gemello.pricer is fixed_bond.pricer
        assert dirty(gemello) < dirty(fixed_bond)


def prezzo_a_yield(bond, y):
    regolamento = business_days_after(TRADE_DATE, 2)
    flussi = bond.cash_flows[bond.cash_flows.paymentDate > regolamento]
    t = accrual_factor(bond.discount_curve.dcc, regolamento, flussi.paymentDate)
    return flussi.cashFlow.to_numpy().dot((1 + y) ** -t)


class TestYieldToMaturity:

    @pytest.mark.parametrize("titolo", ["fixed_bond", "zero_bond", "floater"])
    def test_lo_ytm_riprezza_il_dirty_del_modello(self, titolo, request):
        bond = request.getfixturevalue(titolo)
        assert prezzo_a_yield(bond, bond.ytm()) == pytest.approx(prezzo_al_regolamento(bond), rel=1e-10)

    def test_lo_ytm_dello_zero_e_in_forma_chiusa(self, zero_bond):
        regolamento = business_days_after(TRADE_DATE, 2)
        t = accrual_factor(zero_bond.discount_curve.dcc, regolamento, pd.Timestamp("2034-01-15")).item()
        atteso = (zero_bond.face_amount / prezzo_al_regolamento(zero_bond)) ** (1 / t) - 1
        assert zero_bond.ytm() == pytest.approx(atteso, rel=1e-10)

    def test_il_prezzo_di_mercato_e_clean_per_100(self, fixed_bond):
        clean = fixed_bond.prices()["cleanPrice"] / fixed_bond.face_amount * 100
        assert fixed_bond.ytm(clean_price=clean) == pytest.approx(fixed_bond.ytm(), rel=1e-10)

    def test_lo_ytm_a_prezzo_di_mercato_non_dipende_dalla_curva(self, fixed_bond):
        y = fixed_bond.ytm(clean_price=95.0)
        fixed_bond.z_spread = 0.02
        assert fixed_bond.ytm(clean_price=95.0) == y

    def test_lo_ytm_scende_se_il_prezzo_sale(self, fixed_bond):
        assert fixed_bond.ytm(clean_price=101.0) < fixed_bond.ytm(clean_price=99.0)

    def test_la_ytm_duration_e_la_derivata_del_prezzo_rispetto_allo_ytm(self, fixed_bond):
        y, h = fixed_bond.ytm(), 1e-6
        derivata = (prezzo_a_yield(fixed_bond, y + h) - prezzo_a_yield(fixed_bond, y - h)) / (2 * h)
        assert fixed_bond.ytm_duration() == pytest.approx(-derivata / prezzo_al_regolamento(fixed_bond), rel=1e-6)

    def test_la_ytm_duration_dello_zero_e_la_scadenza_scontata_di_uno_piu_y(self, zero_bond):
        regolamento = business_days_after(TRADE_DATE, 2)
        t = accrual_factor(zero_bond.discount_curve.dcc, regolamento, pd.Timestamp("2034-01-15")).item()
        assert zero_bond.ytm_duration() == pytest.approx(t / (1 + zero_bond.ytm()), rel=1e-10)

    def test_la_ytm_duration_e_sotto_la_macaulay(self, fixed_bond):
        assert fixed_bond.ytm_duration() < fixed_bond.duration()

    def test_la_ytm_duration_usa_il_prezzo_passato(self, fixed_bond):
        assert fixed_bond.ytm_duration(clean_price=90.0) != fixed_bond.ytm_duration()

    def test_la_ytm_duration_del_variabile_solleva(self, floater):
        with pytest.raises(NotImplementedError):
            floater.ytm_duration()


# ---------------------------------------------------------------------------
# La DV01 parallela, una voce per curva
# ---------------------------------------------------------------------------

class TestDv01:

    def test_una_voce_per_curva_con_il_nome(self, fixed_bond, floater, portfolio):
        assert list(fixed_bond.dv01()) == ["OIS"]
        assert list(floater.dv01()) == ["OIS", "EURIBOR6M"]
        assert list(portfolio.dv01()) == ["OIS", "EURIBOR6M", "EURIBOR3M"]

    def test_e_la_sensitivity_parallela_sugli_zero_rate(self, floater, ois, euribor):
        atteso = {"OIS": floater.sensitivity(ois, bump="zeros", sticky="zeros"),
                  "EURIBOR6M": floater.sensitivity(euribor, bump="zeros", sticky="zeros")}
        assert floater.dv01() == pytest.approx(atteso)

    def test_negativa_sullo_sconto_positiva_sulla_proiezione(self, floater):
        dv01 = floater.dv01()
        assert dv01["OIS"] < 0 < dv01["EURIBOR6M"]

    def test_la_somma_e_lo_shock_di_tutte_le_curve_insieme(self, floater):
        insieme = floater.sensitivity(bump="zeros", sticky="spread")
        assert sum(floater.dv01().values()) == pytest.approx(insieme, abs=1e-3)

    def test_e_la_somma_dei_key_rate(self, floater):
        dv01 = floater.dv01()
        for nome, krd in floater.key_rate_dv01().items():
            assert sum(krd.values()) == pytest.approx(dv01[nome], rel=1e-5)

    def test_in_quotazioni_e_un_altro_numero(self, fixed_bond):
        assert fixed_bond.dv01(bump="quotes", sticky="quotes")["OIS"] != pytest.approx(fixed_bond.dv01()["OIS"],
                                                                                        rel=1e-6)

    def test_sul_portafoglio_e_la_somma_dei_titoli(self, portfolio):
        somma = pd.DataFrame([bond.dv01() for bond in portfolio.bonds]).sum().to_dict()
        assert portfolio.dv01() == pytest.approx(somma, rel=1e-6)

    def test_la_modified_duration_e_la_dv01_normalizzata(self, floater):
        prezzo = dirty(floater)
        atteso = {name: -value * 10000 / prezzo for name, value in floater.dv01().items()}
        assert floater.modified_duration() == pytest.approx(atteso)

    def test_non_lascia_traccia(self, portfolio, ois, euribor, euribor3m):
        prima = ois.pillars, euribor.pillars, euribor3m.pillars
        portfolio.dv01()
        assert (ois.pillars, euribor.pillars, euribor3m.pillars) == prima

    @pytest.mark.parametrize("metodo", ["dv01", "total_dv01", "modified_duration", "total_modified_duration"])
    def test_sticky_spread_non_e_ammesso(self, floater, metodo):
        with pytest.raises(ValueError):
            getattr(floater, metodo)(sticky="spread")

    @pytest.mark.parametrize("bump", ["zeros", "quotes"])
    def test_il_totale_e_la_somma_delle_curve(self, floater, bump):
        atteso = sum(floater.dv01(bump=bump, sticky=bump).values())
        assert floater.total_dv01(bump=bump, sticky=bump) == pytest.approx(atteso, rel=1e-12)

    def test_il_totale_e_lo_shock_di_tutte_le_curve_insieme(self, floater):
        insieme = floater.sensitivity(bump="zeros", sticky="spread")
        assert floater.total_dv01() == pytest.approx(insieme, abs=1e-3)

    def test_la_duration_totale_e_la_somma_delle_duration(self, floater):
        atteso = sum(floater.modified_duration().values())
        assert floater.total_modified_duration() == pytest.approx(atteso, rel=1e-12)

    def test_sul_variabile_la_duration_totale_e_quasi_nulla(self, floater):
        parti = floater.modified_duration()
        assert abs(floater.total_modified_duration()) < 0.1 * parti["OIS"]


# ---------------------------------------------------------------------------
# Key rate: una serie per curva, il totale in una serie sola
# ---------------------------------------------------------------------------

class TestKeyRateDv01:

    @pytest.fixture
    def senza_opzioni(self, fixed_bond, zero_bond, floater, floater3m):
        return BondPortfolio([fixed_bond, zero_bond, floater, floater3m])

    def test_una_serie_per_curva_con_il_nome(self, fixed_bond, floater, senza_opzioni):
        assert list(fixed_bond.key_rate_dv01()) == ["OIS"]
        assert list(floater.key_rate_dv01()) == ["OIS", "EURIBOR6M"]
        assert list(senza_opzioni.key_rate_dv01()) == ["OIS", "EURIBOR6M", "EURIBOR3M"]

    def test_ogni_serie_ha_i_nodi_della_sua_curva(self, floater, ois, euribor):
        krd = floater.key_rate_dv01()
        assert list(krd["OIS"]) == [instrument.maturity for instrument in ois.instruments]
        assert list(krd["EURIBOR6M"]) == [instrument.maturity for instrument in euribor.instruments]

    def test_il_totale_somma_le_curve_nodo_per_nodo(self, floater):
        krd, totale = floater.key_rate_dv01(), floater.total_key_rate_dv01()
        assert set(totale) == set(krd["OIS"]) | set(krd["EURIBOR6M"])
        for nodo, valore in totale.items():
            assert valore == pytest.approx(sum(serie.get(nodo, 0.0) for serie in krd.values()), rel=1e-12)

    def test_il_totale_e_ordinato_per_data(self, senza_opzioni):
        nodi = list(senza_opzioni.total_key_rate_dv01())
        assert nodi == sorted(nodi)

    def test_con_una_curva_sola_il_totale_e_la_sua_serie(self, fixed_bond):
        assert fixed_bond.total_key_rate_dv01() == pytest.approx(fixed_bond.key_rate_dv01()["OIS"])

    def test_con_i_tenor_il_totale_tiene_l_ordine_dato(self, floater):
        assert list(floater.total_key_rate_dv01(nodes=["10Y", "2Y", "5Y"])) == ["10Y", "2Y", "5Y"]

    def test_il_totale_somma_a_total_dv01_senza_opzioni(self, senza_opzioni):
        totale = sum(senza_opzioni.total_key_rate_dv01().values())
        assert totale == pytest.approx(senza_opzioni.total_dv01(), rel=1e-5)

    def test_in_quotazioni_con_lo_jacobiano_somma_a_total_dv01(self, floater):
        totale = sum(floater.total_key_rate_dv01(bump="quotes", sticky="quotes", via_jacobian=True).values())
        assert totale == pytest.approx(floater.total_dv01(bump="quotes", sticky="quotes"), rel=1e-5)


# ---------------------------------------------------------------------------
# Rischio di credito: lo spread del titolo
# ---------------------------------------------------------------------------

class TestCreditRisk:

    def test_un_kind_ignoto_solleva(self, fixed_bond):
        with pytest.raises(ValueError):
            fixed_bond.cs01(kind="boh")

    @pytest.mark.parametrize("titolo", ["fixed_bond", "zero_bond", "floater", "callable_bond"])
    def test_il_cs01_e_negativo_per_un_titolo_lungo(self, titolo, request):
        assert request.getfixturevalue(titolo).cs01() < 0

    def test_il_cs01_dello_zero_e_meno_t_per_il_prezzo(self, zero_bond, ois):
        t = accrual_factor(ois.dcc, TRADE_DATE, pd.Timestamp("2034-01-15")).item()
        assert zero_bond.cs01() == pytest.approx(-t * dirty(zero_bond) * UN_BP, rel=1e-6)

    def test_oneside_e_simmetrico_si_somigliano(self, fixed_bond):
        assert fixed_bond.cs01(kind="oneside") == pytest.approx(fixed_bond.cs01(), rel=1e-3)

    @pytest.mark.parametrize("titolo", ["fixed_bond", "floater"])
    def test_il_cs01_e_la_dv01_parallela_sugli_zero_della_curva_di_sconto(self, titolo, request):
        bond = request.getfixturevalue(titolo)
        assert bond.cs01() == pytest.approx(bond.sensitivity(bump="zeros", sticky="zeros"), rel=1e-9)

    def test_lo_spread_torna_al_suo_valore(self, fixed_bond):
        fixed_bond.z_spread = 0.0123
        fixed_bond.cs01()
        fixed_bond.spread_duration()
        assert fixed_bond.z_spread == 0.0123

    def test_lo_spread_torna_anche_se_il_prezzo_solleva(self, fixed_bond, monkeypatch):
        fixed_bond.z_spread = 0.0123
        monkeypatch.setattr(fixed_bond, "_dirty_price", lambda: 1 / 0)
        with pytest.raises(ZeroDivisionError):
            fixed_bond.cs01()
        assert fixed_bond.z_spread == 0.0123

    def test_lo_shock_e_sullo_spread_del_titolo(self, fixed_bond, monkeypatch):
        visti = []
        monkeypatch.setattr(fixed_bond, "_dirty_price", lambda: visti.append(fixed_bond.z_spread) or 0.0)
        fixed_bond.z_spread = 0.01
        fixed_bond.cs01(shift_size=0.0005)
        assert visti == pytest.approx([0.0105, 0.0095])

    def test_la_spread_duration_e_il_cs01_normalizzato(self, fixed_bond):
        atteso = -fixed_bond.cs01() * 10000 / dirty(fixed_bond)
        assert fixed_bond.spread_duration() == pytest.approx(atteso)

    def test_la_spread_duration_del_fisso_e_la_modified_duration_ois(self, fixed_bond):
        assert fixed_bond.spread_duration() == pytest.approx(fixed_bond.modified_duration()["OIS"], rel=1e-9)

    def test_il_callable_shocka_lo_spread_del_sottostante(self, callable_bond):
        assert callable_bond.bonds == [callable_bond.bond]
        callable_bond.bond.z_spread = 0.02
        assert callable_bond.z_spread == 0.02
        callable_bond.z_spread = 0.03
        assert callable_bond.bond.z_spread == 0.03

    def test_il_callable_vale_meno_del_sottostante_anche_in_cs01(self, callable_bond):
        assert abs(callable_bond.cs01()) < abs(callable_bond.bond.cs01())

    def test_il_cs01_del_portafoglio_e_la_somma_dei_titoli(self, portfolio):
        somma = sum(bond.cs01() for bond in portfolio.bonds)
        assert portfolio.cs01() == pytest.approx(somma, rel=1e-6)

    def test_il_portafoglio_ripristina_tutti_gli_spread(self, portfolio):
        prima = [bond.z_spread for bond in portfolio.bonds]
        portfolio.cs01()
        assert [bond.z_spread for bond in portfolio.bonds] == prima


# ---------------------------------------------------------------------------
# Tasso fisso
# ---------------------------------------------------------------------------

class TestFixedRateBond:

    def test_solo_le_cedole_future(self, fixed_bond):
        assert (fixed_bond.cash_flows.paymentDate > TRADE_DATE).all()
        assert (fixed_bond.coupons_history.paymentDate <= TRADE_DATE).all()

    def test_il_rimborso_sta_sull_ultima_riga(self, fixed_bond):
        redemption = fixed_bond.cash_flows.redemption.to_numpy()
        assert redemption[-1] == pytest.approx(fixed_bond.face_amount)
        assert redemption[:-1] == pytest.approx(np.zeros(len(redemption) - 1))

    def test_il_flusso_e_cedola_piu_rimborso(self, fixed_bond):
        flussi = fixed_bond.cash_flows
        assert flussi.cashFlow.to_numpy() == pytest.approx(
            flussi.coupon.to_numpy() + flussi.redemption.to_numpy())

    def test_la_cedola_e_tasso_per_accrual_per_nozionale(self, fixed_bond):
        flussi = fixed_bond.cash_flows
        atteso = fixed_bond.coupon_rate * flussi.accrualFactor.to_numpy() * fixed_bond.face_amount
        assert flussi.coupon.to_numpy() == pytest.approx(atteso)

    def test_i_costruttori_non_scrivono_nella_cache(self, fixed_bond):
        fixed_bond._cash_flows = None
        fixed_bond._get_cash_flows()
        assert fixed_bond._cash_flows is None

    def test_i_key_rate_sommano_alla_parallela(self, fixed_bond):
        krd = fixed_bond.key_rate_dv01()["OIS"]
        assert sum(krd.values()) == pytest.approx(fixed_bond.sensitivity(bump="zeros", sticky="zeros"), rel=1e-5)

    def test_un_nodo_oltre_la_scadenza_non_conta(self, fixed_bond, ois):
        krd = fixed_bond.key_rate_dv01(bump="zeros")["OIS"]
        oltre = [valore for nodo, valore in krd.items() if nodo > pd.Timestamp("2040-01-01")]
        assert oltre == pytest.approx(np.zeros(len(oltre)))

    def test_la_dv01_e_negativa(self, fixed_bond):
        assert fixed_bond.sensitivity() < 0

    def test_le_tre_forme_di_shift_danno_numeri_diversi(self, fixed_bond):
        numeri = {round(fixed_bond.sensitivity(shift_type=forma), 8)
                  for forma in ("parallel", "slope", "curvature")}
        assert len(numeri) == 3


# ---------------------------------------------------------------------------
# Zero coupon: qui la sensibilita' ha una forma chiusa
# ---------------------------------------------------------------------------

class TestZeroCouponBond:

    def test_un_flusso_solo(self, zero_bond):
        assert len(zero_bond.cash_flows) == 1
        assert zero_bond.cash_flows.coupon.iloc[0] == 0.0

    def test_il_prezzo_e_il_fattore_di_sconto(self, zero_bond, ois):
        atteso = zero_bond.face_amount * ois.discount_factor_at(pd.Timestamp("2034-01-15"))
        assert dirty(zero_bond) == pytest.approx(atteso)

    def test_la_duration_e_la_vita_residua(self, zero_bond, ois):
        atteso = accrual_factor(ois.dcc, TRADE_DATE, pd.DatetimeIndex(["2034-01-15"])).item()
        assert zero_bond.duration() == pytest.approx(atteso, abs=1e-14)

    def test_lo_spread_sconta_esattamente_exp_meno_z_t(self, zero_bond, ois):
        prezzo = dirty(zero_bond)
        t = accrual_factor(ois.dcc, TRADE_DATE, zero_bond.maturity_date)
        zero_bond.z_spread = 0.01
        assert dirty(zero_bond) == pytest.approx(prezzo * np.exp(-0.01 * t), rel=1e-12)

    def test_la_dv01_sui_tassi_zero_vale_meno_t_per_prezzo(self, zero_bond):
        atteso = -zero_bond.duration() * dirty(zero_bond) * UN_BP
        assert zero_bond.sensitivity(bump="zeros") == pytest.approx(atteso, rel=1e-6)

    def test_la_dv01_sulle_quotazioni_e_diversa(self, zero_bond):
        assert zero_bond.sensitivity(bump="quotes") != pytest.approx(
            zero_bond.sensitivity(bump="zeros"), rel=1e-4)

    def test_i_key_rate_sommano_alla_parallela(self, zero_bond):
        krd = zero_bond.key_rate_dv01(bump="zeros")["OIS"]
        assert sum(krd.values()) == pytest.approx(zero_bond.sensitivity(bump="zeros"), rel=1e-5)


# ---------------------------------------------------------------------------
# Tasso variabile: l'unico con due curve, e quindi l'unico dove sticky conta
# ---------------------------------------------------------------------------

class TestFloatingRateBond:

    def test_la_curva_di_proiezione_e_quella_dell_indice(self, floater, euribor):
        assert floater.projection_curves == [euribor]

    def test_le_cedole_gia_fissate_vengono_dall_archivio(self, floater):
        flussi = floater.cash_flows
        fissate = flussi.resetDate <= TRADE_DATE
        assert fissate.any()
        assert flussi.resetRate[fissate].to_numpy() == pytest.approx(0.03)

    def test_la_cedola_e_il_fixing_piu_lo_spread(self, floater):
        flussi = floater.cash_flows
        assert flussi.couponRate.to_numpy(dtype=float) == pytest.approx(
            flussi.resetRate.to_numpy(dtype=float) + floater.spread)

    def test_le_cedole_future_vengono_dalla_curva(self, floater):
        flussi = floater.cash_flows
        future = flussi.resetDate > TRADE_DATE
        assert (flussi.resetRate[future].to_numpy() != 0.03).all()

    def test_la_cache_si_rifa_quando_la_proiezione_si_muove(self, floater, euribor):
        base = floater.cash_flows.couponRate.to_numpy(dtype=float).copy()
        with euribor.shocked_quotes(UN_BP):
            dentro = floater.cash_flows.couponRate.to_numpy(dtype=float)
            assert (dentro != base).any()
        assert floater.cash_flows.couponRate.to_numpy(dtype=float) == pytest.approx(base)

    def test_in_quotazioni_e_sugli_zero_rate_sono_due_numeri_diversi(self, floater):
        quotes = floater.sensitivity(curve="discount", bump="quotes", sticky="quotes")
        floater.cash_flows
        zeros = floater.sensitivity(curve="discount", bump="zeros", sticky="zeros")
        assert quotes != pytest.approx(zeros, rel=1e-6)

    def test_sotto_sticky_zero_le_cedole_non_si_muovono(self, floater, ois, euribor):
        base = floater.cash_flows.couponRate.to_numpy(dtype=float).copy()
        with euribor.frozen(), ois.shocked_quotes(UN_BP):
            assert floater.cash_flows.couponRate.to_numpy(dtype=float) == pytest.approx(base)

    def test_sotto_sticky_quotes_le_cedole_si_muovono(self, floater, ois):
        base = floater.cash_flows.couponRate.to_numpy(dtype=float).copy()
        with ois.shocked_quotes(UN_BP):
            assert (floater.cash_flows.couponRate.to_numpy(dtype=float) != base).any()

    def test_la_dv01_di_proiezione_ha_segno_opposto_a_quella_di_sconto(self, floater):
        assert floater.sensitivity(curve="discount") < 0
        assert floater.sensitivity(curve="projection") > 0

    def test_sotto_sticky_spread_il_variabile_e_quasi_immune(self, floater):
        sconto = abs(floater.sensitivity(curve="discount", bump="zeros", sticky="zeros"))
        insieme = abs(floater.sensitivity(curve="discount", bump="zeros", sticky="spread"))
        assert insieme < sconto / 10

    def test_i_due_delta_ortogonali_quasi_si_compensano(self, floater):
        sconto = floater.sensitivity(curve="discount", bump="zeros", sticky="zeros")
        proiezione = floater.sensitivity(curve="projection", bump="zeros", sticky="zeros")
        assert abs(sconto + proiezione) < abs(sconto) / 10

    def test_l_euribor_fissa_due_giorni_lavorativi_prima_dell_inizio(self, floater):
        assert floater.index.fixing_date(pd.Timestamp("2026-07-15")) == pd.Timestamp("2026-07-13")

    def test_le_date_di_fixing_vengono_dall_indice(self, floater):
        flussi = floater.cash_flows
        attese = pd.DatetimeIndex(floater.index.fixing_date(flussi.accrualStart.to_numpy()))
        assert (pd.DatetimeIndex(flussi.resetDate) == attese).all()

    def test_lo_storico_usa_le_date_di_fixing_dell_indice(self, floater):
        storico = floater.get_coupons_history()
        assert len(storico) > 0
        attese = pd.DatetimeIndex(floater.index.fixing_date(storico.accrualStart.to_numpy()))
        assert (pd.DatetimeIndex(storico.resetDate) == attese).all()
        assert storico.resetRate.to_numpy(dtype=float) == pytest.approx(0.03)

    @pytest.mark.parametrize("modello", [BlackCouponPricer, BachelierCouponPricer, DisplacedBlackCouponPricer])
    def test_il_floorlet_usa_la_frazione_d_anno_della_cedola_fissata_o_no(self, floater, modello):
        superficie = pd.DataFrame(1e-4, index=[0.5, 30.0], columns=[0.0, 0.2])
        floater.floor = 0.10
        floater.set_coupon_pricer(modello(superficie))
        flussi = floater.cash_flows
        assert (flussi.resetDate <= TRADE_DATE).any() and (flussi.resetDate > TRADE_DATE).any()
        tasso = flussi.couponRate.to_numpy(dtype=float)
        af = flussi.accrualFactor.to_numpy(dtype=float)
        atteso = (0.10 - tasso) * af * floater.face_amount
        assert flussi.floorlet.to_numpy(dtype=float) == pytest.approx(atteso, rel=1e-9)


# ---------------------------------------------------------------------------
# Callable
# ---------------------------------------------------------------------------

class TestCallableBond:

    def test_delega_lo_stato_al_titolo_sottostante(self, callable_bond, fixed_bond, ois):
        assert callable_bond.face_amount == fixed_bond.face_amount
        assert callable_bond.evaluation_date == fixed_bond.evaluation_date
        assert callable_bond.discount_curve is ois
        assert callable_bond.projection_curves == []

    def test_il_callable_vale_meno_del_titolo_nudo(self, callable_bond, fixed_bond):
        prezzi = callable_bond.prices()
        assert prezzi["optionValue"] > 0
        assert prezzi["dirtyPrice"] < prezzo_al_regolamento(fixed_bond)

    def test_il_prezzo_da_derivare_e_quello_callable(self, callable_bond, fixed_bond):
        regolamento = [fixed_bond.settlement_date]
        df = fixed_bond.pricer.discount_factor_at(fixed_bond, regolamento)[0]
        assert callable_bond._dirty_price() < dirty(fixed_bond)
        assert callable_bond.prices()["dirtyPrice"] == pytest.approx(callable_bond._dirty_price() / df, rel=1e-12)

    def test_l_opzione_riduce_la_sensibilita(self, callable_bond, fixed_bond):
        assert abs(callable_bond.sensitivity()) < abs(fixed_bond.sensitivity())

    def test_i_key_rate_sommano_alla_parallela(self, fixed_bond, callable_bond):
        worst = CallableBond(fixed_bond, callable_bond._call_schedule, method="worst")
        krd = worst.key_rate_dv01()["OIS"]
        assert sum(krd.values()) == pytest.approx(worst.sensitivity(bump="zeros", sticky="zeros"), rel=1e-5)

    def test_il_metodo_di_valutazione_arriva_fino_a_prices(self, fixed_bond, callable_bond):
        worst = CallableBond(fixed_bond, callable_bond._call_schedule, method="worst")
        assert worst.sensitivity() != pytest.approx(callable_bond.sensitivity(), rel=1e-3)

    def test_un_metodo_ignoto_solleva(self, fixed_bond, callable_bond):
        with pytest.raises(ValueError):
            CallableBond(fixed_bond, callable_bond._call_schedule, method="boh")

    def test_una_volatilita_piu_alta_vale_di_piu(self, fixed_bond, callable_bond):
        alto = CallableBond(fixed_bond, callable_bond._call_schedule, volatility=0.02, method="tree")
        basso = CallableBond(fixed_bond, callable_bond._call_schedule, volatility=0.005, method="tree")
        assert alto.prices()["optionValue"] > basso.prices()["optionValue"]


# ---------------------------------------------------------------------------
# Il portafoglio
# ---------------------------------------------------------------------------

EURIBOR3M_SWAPS = {"2Y": 2.640, "3Y": 2.635, "5Y": 2.668, "10Y": 2.851, "30Y": 3.050}


@pytest.fixture
def euribor3m(ois):
    instruments = [Deposit("3M", 0.02324, TRADE_DATE)]
    instruments += [Swap(tenor, quote / 100, TRADE_DATE, frequency=1, float_frequency=4,
                         dcc="30/360", discount_curve=ois) for tenor, quote in EURIBOR3M_SWAPS.items()]
    return YieldCurve(instruments, "EURIBOR3M", TRADE_DATE)


@pytest.fixture
def floater3m(ois, euribor3m):
    schedule = Schedule("2025-03-20", "2029-03-20", 4)
    resets = pd.DatetimeIndex(business_days_before(schedule.schedule["startingDate"], 2))
    index = Euribor3M(projection_curve=euribor3m, fixings=pd.Series(0.026, index=resets))
    bond = FloatingRateBond(schedule=schedule, index=index, dcc="ACT/360", face_amount=4_000_000,
                            spread=0.003)
    bond.set_evaluation_date(TRADE_DATE)
    bond.set_pricer(Pricer(ois))
    return bond


@pytest.fixture
def portfolio(fixed_bond, zero_bond, floater, floater3m, callable_bond):
    return BondPortfolio([fixed_bond, zero_bond, floater, floater3m, callable_bond])


class TestBondPortfolio:

    def test_e_un_ospite_di_rate_risk(self):
        assert issubclass(BondPortfolio, RateRisk)

    def test_la_curva_di_sconto_e_quella_comune(self, portfolio, ois):
        assert portfolio.discount_curve is ois

    def test_le_proiezioni_sono_le_distinte_dei_titoli(self, portfolio, euribor, euribor3m):
        assert portfolio.projection_curves == [euribor, euribor3m]

    def test_due_curve_di_sconto_sollevano(self, fixed_bond, zero_bond):
        altra = YieldCurve([Deposit("1Y", 0.03, TRADE_DATE)], "altra", TRADE_DATE)
        zero_bond.set_pricer(Pricer(altra))
        with pytest.raises(ValueError):
            BondPortfolio([fixed_bond, zero_bond])

    def test_il_prezzo_e_la_somma_dei_titoli(self, portfolio):
        atteso = sum(bond._dirty_price() for bond in portfolio.bonds)
        assert portfolio._dirty_price() == pytest.approx(atteso)
        assert portfolio.prices()["dirtyPrice"] == pytest.approx(sum(prezzo_al_regolamento(b) for b in portfolio.bonds))

    def test_prices_somma_ogni_chiave(self, portfolio, callable_bond):
        prezzi = portfolio.prices()
        assert prezzi["cleanPrice"] == pytest.approx(prezzi["dirtyPrice"] - prezzi["accruedInterest"])
        assert prezzi["optionValue"] == pytest.approx(callable_bond.prices()["optionValue"])

    def test_i_flussi_si_sommano_per_data(self, portfolio):
        flussi = portfolio.cash_flows
        assert flussi.index.is_unique
        assert flussi.cashFlow.sum() == pytest.approx(
            sum(bond.cash_flows.cashFlow.sum() for bond in portfolio.bonds))
        data = portfolio.bonds[0].cash_flows.paymentDate.iloc[-1]
        atteso = sum(bond.cash_flows.set_index("paymentDate").cashFlow.get(data, 0.0) for bond in portfolio.bonds)
        assert flussi.cashFlow[data] == pytest.approx(atteso)

    def test_la_sensitivity_e_la_somma_delle_sensitivity(self, portfolio):
        atteso = sum(bond.sensitivity() for bond in portfolio.bonds)
        assert portfolio.sensitivity() == pytest.approx(atteso, rel=1e-9)

    def test_la_proiezione_va_nominata(self, portfolio, euribor):
        with pytest.raises(ValueError):
            portfolio.sensitivity(curve="projection")
        assert portfolio.sensitivity(curve=euribor) != 0.0

    def test_la_6m_non_muove_chi_sta_sulla_3m(self, portfolio, euribor, euribor3m, floater, floater3m):
        assert portfolio.sensitivity(curve=euribor) == pytest.approx(floater.sensitivity(curve=euribor), rel=1e-9)
        assert portfolio.sensitivity(curve=euribor3m) == pytest.approx(floater3m.sensitivity(curve=euribor3m), rel=1e-9)

    def test_sticky_zero_congela_tutte_le_proiezioni(self, portfolio, euribor, euribor3m, monkeypatch):
        congelate = []
        originale = YieldCurve.frozen

        def spia(self):
            congelate.append(self)
            return originale(self)

        monkeypatch.setattr(YieldCurve, "frozen", spia)
        portfolio.sensitivity(bump="zeros", sticky="zeros", kind="oneside")
        assert congelate[:2] == [euribor, euribor3m]

    def test_uno_shock_per_nodo_non_uno_per_titolo(self, portfolio, ois, monkeypatch):
        conteggio = []
        originale = YieldCurve._bootstrap

        def spia(self):
            conteggio.append(self)
            return originale(self)

        monkeypatch.setattr(YieldCurve, "_bootstrap", spia)
        portfolio.key_rate_dv01(nodes=[1, 3], bump="quotes", sticky="quotes", kind="oneside")
        assert conteggio.count(ois) == 2

    def test_lo_shock_non_lascia_traccia(self, portfolio, ois, euribor, euribor3m):
        prima = ois.pillars, euribor.pillars, euribor3m.pillars
        portfolio.key_rate_dv01(nodes=["2Y"], sticky="spread", bump="zeros", kind="oneside")
        assert (ois.pillars, euribor.pillars, euribor3m.pillars) == prima


# ---------------------------------------------------------------------------
# Il KR01 in quotazioni attraverso lo Jacobiano
# ---------------------------------------------------------------------------

class TestViaJacobian:

    def test_coincide_con_le_differenze_finite(self, fixed_bond):
        finite = fixed_bond.key_rate_dv01(bump="quotes")["OIS"]
        jacobiano = fixed_bond.key_rate_dv01(bump="quotes", via_jacobian=True)["OIS"]
        assert list(jacobiano) == list(finite)
        for nodo in finite:
            assert jacobiano[nodo] == pytest.approx(finite[nodo], rel=1e-5, abs=1e-6)

    def test_somma_alla_parallela(self, fixed_bond):
        jacobiano = fixed_bond.key_rate_dv01(bump="quotes", via_jacobian=True)["OIS"]
        assert sum(jacobiano.values()) == pytest.approx(fixed_bond.sensitivity(bump="quotes"), rel=1e-5)

    def test_vale_anche_sul_variabile_con_due_curve(self, floater):
        finite = floater.key_rate_dv01(bump="quotes", sticky="quotes")["OIS"]
        jacobiano = floater.key_rate_dv01(bump="quotes", sticky="quotes", via_jacobian=True)["OIS"]
        for nodo in finite:
            assert jacobiano[nodo] == pytest.approx(finite[nodo], rel=1e-4, abs=1e-4)

    def test_sul_portafoglio_con_due_euribor(self, portfolio):
        finite = portfolio._key_rate_dv01(curve="discount", bump="quotes", sticky="quotes")
        jacobiano = portfolio._key_rate_dv01(curve="discount", bump="quotes", sticky="quotes", via_jacobian=True)
        for nodo in finite:
            assert jacobiano[nodo] == pytest.approx(finite[nodo], rel=1e-3, abs=1e-2)

    def test_sulla_ois_le_quotazioni_ferme_includono_il_ricalcolo_della_euribor(self, floater, ois):
        con = np.array(list(floater.key_rate_dv01(bump="quotes", sticky="quotes", via_jacobian=True)["OIS"].values()))
        zeri = np.array(list(floater.key_rate_dv01(bump="zeros", sticky="zeros")["OIS"].values()))
        senza = solve_triangular(ois.jacobian.T, zeri, lower=False)
        assert abs(con - senza).max() > 1e-2

    def test_con_le_quotazioni_ferme_non_ribootstrappa_nessuna_curva(self, floater, ois, euribor, monkeypatch):
        ois.jacobian, euribor.jacobian
        chiamate = []
        for curve in (ois, euribor):
            originale = curve._bootstrap
            monkeypatch.setattr(curve, "_bootstrap", lambda originale=originale: chiamate.append(1) or originale())
        floater.key_rate_dv01(bump="quotes", sticky="quotes", via_jacobian=True)
        assert chiamate == []

    def test_restituisce_float_di_python(self, fixed_bond):
        krd = fixed_bond.key_rate_dv01(bump="quotes", via_jacobian=True)
        assert all(type(v) is float for serie in krd.values() for v in serie.values())

    def test_vale_sulla_curva_di_proiezione(self, floater):
        finite = floater.key_rate_dv01(bump="quotes")["EURIBOR6M"]
        jacobiano = floater.key_rate_dv01(bump="quotes", via_jacobian=True)["EURIBOR6M"]
        for nodo in finite:
            assert jacobiano[nodo] == pytest.approx(finite[nodo], rel=1e-3, abs=1e-4)

    def test_e_diverso_dal_kr01_sui_tassi_zero(self, fixed_bond):
        zeri = fixed_bond.key_rate_dv01(bump="zeros")["OIS"]
        jacobiano = fixed_bond.key_rate_dv01(bump="quotes", via_jacobian=True)["OIS"]
        assert any(abs(jacobiano[n] - zeri[n]) > 1e-3 for n in zeri)

    def test_rifiuta_il_bump_sui_tassi_zero(self, fixed_bond):
        with pytest.raises(ValueError):
            fixed_bond.key_rate_dv01(bump="zeros", via_jacobian=True)

    def test_rifiuta_una_griglia_personalizzata(self, fixed_bond):
        with pytest.raises(ValueError):
            fixed_bond.key_rate_dv01(nodes=["5Y", "10Y"], bump="quotes", via_jacobian=True)

    def test_rifiuta_sticky_spread(self, floater):
        with pytest.raises(ValueError):
            floater.key_rate_dv01(bump="quotes", sticky="spread", via_jacobian=True)

    def test_non_lascia_traccia_sulla_curva(self, fixed_bond, ois):
        prima = list(ois._log_df), [instrument.quote for instrument in ois.instruments]
        fixed_bond.key_rate_dv01(bump="quotes", via_jacobian=True)
        assert (list(ois._log_df), [instrument.quote for instrument in ois.instruments]) == prima

    def test_non_ribootstrappa(self, fixed_bond, ois, monkeypatch):
        ois.jacobian
        chiamate = []
        originale = ois._bootstrap
        monkeypatch.setattr(ois, "_bootstrap", lambda: chiamate.append(1) or originale())
        fixed_bond.key_rate_dv01(bump="quotes", via_jacobian=True)
        assert chiamate == []

    def test_lo_jacobiano_resta_in_cache_fra_due_chiamate(self, fixed_bond, ois):
        fixed_bond.key_rate_dv01(bump="quotes", via_jacobian=True)
        prima = ois._jacobian
        fixed_bond.key_rate_dv01(bump="quotes", via_jacobian=True)
        assert ois._jacobian is prima
