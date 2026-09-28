"""
Test per il bootstrap multicurva: YieldCurve e i suoi strumenti.
"""
import numpy as np
import pandas as pd
import pytest

from QuantGYMM.fixed_income.curves import (
    Deposit, ForwardRateAgreement, Future, Swap, YieldCurve)
from QuantGYMM.utils import business_days_after, imm_date, is_bd, tenor_offset

# ---------------------------------------------------------------------------
# Quotazioni vere al 30/06/2026. OIS ESTER per la curva di sconto, Euribor 6M
# per quella di proiezione.
# ---------------------------------------------------------------------------

TRADE_DATE = pd.Timestamp("2026-06-30")
SPOT = pd.Timestamp("2026-07-02")

OIS_DEPOSITS = {"1W": 2.185, "2W": 2.186, "1M": 2.189, "2M": 2.196,
                "3M": 2.227, "6M": 2.309, "9M": 2.359, "1Y": 2.394}

OIS_SWAPS = {"2Y": 2.412, "3Y": 2.413, "4Y": 2.434, "5Y": 2.465, "10Y": 2.688,
             "20Y": 2.975, "30Y": 2.957, "50Y": 2.734}

EURIBOR_SWAPS = {"2Y": 2.711, "3Y": 2.704, "4Y": 2.714, "5Y": 2.734,
                 "10Y": 2.914, "20Y": 3.155, "30Y": 3.106}

# FRA 6M sull'Euribor, dal foglio Bloomberg.
EURIBOR_FRAS = {("1M", "7M"): 2.599, ("2M", "8M"): 2.639, ("3M", "9M"): 2.667,
                ("4M", "10M"): 2.685, ("5M", "11M"): 2.700, ("6M", "12M"): 2.719,
                ("9M", "15M"): 2.723, ("12M", "18M"): 2.704}

# Futures Euribor 3M, dal foglio Bloomberg.
EURIBOR_FUTURES = {"2026-10": 2.534, "2027-03": 2.613, "2027-06": 2.607,
                   "2027-09": 2.575, "2027-12": 2.539, "2028-03": 2.522,
                   "2028-06": 2.519}


def deposits():
    return [Deposit(tenor, quote / 100, TRADE_DATE) for tenor, quote in OIS_DEPOSITS.items()]


def ois_swaps():
    return [Swap(tenor, quote / 100, TRADE_DATE, dcc="ACT/360") for tenor, quote in OIS_SWAPS.items()]


@pytest.fixture
def ois():
    return YieldCurve(deposits() + ois_swaps(), "OIS", TRADE_DATE)


@pytest.fixture
def euribor(ois):
    instruments = [Deposit("6M", 0.02568, TRADE_DATE)]
    instruments += [ForwardRateAgreement(start, end, quote / 100, TRADE_DATE)
                    for (start, end), quote in EURIBOR_FRAS.items()]
    instruments += [Swap(tenor, quote / 100, TRADE_DATE, frequency=1, float_frequency=2,
                         dcc="30/360", discount_curve=ois)
                    for tenor, quote in EURIBOR_SWAPS.items()]
    return YieldCurve(instruments, "EURIBOR6M", TRADE_DATE)


# ---------------------------------------------------------------------------
# Date degli strumenti
# ---------------------------------------------------------------------------

class TestDeposit:

    def test_parte_a_spot(self):
        assert Deposit("6M", 0.02, TRADE_DATE).starting_date == SPOT

    def test_scadenza_da_spot_non_da_trade_date(self):
        deposit = Deposit("6M", 0.02, TRADE_DATE)
        assert deposit.maturity == pd.Timestamp("2027-01-04")
        assert deposit.maturity != TRADE_DATE + tenor_offset("6M")

    def test_scadenza_e_giorno_lavorativo(self):
        for tenor in OIS_DEPOSITS:
            assert is_bd(Deposit(tenor, 0.02, TRADE_DATE).maturity)

    def test_accrual_su_act360(self):
        deposit = Deposit("1W", 0.02, TRADE_DATE)
        assert deposit.accrual == pytest.approx(7 / 360)

    def test_accrual_scalare(self):
        assert isinstance(Deposit("1W", 0.02, TRADE_DATE).accrual, float)


class TestForwardRateAgreement:

    def test_parte_al_tenor_iniziale_non_a_spot(self):
        fra = ForwardRateAgreement("3M", "9M", 0.02, TRADE_DATE)
        assert fra.starting_date == pd.Timestamp("2026-10-02")
        assert fra.starting_date != fra.spot_date

    def test_scade_al_tenor_finale(self):
        fra = ForwardRateAgreement("3M", "9M", 0.02, TRADE_DATE)
        assert fra.maturity == pd.Timestamp("2027-04-02")

    def test_accrual_copre_il_solo_periodo_forward(self):
        fra = ForwardRateAgreement("3M", "9M", 0.02, TRADE_DATE)
        atteso = (fra.maturity - fra.starting_date).days / 360
        assert fra.accrual == pytest.approx(atteso)


class TestFuture:

    def test_date_imm(self):
        future = Future("2026-10", 0.02534, TRADE_DATE, tenor="3M")
        assert future.starting_date == pd.Timestamp("2026-10-21")
        assert future.maturity == pd.Timestamp("2027-01-20")

    def test_il_giorno_del_mese_di_consegna_e_irrilevante(self):
        primo = Future("2026-10-01", 0.02, TRADE_DATE, tenor="3M")
        ultimo = Future("2026-10-28", 0.02, TRADE_DATE, tenor="3M")
        assert primo.starting_date == ultimo.starting_date
        assert primo.maturity == ultimo.maturity

    def test_la_striscia_si_incastra(self):
        catena = [Future(mese, quote / 100, TRADE_DATE, tenor="3M")
                  for mese, quote in EURIBOR_FUTURES.items()][1:]
        for precedente, successivo in zip(catena, catena[1:]):
            assert precedente.maturity == successivo.starting_date

    def test_le_date_non_dipendono_dalla_trade_date(self):
        oggi = Future("2027-03", 0.02, TRADE_DATE, tenor="3M")
        domani = Future("2027-03", 0.02, pd.Timestamp("2026-09-15"), tenor="3M")
        assert oggi.starting_date == domani.starting_date

    def test_tenor_diverso_da_tre_mesi(self):
        future = Future("2026-10", 0.02, TRADE_DATE, tenor="6M")
        assert future.maturity == imm_date("2027-04")


class TestSwap:

    def test_parte_a_spot(self):
        assert Swap("5Y", 0.02, TRADE_DATE).spot_date == SPOT

    def test_ultimo_pagamento_alla_scadenza(self):
        for tenor in OIS_SWAPS:
            swap = Swap(tenor, 0.02, TRADE_DATE)
            assert swap.payment_date[-1] == swap.maturity

    def test_un_pagamento_per_anno(self):
        assert len(Swap("10Y", 0.02, TRADE_DATE, frequency=1).payment_date) == 10

    def test_la_gamba_variabile_paga_il_doppio(self):
        swap = Swap("10Y", 0.02, TRADE_DATE, frequency=1, float_frequency=2)
        assert len(swap.float_payment_date) == 2 * len(swap.payment_date)

    def test_le_due_gambe_condividono_gli_estremi(self):
        swap = Swap("10Y", 0.02, TRADE_DATE, frequency=1, float_frequency=2)
        assert swap.float_starting_date[0] == swap.spot_date
        assert swap.float_payment_date[-1] == swap.maturity

    def test_le_due_gambe_coincidono_quando_la_frequenza_e_la_stessa(self):
        swap = Swap("5Y", 0.02, TRADE_DATE, frequency=1)
        assert np.array_equal(swap.float_payment_date, swap.payment_date)

    def test_accrual_uno_per_periodo(self):
        swap = Swap("10Y", 0.02, TRADE_DATE)
        assert len(swap.accrual) == len(swap.payment_date)


# ---------------------------------------------------------------------------
# La curva come contenitore
# ---------------------------------------------------------------------------

class TestYieldCurve:

    def test_un_pilastro_per_strumento(self, ois):
        assert len(ois._times) == len(ois.instruments) + 1

    def test_ancorata_a_uno_sulla_trade_date(self, ois):
        assert ois.discount_factor_at(TRADE_DATE) == pytest.approx(1.0)

    def test_i_pilastri_stanno_alle_scadenze(self, ois):
        atteso = [0.0] + [(instrument.maturity - TRADE_DATE).days / 365
                          for instrument in ois.instruments]
        assert ois._times == pytest.approx(atteso)

    def test_strumenti_ordinati_per_scadenza(self, ois):
        scadenze = [instrument.maturity for instrument in ois.instruments]
        assert scadenze == sorted(scadenze)

    def test_ordine_di_ingresso_irrilevante(self):
        dritto = YieldCurve(deposits() + ois_swaps(), "curva", TRADE_DATE)
        rovescio = YieldCurve(list(reversed(deposits() + ois_swaps())), "curva", TRADE_DATE)
        assert dritto._log_df == pytest.approx(rovescio._log_df)

    def test_fattori_decrescenti(self, ois):
        df = [ois.discount_factor_at(instrument.maturity) for instrument in ois.instruments]
        assert all(dopo < prima for prima, dopo in zip(df, df[1:]))

    def test_accetta_un_vettore_di_date(self, ois):
        date = np.array([instrument.maturity for instrument in ois.instruments])
        df = ois.discount_factor_at(date)
        assert df.shape == date.shape
        assert df[3] == pytest.approx(ois.discount_factor_at(date[3]))

    def test_una_data_sola_torna_uno_scalare(self, ois):
        assert isinstance(ois.discount_factor_at(SPOT), float)

    def test_interpolatore_ricostruito_quando_un_pilastro_si_muove(self, ois):
        prima = ois.discount_factor_at(pd.Timestamp("2031-07-02"))
        ois._log_df[-1] -= 0.05
        ois._interpolator = None
        assert ois.discount_factor_at(pd.Timestamp("2056-07-03")) != pytest.approx(prima)

    def test_estrapola_oltre_l_ultimo_pilastro(self, ois):
        ultimo = ois.instruments[-1].maturity
        oltre = ultimo + tenor_offset("10Y")
        assert ois.discount_factor_at(oltre) < ois.discount_factor_at(ultimo)

    def test_lo_zero_rate_e_coerente_col_fattore(self, ois):
        data = pd.Timestamp("2036-07-02")
        anni = (data - TRADE_DATE).days / 365
        atteso = np.exp(-ois.zero_rates_at(data) * anni)
        assert ois.discount_factor_at(data) == pytest.approx(atteso)


# ---------------------------------------------------------------------------
# Il bootstrap riprezza quello che ha in pancia
# ---------------------------------------------------------------------------

class TestBootstrap:

    def test_solo_depositi(self):
        curve = YieldCurve(deposits(), "curva", TRADE_DATE)
        for instrument in curve.instruments:
            assert instrument.implied_quote(curve) == pytest.approx(instrument.quote, abs=1e-12)

    def test_depositi_e_swap(self, ois):
        for instrument in ois.instruments:
            assert instrument.implied_quote(ois) == pytest.approx(instrument.quote, abs=1e-12)

    def test_fra(self):
        instruments = [Deposit("6M", 0.02568, TRADE_DATE)]
        instruments += [ForwardRateAgreement(start, end, quote / 100, TRADE_DATE)
                        for (start, end), quote in EURIBOR_FRAS.items()]
        curve = YieldCurve(instruments, "curva", TRADE_DATE)
        for instrument in curve.instruments:
            assert instrument.implied_quote(curve) == pytest.approx(instrument.quote, abs=1e-12)

    def test_futures(self):
        instruments = [Deposit("3M", 0.02324, TRADE_DATE)]
        instruments += [Future(mese, quote / 100, TRADE_DATE, tenor="3M")
                        for mese, quote in EURIBOR_FUTURES.items()]
        curve = YieldCurve(instruments, "curva", TRADE_DATE)
        for instrument in curve.instruments:
            assert instrument.implied_quote(curve) == pytest.approx(instrument.quote, abs=1e-12)

    def test_euribor_scontata_ois(self, euribor):
        for instrument in euribor.instruments:
            assert instrument.implied_quote(euribor) == pytest.approx(instrument.quote, abs=1e-12)

    def test_uno_strumento_solo(self):
        curve = YieldCurve([Deposit("1Y", 0.02394, TRADE_DATE)], "curva", TRADE_DATE)
        assert len(curve._times) == 2
        assert curve.instruments[0].implied_quote(curve) == pytest.approx(0.02394)

    def test_il_deposito_sconta_da_spot_non_da_oggi(self):
        curve = YieldCurve([Deposit("1W", 0.02185, TRADE_DATE)], "curva", TRADE_DATE)
        deposit = curve.instruments[0]
        df = curve.discount_factor_at(deposit.maturity)
        sbagliato = (1 / df - 1) / deposit.accrual
        assert sbagliato != pytest.approx(deposit.quote, abs=1e-6)


# ---------------------------------------------------------------------------
# Convexity
# ---------------------------------------------------------------------------

class TestConvexity:

    def test_alza_il_forward_richiesto(self):
        base = [Deposit("3M", 0.02324, TRADE_DATE),
                Future("2026-10", 0.02534, TRADE_DATE, tenor="3M")]
        aggiustati = [Deposit("3M", 0.02324, TRADE_DATE),
                      Future("2026-10", 0.02534, TRADE_DATE, tenor="3M", convexity=0.0005)]
        scadenza = base[-1].maturity
        assert (YieldCurve(aggiustati, "curva", TRADE_DATE).zero_rates_at(scadenza)
                < YieldCurve(base, "curva", TRADE_DATE).zero_rates_at(scadenza))

    def test_a_zero_non_cambia_niente(self):
        con = YieldCurve([Deposit("3M", 0.02324, TRADE_DATE),
                          Future("2026-10", 0.02534, TRADE_DATE, tenor="3M", convexity=0.0)], "curva", TRADE_DATE)
        senza = YieldCurve([Deposit("3M", 0.02324, TRADE_DATE),
                            Future("2026-10", 0.02534, TRADE_DATE, tenor="3M")], "curva", TRADE_DATE)
        assert con._log_df == pytest.approx(senza._log_df)


# ---------------------------------------------------------------------------
# Multicurva
# ---------------------------------------------------------------------------

class TestMultiCurve:

    def test_la_curva_di_sconto_entra_nel_risultato(self, ois):
        propria = YieldCurve([Deposit("6M", 0.02568, TRADE_DATE),
                              Swap("10Y", 0.02914, TRADE_DATE, frequency=1, float_frequency=2,
                                   dcc="30/360")], "curva", TRADE_DATE)
        esterna = YieldCurve([Deposit("6M", 0.02568, TRADE_DATE),
                              Swap("10Y", 0.02914, TRADE_DATE, frequency=1, float_frequency=2,
                                   dcc="30/360", discount_curve=ois)], "curva", TRADE_DATE)
        data = pd.Timestamp("2036-07-02")
        assert propria.zero_rates_at(data) != pytest.approx(esterna.zero_rates_at(data), abs=1e-9)

    def test_la_proiezione_sta_sopra_lo_sconto(self, ois, euribor):
        for anni in (2, 5, 10, 20, 30):
            data = TRADE_DATE + tenor_offset(f"{anni}Y")
            assert euribor.zero_rates_at(data) > ois.zero_rates_at(data)

    def test_il_basis_e_stabile(self, ois, euribor):
        basis = [euribor.zero_rates_at(TRADE_DATE + tenor_offset(f"{anni}Y"))
                 - ois.zero_rates_at(TRADE_DATE + tenor_offset(f"{anni}Y"))
                 for anni in (2, 5, 10, 20, 30)]
        assert all(0.0 < singolo < 0.005 for singolo in basis)

    def test_la_curva_di_sconto_non_si_muove(self, ois, euribor):
        prima = ois.discount_factor_at(pd.Timestamp("2036-07-02"))
        YieldCurve([Deposit("6M", 0.02568, TRADE_DATE),
                    Swap("10Y", 0.02914, TRADE_DATE, frequency=1, float_frequency=2,
                         dcc="30/360", discount_curve=ois)], "curva", TRADE_DATE)
        assert ois.discount_factor_at(pd.Timestamp("2036-07-02")) == prima

    def test_senza_sconto_esterno_la_gamba_variabile_telescopa(self):
        curve = YieldCurve(deposits() + ois_swaps(), "curva", TRADE_DATE)
        swap = [instrument for instrument in curve.instruments
                if isinstance(instrument, Swap) and instrument.tenor == "10Y"][0]
        telescopico = ((curve.discount_factor_at(swap.spot_date)
                        - curve.discount_factor_at(swap.maturity))
                       / (curve.discount_factor_at(swap.payment_date) @ swap.accrual))
        assert swap.implied_quote(curve) == pytest.approx(telescopico)


# ---------------------------------------------------------------------------
# Coerenza fra i pezzi
# ---------------------------------------------------------------------------

class TestCoerenza:

    def test_deposito_e_fra_degenere_coincidono(self):
        curve = YieldCurve(deposits(), "curva", TRADE_DATE)
        deposit = Deposit("6M", 0.02, TRADE_DATE)
        fra = ForwardRateAgreement("0M", "6M", 0.02, TRADE_DATE)
        assert fra.starting_date == deposit.starting_date
        assert fra.implied_quote(curve) == pytest.approx(deposit.implied_quote(curve))

    def test_lo_swap_a_un_anno_e_il_deposito_a_un_anno(self):
        curve = YieldCurve(deposits(), "curva", TRADE_DATE)
        swap = Swap("1Y", 0.02, TRADE_DATE, frequency=1, dcc="ACT/360")
        deposit = Deposit("1Y", 0.02, TRADE_DATE)
        assert swap.implied_quote(curve) == pytest.approx(deposit.implied_quote(curve))

    def test_il_deposito_e_il_rapporto_fra_spot_e_scadenza(self, ois):
        deposit = Deposit("6M", 0.0, TRADE_DATE)
        atteso = (ois.discount_factor_at(deposit.spot_date)
                  / ois.discount_factor_at(deposit.maturity) - 1) / deposit.accrual
        assert deposit.implied_quote(ois) == pytest.approx(atteso)

    def test_il_fra_e_il_rapporto_fra_le_sue_due_date(self, ois):
        fra = ForwardRateAgreement("3M", "9M", 0.0, TRADE_DATE)
        atteso = (ois.discount_factor_at(fra.starting_date)
                  / ois.discount_factor_at(fra.maturity) - 1) / fra.accrual
        assert fra.implied_quote(ois) == pytest.approx(atteso)
        assert fra.implied_quote(ois) != pytest.approx(
            (ois.discount_factor_at(fra.spot_date)
             / ois.discount_factor_at(fra.maturity) - 1) / fra.accrual)

    def test_il_future_e_il_rapporto_fra_le_date_imm(self, ois):
        future = Future("2027-03", 0.0, TRADE_DATE, tenor="3M", convexity=0.0004)
        atteso = (ois.discount_factor_at(future.starting_date)
                  / ois.discount_factor_at(future.maturity) - 1) / future.accrual
        assert future.implied_quote(ois) == pytest.approx(atteso + 0.0004)

    def test_il_par_rate_dello_swap_calcolato_fuori(self, ois, euribor):
        swap = Swap("10Y", 0.0, TRADE_DATE, frequency=1, float_frequency=2,
                    dcc="30/360", discount_curve=ois)
        p_start = euribor.discount_factor_at(swap.float_starting_date)
        p_end = euribor.discount_factor_at(swap.float_payment_date)
        gamba_variabile = ((p_start / p_end - 1) * ois.discount_factor_at(swap.float_payment_date)).sum()
        annuity = (ois.discount_factor_at(swap.payment_date) * swap.accrual).sum()
        assert swap.implied_quote(euribor) == pytest.approx(gamba_variabile / annuity)

    def test_la_curva_riproduce_i_forward_dei_fra(self):
        instruments = [Deposit("6M", 0.02568, TRADE_DATE)]
        instruments += [ForwardRateAgreement(start, end, quote / 100, TRADE_DATE)
                        for (start, end), quote in EURIBOR_FRAS.items()]
        curve = YieldCurve(instruments, "curva", TRADE_DATE)
        controllo = ForwardRateAgreement("3M", "9M", 0.0, TRADE_DATE)
        assert controllo.implied_quote(curve) == pytest.approx(EURIBOR_FRAS[("3M", "9M")] / 100, abs=1e-12)


# ---------------------------------------------------------------------------
# Shock: quotazioni di mercato contro tassi zero, e propagazione
# ---------------------------------------------------------------------------

DIECI_ANNI = pd.Timestamp("2036-07-02")
UN_BP = 0.0001


def errore_di_riprezzamento(curve):
    """Quanto la curva sbaglia sulle proprie quotazioni. Zero se le riprezza."""
    return max(abs(instrument.implied_quote(curve) - instrument.quote)
               for instrument in curve.instruments)


def nodo_del_tenor(curve, tenor):
    return next(k for k, instrument in enumerate(curve.instruments)
                if getattr(instrument, "tenor", None) == tenor)


def stato(curve):
    return (list(curve._times), list(curve._log_df),
            [instrument.quote for instrument in curve.instruments])


class TestShockedQuotes:

    def test_la_quotazione_e_mossa_dentro_l_ambito(self, ois):
        nodo = nodo_del_tenor(ois, "10Y")
        prima = ois.instruments[nodo].quote
        with ois.shocked_quotes(UN_BP, node=nodo):
            assert ois.instruments[nodo].quote == pytest.approx(prima + UN_BP)
        assert ois.instruments[nodo].quote == prima

    def test_la_curva_riprezza_le_quotazioni_bumpate(self, ois):
        with ois.shocked_quotes(UN_BP):
            assert errore_di_riprezzamento(ois) == pytest.approx(0.0, abs=1e-12)

    def test_il_bump_parallelo_alza_i_tassi_zero(self, ois):
        prima = ois.zero_rates_at(DIECI_ANNI)
        with ois.shocked_quotes(UN_BP):
            assert ois.zero_rates_at(DIECI_ANNI) > prima

    def test_un_nodo_solo_lascia_fermi_i_pilastri_precedenti(self, ois):
        nodo = nodo_del_tenor(ois, "10Y")
        prima = list(ois._log_df)
        with ois.shocked_quotes(UN_BP, node=nodo):
            dopo = list(ois._log_df)
        assert dopo[:nodo + 1] == pytest.approx(prima[:nodo + 1])
        assert dopo[nodo + 1] != pytest.approx(prima[nodo + 1])

    def test_lo_stato_torna_come_prima(self, ois):
        prima = stato(ois)
        with ois.shocked_quotes(UN_BP):
            pass
        assert stato(ois) == prima

    def test_un_eccezione_non_lascia_la_curva_shockata(self, ois):
        prima = stato(ois)
        with pytest.raises(RuntimeError):
            with ois.shocked_quotes(UN_BP):
                raise RuntimeError
        assert stato(ois) == prima

    def test_i_tassi_tornano_come_prima_dopo_l_ambito(self, ois):
        prima = ois.zero_rates_at(DIECI_ANNI)
        with ois.shocked_quotes(UN_BP):
            ois.zero_rates_at(DIECI_ANNI)
        assert ois.zero_rates_at(DIECI_ANNI) == pytest.approx(prima)

    def test_i_pilastri_cambiano_dentro_e_tornano_identici_fuori(self, ois):
        prima = ois.pillars
        with ois.shocked_quotes(UN_BP):
            assert ois.pillars != prima
        assert ois.pillars == prima


class TestShockedZero:

    def test_il_bump_parallelo_alza_ogni_zero_di_size(self, ois):
        prima = [ois.zero_rates_at(instrument.maturity) for instrument in ois.instruments]
        with ois.shocked_zeros(UN_BP):
            dopo = [ois.zero_rates_at(instrument.maturity) for instrument in ois.instruments]
        assert dopo == pytest.approx([rate + UN_BP for rate in prima])

    def test_un_nodo_solo_muove_solo_quel_pilastro(self, ois):
        nodo = nodo_del_tenor(ois, "10Y")
        prima = [ois.zero_rates_at(instrument.maturity) for instrument in ois.instruments]
        with ois.shocked_zeros(UN_BP, node=nodo):
            dopo = [ois.zero_rates_at(instrument.maturity) for instrument in ois.instruments]
        atteso = list(prima)
        atteso[nodo] += UN_BP
        assert dopo == pytest.approx(atteso)

    def test_le_quotazioni_non_sono_piu_riprezzate(self, ois):
        with ois.shocked_zeros(UN_BP):
            assert errore_di_riprezzamento(ois) > UN_BP / 2

    def test_la_curva_e_congelata_dentro_l_ambito(self, ois):
        with ois.shocked_zeros(UN_BP):
            assert ois._frozen
        assert not ois._frozen

    def test_lo_stato_torna_come_prima(self, ois):
        prima = stato(ois)
        with ois.shocked_zeros(UN_BP):
            pass
        assert stato(ois) == prima

    def test_i_tassi_tornano_come_prima_dopo_l_ambito(self, ois):
        prima = ois.zero_rates_at(DIECI_ANNI)
        with ois.shocked_zeros(UN_BP):
            ois.zero_rates_at(DIECI_ANNI)
        assert ois.zero_rates_at(DIECI_ANNI) == pytest.approx(prima)

    def test_un_eccezione_non_lascia_la_curva_shockata(self, ois):
        prima = stato(ois)
        with pytest.raises(RuntimeError):
            with ois.shocked_zeros(UN_BP):
                raise RuntimeError
        assert stato(ois) == prima
        assert not ois._frozen


class TestPropagazione:
    """
    Le due modalita' si riconoscono da cosa resta fermo sulla curva di proiezione:
    sotto sticky quotes restano le sue quotazioni, sotto sticky zero i suoi tassi.
    """

    def test_quotes_sticky_quotes_la_euribor_riprezza_ancora_le_sue(self, ois, euribor):
        with ois.shocked_quotes(UN_BP):
            assert errore_di_riprezzamento(euribor) == pytest.approx(0.0, abs=1e-12)

    def test_quotes_sticky_quotes_i_tassi_euribor_si_muovono(self, ois, euribor):
        prima = euribor.zero_rates_at(DIECI_ANNI)
        with ois.shocked_quotes(UN_BP):
            assert euribor.zero_rates_at(DIECI_ANNI) != prima

    def test_quotes_sticky_zero_i_tassi_euribor_non_si_muovono(self, ois, euribor):
        prima = euribor.zero_rates_at(DIECI_ANNI)
        with euribor.frozen(), ois.shocked_quotes(UN_BP):
            assert euribor.zero_rates_at(DIECI_ANNI) == prima

    def test_quotes_sticky_zero_la_euribor_non_riprezza_piu(self, ois, euribor):
        with euribor.frozen(), ois.shocked_quotes(UN_BP):
            assert errore_di_riprezzamento(euribor) > 0.0

    def test_zero_sticky_quotes_la_euribor_riprezza_ancora_le_sue(self, ois, euribor):
        with ois.shocked_zeros(UN_BP):
            assert errore_di_riprezzamento(euribor) == pytest.approx(0.0, abs=1e-12)

    def test_zero_sticky_quotes_i_tassi_euribor_si_muovono(self, ois, euribor):
        prima = euribor.zero_rates_at(DIECI_ANNI)
        with ois.shocked_zeros(UN_BP):
            assert euribor.zero_rates_at(DIECI_ANNI) != prima

    def test_zero_sticky_zero_i_tassi_euribor_non_si_muovono(self, ois, euribor):
        prima = euribor.zero_rates_at(DIECI_ANNI)
        with euribor.frozen(), ois.shocked_zeros(UN_BP):
            assert euribor.zero_rates_at(DIECI_ANNI) == prima

    def test_sticky_spread_lascia_fermo_il_basis(self, ois, euribor):
        prima = euribor.zero_rates_at(DIECI_ANNI) - ois.zero_rates_at(DIECI_ANNI)
        with ois.shocked_zeros(UN_BP), euribor.shocked_zeros(UN_BP):
            dentro = euribor.zero_rates_at(DIECI_ANNI) - ois.zero_rates_at(DIECI_ANNI)
        assert dentro == pytest.approx(prima, abs=1e-15)

    def test_la_ois_non_dipende_dalla_euribor(self, ois, euribor):
        prima = ois.zero_rates_at(DIECI_ANNI)
        with euribor.shocked_quotes(UN_BP):
            assert ois.zero_rates_at(DIECI_ANNI) == prima
        with euribor.shocked_zeros(UN_BP):
            assert ois.zero_rates_at(DIECI_ANNI) == prima

    def test_i_pilastri_sanno_gia_che_la_curva_e_da_rifare(self, ois, euribor):
        euribor.zero_rates_at(DIECI_ANNI)
        prima = euribor.pillars
        with ois.shocked_quotes(UN_BP):
            assert euribor.pillars != prima

    def test_la_euribor_torna_da_sola_dopo_l_ambito(self, ois, euribor):
        prima = euribor.zero_rates_at(DIECI_ANNI)
        with ois.shocked_quotes(UN_BP):
            euribor.zero_rates_at(DIECI_ANNI)
        assert euribor.zero_rates_at(DIECI_ANNI) == pytest.approx(prima)

    def test_un_congelamento_annidato_non_scongela_uscendo(self, ois, euribor):
        prima = euribor.zero_rates_at(DIECI_ANNI)
        with euribor.frozen():
            with euribor.frozen():
                pass
            with ois.shocked_quotes(UN_BP):
                assert euribor.zero_rates_at(DIECI_ANNI) == prima

    def test_congelare_una_curva_gia_disallineata_la_congela_disallineata(self, ois, euribor):
        prima = euribor.zero_rates_at(DIECI_ANNI)
        with ois.shocked_quotes(UN_BP):
            euribor.zero_rates_at(DIECI_ANNI)
        with euribor.frozen(), ois.shocked_quotes(UN_BP):
            assert euribor.zero_rates_at(DIECI_ANNI) != prima


class TestShiftVettoriale:
    """
    'size' puo' essere uno scalare, e allora vale per tutti i nodi, oppure un vettore
    lungo quanto gli strumenti, e allora ogni nodo si muove del suo.
    """

    def test_uno_scalare_equivale_a_un_vettore_costante(self, ois):
        with ois.shocked_quotes(UN_BP):
            scalare = list(ois._log_df)
        with ois.shocked_quotes(np.full(len(ois.instruments), UN_BP)):
            vettore = list(ois._log_df)
        assert scalare == pytest.approx(vettore)

    def test_un_vettore_muove_ogni_quotazione_della_sua(self, ois):
        sizes = np.linspace(UN_BP, -UN_BP, len(ois.instruments))
        base = [instrument.quote for instrument in ois.instruments]
        with ois.shocked_quotes(sizes):
            dentro = [instrument.quote for instrument in ois.instruments]
        assert dentro == pytest.approx([quote + size for quote, size in zip(base, sizes)])
        assert [instrument.quote for instrument in ois.instruments] == base

    def test_un_vettore_muove_ogni_zero_del_suo(self, ois):
        sizes = np.linspace(UN_BP, -UN_BP, len(ois.instruments))
        prima = [ois.zero_rates_at(instrument.maturity) for instrument in ois.instruments]
        with ois.shocked_zeros(sizes):
            dopo = [ois.zero_rates_at(instrument.maturity) for instrument in ois.instruments]
        assert dopo == pytest.approx([rate + size for rate, size in zip(prima, sizes)])

    def test_uno_shift_di_pendenza_muove_i_due_capi_in_versi_opposti(self, ois):
        sizes = np.linspace(UN_BP, -UN_BP, len(ois.instruments))
        corto, lungo = ois.instruments[0].maturity, ois.instruments[-1].maturity
        prima_corto, prima_lungo = ois.zero_rates_at(corto), ois.zero_rates_at(lungo)
        with ois.shocked_zeros(sizes):
            assert ois.zero_rates_at(corto) > prima_corto
            assert ois.zero_rates_at(lungo) < prima_lungo

    def test_un_vettore_insieme_a_un_nodo_solleva(self, ois):
        with pytest.raises(ValueError):
            with ois.shocked_quotes(np.full(len(ois.instruments), UN_BP), node=0):
                pass

    def test_un_vettore_di_lunghezza_sbagliata_solleva(self, ois):
        with pytest.raises(ValueError):
            with ois.shocked_zeros(np.full(3, UN_BP)):
                pass

    def test_le_quotazioni_restano_float(self, ois):
        with ois.shocked_quotes(np.full(len(ois.instruments), UN_BP)):
            assert all(type(instrument.quote) is float for instrument in ois.instruments)

    def test_i_pilastri_restano_float(self, ois):
        with ois.shocked_zeros(np.full(len(ois.instruments), UN_BP)):
            assert all(type(value) is float for value in ois._log_df)


class TestShockPerTenor:
    """
    Un tenor che non e' un pilastro: la curva ne inserisce uno per la durata dell'ambito,
    cosi' lo shock resta un triangolo con la punta li' e zero sui pilastri vicini.
    """

    def test_alza_il_tasso_di_size_a_quel_tenor(self, ois):
        data = TRADE_DATE + tenor_offset("15Y")
        prima = ois.zero_rates_at(data)
        with ois.shocked_zeros(UN_BP, node="15Y"):
            assert ois.zero_rates_at(data) == pytest.approx(prima + UN_BP)

    def test_nessun_pilastro_si_muove(self, ois):
        prima = [ois.zero_rates_at(instrument.maturity) for instrument in ois.instruments]
        with ois.shocked_zeros(UN_BP, node="15Y"):
            dopo = [ois.zero_rates_at(instrument.maturity) for instrument in ois.instruments]
        assert dopo == pytest.approx(prima, abs=1e-15)

    def test_il_tratto_prima_del_tenor_resta_fermo(self, ois):
        corto = TRADE_DATE + tenor_offset("5Y")
        prima = ois.zero_rates_at(corto)
        with ois.shocked_zeros(UN_BP, node="15Y"):
            assert ois.zero_rates_at(corto) == prima

    def test_il_pilastro_inserito_sparisce_all_uscita(self, ois):
        quanti = len(ois._times)
        with ois.shocked_zeros(UN_BP, node="15Y"):
            assert len(ois._times) == quanti + 1
            assert len(ois._log_df) == quanti + 1
        assert len(ois._times) == quanti
        assert len(ois._log_df) == quanti

    def test_una_data_su_un_pilastro_non_ne_aggiunge_e_vale_come_l_indice(self, ois):
        nodo = nodo_del_tenor(ois, "10Y")
        quanti = len(ois._times)
        with ois.shocked_zeros(UN_BP, node=nodo):
            per_indice = list(ois._log_df)
        with ois.shocked_zeros(UN_BP, node=ois.instruments[nodo].maturity):
            assert len(ois._times) == quanti
            assert ois._log_df == pytest.approx(per_indice)

    def test_un_tenor_oltre_l_ultimo_pilastro(self, ois):
        data = TRADE_DATE + tenor_offset("60Y")
        prima = ois.zero_rates_at(data)
        with ois.shocked_zeros(UN_BP, node="60Y"):
            assert ois.zero_rates_at(data) == pytest.approx(prima + UN_BP)

    def test_sticky_spread_per_tenor_lascia_fermo_il_basis(self, ois, euribor):
        data = TRADE_DATE + tenor_offset("15Y")
        prima = euribor.zero_rates_at(data) - ois.zero_rates_at(data)
        with ois.shocked_zeros(UN_BP, node="15Y"), euribor.shocked_zeros(UN_BP, node="15Y"):
            dentro = euribor.zero_rates_at(data) - ois.zero_rates_at(data)
        assert dentro == pytest.approx(prima, abs=1e-15)


# ---------------------------------------------------------------------------
# Lo Jacobiano: come le quotazioni implicite reagiscono ai tassi zero
# ---------------------------------------------------------------------------

class TestJacobian:

    def test_una_riga_per_strumento_e_una_colonna_per_pilastro(self, ois):
        assert ois.jacobian.shape == (len(ois.instruments), len(ois.instruments))

    def test_triangolare_inferiore(self, ois):
        assert np.abs(np.triu(ois.jacobian, 1)).max() == 0.0

    def test_la_diagonale_e_positiva_e_di_ordine_uno(self, ois):
        diagonale = np.diag(ois.jacobian)
        assert (diagonale > 0.5).all() and (diagonale < 1.5).all()

    def test_invertibile(self, ois):
        assert np.linalg.cond(ois.jacobian) < 1e5

    def test_non_lascia_traccia_sulla_curva(self, ois):
        prima = list(ois._log_df)
        ois.jacobian
        assert ois._log_df == prima
        assert errore_di_riprezzamento(ois) == pytest.approx(0.0, abs=1e-12)

    def test_l_inversa_riproduce_il_ribootstrap(self, ois):
        """
        Muovo una quotazione di h e ribootstrappo: i pilastri si spostano di dp. La stessa
        dp deve uscire da J^-1 senza nessun bootstrap.
        """
        h = 1e-6
        nodo = nodo_del_tenor(ois, "5Y")
        prima = np.array(ois._log_df[1:])
        with ois.shocked_quotes(h, node=nodo):
            dopo = np.array(ois._log_df[1:])
        dp_dq_ribootstrap = (dopo - prima) / h
        dz_dq = np.linalg.inv(ois.jacobian)[:, nodo]
        dp_dq_jacobiano = -np.array(ois._times[1:]) * dz_dq
        assert dp_dq_jacobiano == pytest.approx(dp_dq_ribootstrap, rel=1e-5, abs=1e-9)

    def test_un_pilastro_prima_del_nodo_non_si_muove(self, ois):
        nodo = nodo_del_tenor(ois, "5Y")
        dz_dq = np.linalg.inv(ois.jacobian)[:, nodo]
        assert dz_dq[:nodo] == pytest.approx(np.zeros(nodo), abs=1e-12)

    def test_in_cache_finche_la_curva_non_cambia(self, ois):
        assert ois.jacobian is ois.jacobian

    def test_ricalcolato_dentro_e_dopo_uno_shock(self, ois):
        base = ois.jacobian
        with ois.shocked_zeros(UN_BP):
            dentro = ois.jacobian
        dopo = ois.jacobian
        assert dentro is not base
        assert dopo is not base and dopo is not dentro
        assert dopo == pytest.approx(base)

    def test_si_allinea_alla_sorgente_prima_di_calcolare(self, ois, euribor):
        euribor.zero_rates_at(DIECI_ANNI)
        base = euribor.jacobian
        with ois.shocked_quotes(UN_BP):
            assert euribor.jacobian is not base
            assert errore_di_riprezzamento(euribor) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Il blocco fuori diagonale: come si muovono gli zero rate della Euribor quando
# si muove uno zero rate della OIS, con le quotazioni Euribor ferme
# (Henrard 2013, par. 3.6, teorema della funzione implicita)
# ---------------------------------------------------------------------------

def zero_rates(curve):
    return -np.asarray(curve._log_df[1:]) / np.asarray(curve._times[1:])


class TestDependencyJacobian:

    def test_ha_una_riga_per_zero_rate_mio_e_una_colonna_per_zero_rate_della_sorgente(self, ois, euribor):
        assert euribor.dependency_jacobian(ois).shape == (len(euribor.instruments), len(ois.instruments))

    def test_coincide_con_il_ribootstrap(self, ois, euribor):
        X = euribor.dependency_jacobian(ois)
        base, h = zero_rates(euribor), 1e-6
        for j in range(len(ois.instruments)):
            with ois.shocked_zeros(h, node=j):
                euribor._ensure_current()
                assert (zero_rates(euribor) - base) / h == pytest.approx(X[:, j], abs=1e-5)

    def test_e_nullo_se_non_dipendo_dalla_sorgente(self, ois, euribor):
        assert ois.dependency_jacobian(euribor) == pytest.approx(np.zeros((len(ois.instruments),
                                                                           len(euribor.instruments))))

    def test_non_ribootstrappa_nessuna_delle_due(self, ois, euribor, monkeypatch):
        euribor.jacobian
        chiamate = []
        for curve in (ois, euribor):
            originale = curve._bootstrap
            monkeypatch.setattr(curve, "_bootstrap", lambda originale=originale: chiamate.append(1) or originale())
        euribor.dependency_jacobian(ois)
        assert chiamate == []

    def test_non_lascia_traccia(self, ois, euribor):
        prima = ois.pillars, euribor.pillars
        euribor.dependency_jacobian(ois)
        assert (ois.pillars, euribor.pillars) == prima
        assert not ois._frozen and not euribor._frozen


# ---------------------------------------------------------------------------
# La curva di Coleman (2011), "A Guide to Duration, DV01, and Yield Curve Risk
# Transformations": quattro swap par USD, forward costanti a tratti.
# ---------------------------------------------------------------------------

class TestColeman:

    @pytest.fixture
    def curva(self):
        quotes = {"1Y": 0.02, "2Y": 0.025, "5Y": 0.03, "10Y": 0.035}
        return YieldCurve([Swap(tenor, quote, TRADE_DATE, frequency=2, dcc="30/360")
                           for tenor, quote in quotes.items()], "curva", TRADE_DATE)

    def test_i_forward_a_tratti_sono_quelli_di_tabella_3(self, curva):
        forward = -np.diff(curva._log_df) / np.diff(curva._times)
        assert forward == pytest.approx([0.0199, 0.0299, 0.0333, 0.0406], abs=2e-4)

    def test_i_prezzi_degli_zero_sono_quelli_di_tabella_4(self, curva):
        prezzi = 100 * np.exp(curva._log_df[1:])
        assert prezzi == pytest.approx([98.03, 95.14, 86.09, 70.28], abs=0.03)

    def test_il_rischio_del_10y_zero_rispetto_ai_par_swap_e_tabella_10(self, curva):
        from QuantGYMM.fixed_income.bonds import ZeroCouponBond
        from QuantGYMM.fixed_income.pricers import Pricer
        zero = ZeroCouponBond(maturity_date=curva.instruments[-1].maturity, face_amount=100.0)
        zero.set_evaluation_date(TRADE_DATE)
        zero.set_pricer(Pricer(curva))
        par = [-v * 100 for v in zero.key_rate_dv01(bump="quotes", sticky="quotes")[curva.name].values()]
        assert par == pytest.approx([-0.03, -0.11, -0.54, 7.70], abs=0.015)
        zeri = [-v * 100 for v in zero.key_rate_dv01(bump="zeros", sticky="zeros")[curva.name].values()]
        assert zeri == pytest.approx([0.0, 0.0, 0.0, 7.02], abs=0.03)

