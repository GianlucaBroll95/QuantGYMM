"""Calendario, ACT/ACT ICMA e rateo contro QuantLib 1.43.

Ogni numero atteso viene da QuantLib: calendario TARGET, Schedule con generazione
all'indietro, FixedRateBond, ActualActual ISMA.
"""

import pandas as pd
import pytest

from QuantGYMM.calendar import Schedule
from QuantGYMM.fixed_income.bonds import FixedRateBond
from QuantGYMM.fixed_income.curves import Deposit, YieldCurve
from QuantGYMM.fixed_income.pricers import Pricer
from QuantGYMM.utils import act_act_icma, business_days_before, following, modified_following, preceding

T = pd.Timestamp


def cedole(schedule, dcc, tasso):
    bond = FixedRateBond(schedule, dcc, 100.0, tasso)
    bond.set_evaluation_date(schedule.start_date - pd.Timedelta(days=1))
    return bond


def rateo_al(bond, regolamento):
    bond.set_evaluation_date(business_days_before(T(regolamento), 2))
    assert bond.settlement_date == T(regolamento)
    return bond.accrued_interest()


class TestGiorniLavorativi:

    def test_da_venerdi_santo_si_salta_anche_pasquetta(self):
        assert following(T("2049-04-16")) == T("2049-04-20")

    def test_dalla_vigilia_di_natale_domenica_si_salta_anche_santo_stefano(self):
        assert modified_following(T("2023-12-24")) == T("2023-12-27")

    def test_da_pasquetta_indietro_si_salta_anche_venerdi_santo(self):
        assert preceding(T("2025-04-21")) == T("2025-04-17")

    def test_a_fine_mese_il_modified_following_torna_indietro(self):
        assert modified_following(T("2026-05-31")) == T("2026-05-29")


class TestActActIcma:

    def test_primo_periodo_corto(self):
        assert act_act_icma(T("2026-05-15"), T("2026-11-15"),
                            reference=(T("2025-11-15"), T("2026-11-15")))[0] == pytest.approx(184 / 365, abs=1e-15)

    def test_primo_periodo_di_undici_mesi(self):
        assert act_act_icma(T("2026-02-10"), T("2027-01-09"),
                            reference=(T("2026-01-09"), T("2027-01-09")))[0] == pytest.approx(333 / 365, abs=1e-15)

    def test_prima_cedola_lunga(self):
        assert act_act_icma(T("2025-05-27"), T("2026-11-27"),
                            reference=(T("2025-11-27"), T("2026-11-27")))[0] == pytest.approx(1 + 184 / 365, abs=1e-15)

    def test_un_periodo_regolare_vale_uno_su_frequenza_anche_con_le_date_spostate(self):
        assert act_act_icma(T("2026-11-16"), T("2027-11-15"))[0] == 1.0

    def test_senza_riferimento_il_periodo_e_il_riferimento_di_se_stesso(self):
        assert act_act_icma(T("2026-02-10"), T("2027-01-09"))[0] == pytest.approx(11 / 12, abs=1e-15)


class TestSchedule:

    def test_maturazione_sulle_date_del_contratto_pagamento_il_giorno_lavorativo(self):
        s = Schedule("2026-02-10", "2031-01-09", 1, convention="unadjusted").schedule
        assert (s["startingDate"][0], s["endingDate"][0], s["paymentDate"][0]) == \
               (T("2026-02-10"), T("2027-01-09"), T("2027-01-11"))
        assert (s["endingDate"][1], s["paymentDate"][1]) == (T("2028-01-09"), T("2028-01-10"))

    def test_il_riferimento_del_primo_periodo_e_il_periodo_regolare_che_finisce_alla_prima_cedola(self):
        s = Schedule("2026-02-10", "2031-01-09", 1, convention="unadjusted").schedule
        assert (s["referenceStart"][0], s["referenceEnd"][0]) == (T("2026-01-09"), T("2027-01-09"))
        assert (s["referenceStart"][1], s["referenceEnd"][1]) == (s["startingDate"][1], s["endingDate"][1])

    def test_con_le_date_aggiustate_il_riferimento_parte_dalla_prima_cedola_spostata(self):
        s = Schedule("2024-07-23", "2033-09-14", 4).schedule
        assert (s["startingDate"][0], s["endingDate"][0], s["referenceStart"][0]) == \
               (T("2024-07-23"), T("2024-09-16"), T("2024-06-17"))

    def test_un_emissione_che_si_sposta_sulla_prima_cedola_non_crea_un_periodo_vuoto(self):
        s = Schedule("2022-03-06", "2030-03-07", 2).schedule
        assert len(s["paymentDate"]) == 16
        assert s["startingDate"][0] == T("2022-03-07")

    def test_le_date_si_contano_dalla_scadenza_e_non_scivolano_dopo_febbraio(self):
        s = Schedule("2019-12-29", "2027-01-30", 12, convention="unadjusted", eom=False).schedule
        assert len(s["paymentDate"]) == 86
        assert all(d.day == 30 for d in s["endingDate"] if d.month != 2)

    def test_con_le_date_aggiustate_pagamento_e_fine_maturazione_coincidono(self):
        s = Schedule("2024-07-23", "2033-09-14", 4).schedule
        assert (s["paymentDate"] == s["endingDate"]).all()


class TestCedoleERateo:

    def test_primo_periodo_di_undici_mesi_e_poi_cedole_piene(self):
        bond = cedole(Schedule("2026-02-10", "2031-01-09", 1, convention="unadjusted"), "ACT/ACT ICMA", 0.03875)
        assert bond.cash_flows.coupon.iloc[:2].tolist() == pytest.approx([3.875 * 333 / 365, 3.875], abs=1e-12)

    def test_prima_cedola_corta(self):
        bond = cedole(Schedule("2026-05-15", "2029-11-15", 1, convention="unadjusted"), "ACT/ACT ICMA", 0.03625)
        assert bond.cash_flows.coupon.iloc[0] == pytest.approx(1.82740, abs=1e-5)
        assert bond.cash_flows.paymentDate.iloc[0] == T("2026-11-16")

    @pytest.mark.parametrize("regolamento, atteso", [("2027-01-08", 3.524658), ("2027-01-11", 0.021233)])
    def test_rateo_prima_e_dopo_il_pagamento(self, regolamento, atteso):
        bond = cedole(Schedule("2026-02-10", "2031-01-09", 1, convention="unadjusted"), "ACT/ACT ICMA", 0.03875)
        assert rateo_al(bond, regolamento) == pytest.approx(atteso, abs=1e-6)

    def test_il_rateo_30e360_conta_i_giorni_della_convenzione(self):
        bond = cedole(Schedule("2026-06-30", "2029-06-30", 4, convention="unadjusted"), "30E/360", 0.05)
        assert rateo_al(bond, "2026-07-31") == pytest.approx(0.416667, abs=1e-6)

    def test_lo_sconto_usa_la_data_di_pagamento(self):
        ois = YieldCurve([Deposit(tenor, 0.02, T("2026-06-30")) for tenor in ("1Y", "5Y", "10Y")], "OIS",
                         T("2026-06-30"))
        bond = FixedRateBond(Schedule("2025-02-10", "2031-01-09", 1, convention="unadjusted"), "ACT/ACT ICMA",
                             100.0, 0.03875)
        bond.set_evaluation_date(ois.trade_date)
        bond.set_pricer(Pricer(ois))
        flussi = bond.cash_flows
        atteso = flussi.cashFlow.to_numpy() @ ois.discount_factor_at(flussi.paymentDate)
        assert bond.pricer.present_value(bond) == pytest.approx(atteso, rel=1e-14)
        assert (flussi.paymentDate != flussi.accrualEnd).any()


class TestPeriodiIrregolari:

    def test_prima_cedola_lunga(self):
        schedule = Schedule("2025-05-27", "2029-11-27", 1, convention="unadjusted", first_date="2026-11-27")
        bond = cedole(schedule, "ACT/ACT ICMA", 0.03125)
        assert bond.cash_flows.coupon.iloc[:2].tolist() == pytest.approx([4.700342465753, 3.125], abs=1e-12)
        assert bond.cash_flows.paymentDate.iloc[:2].tolist() == [T("2026-11-27"), T("2027-11-29")]
        periodo = bond.cash_flows.iloc[0]
        assert (periodo.referenceStart, periodo.referenceEnd) == (T("2025-11-27"), T("2026-11-27"))

    def test_ultimo_periodo_corto(self):
        schedule = Schedule("2024-03-15", "2029-01-15", 2, convention="unadjusted", next_to_last_date="2028-09-15")
        bond = cedole(schedule, "ACT/ACT ICMA", 0.05)
        assert bond.cash_flows.coupon.iloc[-1] == pytest.approx(1.685082872928, abs=1e-12)
        periodo = bond.cash_flows.iloc[-1]
        assert (periodo.referenceStart, periodo.referenceEnd) == (T("2028-09-15"), T("2029-03-15"))

    def test_ultimo_periodo_lungo(self):
        schedule = Schedule("2024-03-15", "2029-01-15", 2, convention="unadjusted", next_to_last_date="2028-03-15")
        bond = cedole(schedule, "ACT/ACT ICMA", 0.05)
        assert bond.cash_flows.coupon.iloc[-1] == pytest.approx(4.185082872928, abs=1e-12)

    def test_le_date_fuori_ordine_sollevano(self):
        with pytest.raises(ValueError):
            _ = Schedule("2024-03-15", "2029-01-15", 2, first_date="2030-01-15").schedule
