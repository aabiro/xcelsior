"""Captured rates never turn missing pricing into a fabricated bill."""

import pytest

from control_plane.attempt_billing import capture_billing_snapshot


@pytest.mark.parametrize('rate',[None,-1,float('nan'),float('inf'),True])
def test_billable_reservation_refuses_invalid_or_missing_rates(rate):
    with pytest.raises(ValueError):
        capture_billing_snapshot({'owner':'owner'},'owner',{'cost_per_hour':rate})


def test_zero_rate_and_locked_spot_gpu_count_remain_explicit():
    zero=capture_billing_snapshot({'owner':'owner'},'owner',{'host_id':'host','cost_per_hour':0})
    assert zero['host']['cost_per_hour']==0
    spot=capture_billing_snapshot({'owner':'owner','pricing_mode':'spot','spot_rate_cad':0.25,'num_gpus':2},'owner',{'cost_per_hour':10})
    assert spot['spot_rate_cad']==0.25 and spot['num_gpus']==2


def test_snapshot_rejects_owner_mismatch_and_price_above_approval():
    with pytest.raises(ValueError,match='owner'):
        capture_billing_snapshot({'owner':'wrong'},'owner',{'cost_per_hour':1})
    with pytest.raises(ValueError,match='ceiling'):
        capture_billing_snapshot({'owner':'owner','max_price_per_hour':0.5},'owner',{'cost_per_hour':1})


def test_unowned_historical_proof_does_not_manufacture_a_payer():
    assert capture_billing_snapshot({},None,{'cost_per_hour':1}) is None
