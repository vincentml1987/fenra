import json
import os

from pdv_utils import apply_and_persist_pdv_adjustments


def test_delta_pct_and_persistence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs('confs', exist_ok=True)

    globals_cfg = {
        'model': 'm',
        'discord_pdv_name': '',
        'discord_pdv_scale': 0.0,
    }
    (tmp_path / 'confs' / 'globals.json').write_text(
        json.dumps(globals_cfg), encoding='utf-8'
    )

    pdvs_cfg = {
        'pdvs': [
            {
                'name': 'Attentiveness',
                'description': '',
                'value': 0.5,
            }
        ]
    }
    (tmp_path / 'confs' / 'pdvs.json').write_text(
        json.dumps(pdvs_cfg), encoding='utf-8'
    )

    res = apply_and_persist_pdv_adjustments([
        {'name': 'Attentiveness', 'delta_pct': 0.05}
    ])
    assert abs(res['Attentiveness'] - 0.55) < 1e-9

    res2 = apply_and_persist_pdv_adjustments([
        {'pdv': 'Attentiveness', 'delta': 0.1}
    ])
    assert res2['Attentiveness'] > res['Attentiveness']

    hist_path = tmp_path / 'chatlogs' / 'pdv_history.jsonl'
    live_path = tmp_path / 'chatlogs' / 'pdvs_live.json'
    assert hist_path.exists()
    assert live_path.exists()

    last_entry = json.loads(hist_path.read_text().strip().splitlines()[-1])
    assert last_entry['pdvs']['Attentiveness'] == res2['Attentiveness']
    live_vals = json.loads(live_path.read_text())
    assert live_vals['Attentiveness'] == res2['Attentiveness']
