# Mandatory guard: latest Local Search file for sensitivity rerun

This guard identifies the public release's cached-LP implementation.

Before running or continuing any sensitivity experiments, verify and use this exact Local Search implementation:

- Primary path:
  `src/test/test_FCPLS_score_cached_lp.py`
- Compatibility path:
  `src/test/test_FCPLS_score_cached_lp_1.py`
- Expected sha256 for BOTH files:
  `da4e9c4942eb2b2d993d90a521a2be77952f26ede6a072272260416c577839dd`

Commands to verify:

```bash
# Run from the repository root.
python3 - <<'PY'
from pathlib import Path
import hashlib
expected = 'da4e9c4942eb2b2d993d90a521a2be77952f26ede6a072272260416c577839dd'
paths = [Path('src/test/test_FCPLS_score_cached_lp.py'), Path('src/test/test_FCPLS_score_cached_lp_1.py')]
for p in paths:
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    print(p, h)
    assert h == expected, (p, h)
print('LATEST_LOCAL_SEARCH_VERIFIED')
PY
python3 -m py_compile src/test/test_FCPLS_score_cached_lp.py src/test/test_FCPLS_score_cached_lp_1.py
```

Experiment requirement:
- K justify and LP-MILP consistency MUST use this updated cached-LP Local Search implementation, not stale imports from `LS_Path_Test.py` or `test_FCP_LS.py` unless those runners are explicitly patched/wrapped to call this implementation.
- If an existing runner imports stale Local Search code, patch or wrap the runner before execution.
- Report the verification command and the sha in the final markdown report.
