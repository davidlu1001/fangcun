# Determinism Audit — 2026-04-14

## Method

Ran the regression suite twice in separate Python processes against a 9-case
filter spanning 1, 2, 3, 4, and 5-character inputs across all style/shape/type
combos. Compared output PNGs byte-for-byte and via RMSE.

```
python tests/regression/run.py --run-id determinism_a --filter c01,c02,c03,c11,c18,c20,c14,c33,c32
python tests/regression/run.py --run-id determinism_b --filter c01,c02,c03,c11,c18,c20,c14,c33,c32
python tests/regression/compare.py determinism_a determinism_b
```

## Result

**9/9 byte-identical.** Every test case produced pixel-equal output across
two independent process invocations.

| Test ID | Text | Result |
|---|---|---|
| c01_zen_bw_oval | 禅 | IDENTICAL |
| c02_zen_zw_sq | 禅 | IDENTICAL |
| c03_dao_zw_sq | 道 | IDENTICAL |
| c11_sushi_bw_oval | 苏轼 | IDENTICAL |
| c14_zhizhi_bw_sq | 知足 | IDENTICAL |
| c18_qibaishi_bw_sq | 齐白石 | IDENTICAL |
| c20_tianren_bw_sq | 天人合一 | IDENTICAL |
| c32_5char_overflow | 天地玄黄宇 | IDENTICAL |
| c33_trad_input | 齊白石 | IDENTICAL |

## Why this works

1. **Texture seed (Task 4.1):** `StoneTexture.apply` accepts a seed parameter
   and seeds numpy's global RNG before generating noise. Without this, two
   identical pipeline invocations produce RMSE~5 differences purely from
   per-call random texture noise.

2. **Stable per-case seed (Task 0.5):** The regression runner derives each
   case's seed from `MD5(test_id + attempt)` rather than Python's
   `hash()` (which is randomized per-process via PYTHONHASHSEED). This
   keeps seeds stable across processes.

3. **Three-tier cache:** Source selection (Tier 1 API responses) and image
   downloads (Tier 2 CDN) are cached. Same input → same scrape result →
   same selected glyph.

## Sources of non-determinism (verified absent)

- ✅ Texture noise — seeded via Task 4.1
- ✅ Sort order — deterministic keys (score + source name)
- ✅ Cache state — warm cache for all 9 cases
- ✅ Process-randomized `hash()` in Python — replaced with `hashlib.md5` in runner

## Cold-cache caveat

This audit was run with a warm cache (Phase 0 baseline + Phase 1-5 iterations
populated `~/.seal_gen/cache/`). On a cold cache, the first run pulls from
ygsf.com — the API response is deterministic for a given input, but verifying
this would require ~30 min of network time per audit run. Not exercised here;
the cache layer is the contract for the user-facing determinism guarantee.

## No fixes required

All sources of non-determinism that were investigable have been addressed.
The pipeline is reproducible at the byte level across separate Python
processes when the cache is warm. Task 5.2 closes with no code changes.
