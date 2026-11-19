# qwen3omni-instr_av_greedy_check50 vs paper Table 2 (Qwen3-Omni-Instruct-30B-A3B)

Subset: check50 (n=50); decoding: greedy. Values in %. Δ = ours − paper.

**Warning: not the full benchmark; the paper's numbers are on all 1027 questions.**

| level | paper Acc / C.G. / B.G. | strict Acc / C.G. / B.G. (Δ) | permissive Acc / C.G. / B.G. (Δ) | permissive, no free text Acc / C.G. / B.G. (Δ) | diagnostic: no-bracket timestamps accepted Acc / C.G. / B.G. (Δ) | n | Acc 95% CI (strict) |
|---|---|---|---|---|---|---|---|
| Level 1 | 28.6 / 5.1 / 0.6 | 33.3 / 21.4 / 8.4 (+4.7 / +16.3 / +7.8) | 38.1 / 21.4 / 8.4 (+9.5 / +16.3 / +7.8) | 38.1 / 21.4 / 8.4 (+9.5 / +16.3 / +7.8) | 38.1 / 21.4 / 8.4 (+9.5 / +16.3 / +7.8) | 21 | [17.2, 54.6] |
| Level 2 | 22.4 / 3.4 / 0.4 | 14.3 / 4.8 / 1.6 (-8.1 / +1.4 / +1.2) | 19.0 / 4.8 / 1.6 (-3.4 / +1.4 / +1.2) | 19.0 / 4.8 / 1.6 (-3.4 / +1.4 / +1.2) | 19.0 / 9.5 / 3.5 (-3.4 / +6.1 / +3.1) | 21 | [5.0, 34.6] |
| Level 3 | 22.2 / 2.4 / 0.1 | 12.5 / 0.0 / 0.0 (-9.7 / -2.4 / -0.1) | 37.5 / 0.0 / 0.0 (+15.3 / -2.4 / -0.1) | 37.5 / 0.0 / 0.0 (+15.3 / -2.4 / -0.1) | 37.5 / 0.0 / 0.0 (+15.3 / -2.4 / -0.1) | 8 | [2.2, 47.1] |
| Overall | 25.2 / 4.0 / 0.4 | 22.0 / 11.0 / 4.2 (-3.2 / +7.0 / +3.8) | 30.0 / 11.0 / 4.2 (+4.8 / +7.0 / +3.8) | 30.0 / 11.0 / 4.2 (+4.8 / +7.0 / +3.8) | 30.0 / 13.0 / 5.0 (+4.8 / +9.0 / +4.6) | 50 | [12.8, 35.2] |

Baselines (Acc. overall): always 'A' 26.0, chance 12.5.
