# Batched long-reduction split-K candidate

## Problem and scope

The SIMD split-K path was restricted to one output matrix. Long batched reductions therefore used a single full-K loop per output tile. An observed attention probability/value multiplication has M512,N256,K115768 with24 query heads and grouped value-head broadcasting. A standalone synthetic fallback attention graph reproduced Metal Impacting Interactivity0x0e on an affected M2Max; another M2Max passed the same short test. This identifies a reproducing workload, not the private driver cause.

This candidate extends existing SIMD FP32 split-K kernels to eligible batches. It does not change model parameters, sequence length, attention normalization, masks or model caches. It is not yet a proven prevention for the reported Metal error, nor a fix for separately observed fused attention failures.

## Implementation

For M2 Max (architecture generation14, variant s), non-NAX FP16/BF16 matmul, batches2–64, K>=65536 and the existing small-output tile threshold, compute each batch's input offsets from collapsed broadcast strides. Reuse the existing split-K and FP32 accumulation kernels. Select enough partitions to target approximately4096 K elements, capped at32; retain the existing unbatched choice. Each batch reuses one scratch buffer. Command encoder read/write barriers serialize reuse. Output offsets use byte addressing. AddMM stays outside this new dispatch route; existing unbatched and NAX dispatch remain unchanged. All other Mac architecture variants retain their original dispatch route.

These thresholds are conservative dispatch choices, not measured watchdog deadlines or model limits. At K above131072 the32-partition cap no longer bounds partitions to4096. Review performance before broadening coverage. No driver environment workaround is enabled.

## Validation to date

Focused GPU regression exercises BF16/FP16, aligned/unaligned K65536/65537, all four input transpose combinations, and GQA-like broadcasting across six output matrices. It compares GPU results with CPU FP32 reference, checks finiteness and relative RMS. Larger captured-shape A/B uses24 heads, K115768 and independent original/candidate libraries, comparing saved FP32 outputs element by element.

Development machine: physical M5Max with g14s dispatch override; this is not M2 hardware emulation. The development libraries share the retained experimental core build, with only the Metal matmul object replaced for the candidate; normal repository CI remains required. Captured-shape relative RMS0.000128656, max absolute difference0.0000038147. Scratch increases by16MiB. Warm timings in the short initial sample: original148/159ms, candidate146/135ms. This is not a statistically established speedup or model-generation benchmark.

## Remaining acceptance

- Run standard repository build/CI and review candidate routing/offset coverage.
- Validate the concrete candidate against repeated failing baseline on affected M2 hardware; no root-cause or prevention claim from M5 tests.
- Qualify app/runtime consumer only after core acceptance. No automatic runtime pin change.
- Investigate the separate fused-attention failure independently; do not mark it fixed by a GEMM change.

## Upstream maintenance

Changes are confined to Metal matmul host dispatch and a focused regression. Existing Metal kernel signatures and serialized formats are untouched. Rebase this small host change when upstream gains batched split-K support, preserving transpose/broadcast/unaligned-K tests. A source revert restores original routing; no persistent user settings are written.

## Routing/performance follow-up

Candidate routing is restricted to architecture generation14/variant s (M2 Max). On physical M5 native routing, eight representative matmul cases produce bit-identical outputs against the original library; largest case median51.78/51.91ms. With g14s routing override on the same M5, three eligible cases improve in this sample; five excluded cases produce bit-identical results. This does not establish M2 performance. One short excluded case had a noisy30% first comparison; five alternating-order pairs overlapped (baseline0.925–1.191ms, candidate0.906–1.200ms), without a consistent slowdown. Preserve the outlier rather than silently dropping it. Hardware counters/full model token throughput were not measured.
