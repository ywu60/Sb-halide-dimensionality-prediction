Determine the dimensionality of the periodic network formed only by direct
Sb-halide bonds for the specified compound.

0D: direct Sb-halide bonding terminates in a monomer, dimer, or finite cluster.
1D: direct bonding extends infinitely in one independent direction.
2D: direct bonding extends infinitely in two independent directions.
3D: direct bonding extends infinitely in three independent directions.
Unknown: the evidence is insufficient or conflicting.

Exclude hydrogen bonds, organic packing, halide-halide contacts, Sb-N/O/S/P
bonds, proximity between separate units, and Sb...halide contacts explicitly
described as nonbonding. Do not classify a pseudo-chain as 1D unless direct
Sb-halide bonds connect the units periodically.

Required procedure: first identify the finite Sb-halide unit, then identify
direct Sb-X-Sb connections, determine their periodic directions, and check
for statements that the apparent connections are nonbonding.

If the evidence status is not `sufficient`, the label must be `Unknown` — do
not guess a dimensionality from an incomplete or conflicting description.

Return `sb_halide_dimensionality_llm`, a concise source-grounded
`dimensionality_reasoning` (a decision record, not unrestricted chain of
thought), `dimensionality_evidence_status`
(`sufficient`/`insufficient`/`conflicting`), `dimensionality_review_reason`
(null unless evidence is weak), and the source IDs used.
