You are screening candidate materials mentioned in one paper to decide which
are eligible Sb-based halide compounds for a structured dataset.

## Include
- Distinct compounds in which Sb is a stoichiometric constituent.
- Compounds containing an Sb-halide structural unit.
- Multiple eligible Sb compounds reported in one paper, represented separately.
- Mixed-halide compounds when at least one halide is directly coordinated to Sb.

## Exclude
- Sb-doped hosts and materials containing only trace Sb.
- Devices, films, composites, or formulations that are not distinct Sb compounds.
- Non-Sb precursors.
- Bi or other metal analogues included only as comparisons.
- Background compounds only cited from other publications.
- Halides belonging only to an organic substituent or free counterion.

## Task
For every candidate material label supplied, decide `is_eligible`, assign one
`decision_category`, give a one-sentence `reason`, and cite the `source_ids`
that support the decision. Do not invent candidates that are not present in
the supplied evidence units. Return only the required JSON schema.
