Extract the reported cation name, abbreviation, and explicitly reported
cation formula from the cation evidence text.

Many compounds in this dataset are neutral coordination complexes (e.g. a
neutral SbX3-ligand adduct) with NO discrete organic or inorganic cation at
all — the antimony center and its ligands form one neutral molecular unit.
If the evidence text only restates the neutral compound's own name or
formula and does not describe a separate, distinct cationic species, return
an EMPTY `cations` list. Do not treat the compound's own name/formula as if
it were a cation just because the word "cation" or "dialdiminium" appears
near it in the evidence — check whether the text is actually describing a
separate ionic species (paired with a distinct counter-anion) or just the
one neutral molecule. A cation entry should only be created when the source
text genuinely distinguishes a cationic component from an anionic one.

Do not derive the formula from the cation name or from the complete compound
formula. Do not use external chemical knowledge. If the formula is not
explicitly tied to the cation specifically (as opposed to the whole
compound), return null for that cation's formula and set its
`cation_formula_status` to `not_reported` or `ambiguous` as appropriate.

Support multiple cations using parallel arrays (`cation_name_reported`,
`cation_abbreviation_reported`, `cation_formula_explicit`,
`cation_formula_status`) rather than concatenating them into one
unparseable string — one array entry per cation, in the same order across
all four arrays. Return only schema-valid JSON with `cation_source` citing
the source IDs used.
