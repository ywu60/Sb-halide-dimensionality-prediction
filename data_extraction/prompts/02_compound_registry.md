You are identifying eligible Sb-based halide compounds reported in one paper.

Include a material only when Sb is a stoichiometric constituent of a distinct
compound. Exclude trace-Sb materials, Sb-doped hosts, devices, composites,
non-Sb precursors, and comparison compounds. Only materials already marked
`is_eligible: true` in the supplied eligibility decisions should become
registry entries.

Identify every eligible compound. Resolve labels, names, formulas, and
aliases that refer to the same compound, including expressions such as
"compound 1", "complex 1", "the title compound", "the resulting yellow
crystals", a complete compound name, a reported formula, and an abbreviation
used by the authors. Do not merge different compounds.

Assign each compound a stable `compound_id` of the form `{paper_id}_C{n}`
(e.g. `P0001_C1`, `P0001_C2`) in first-mention order. Every field must cite
one or more supplied source IDs. Return only the required JSON schema.
