Verify each draft field against the cited evidence units.

You are given `compound_aliases` — the labels, name, and formula this
compound is called in the paper. Papers in this dataset commonly report
several closely related compounds side by side (different halides, different
numbering, same ligand). For `compound_identity` specifically: fail it if
ANY cited evidence unit actually describes a different numbered compound,
halide, or formula than the ones in `compound_aliases` — this is the most
common failure mode and must be checked first, before assessing whether the
content itself is otherwise plausible.

For `row_sanity` specifically: check the draft record as a whole, not
against the evidence, for basic self-consistency:
  - every formula field uses only real element symbols and standard
    chemical notation — flag any character that looks like a rendering
    artifact (garbled symbols, replacement characters, anything that is not
    a letter, digit, or standard chemical/typographic symbol);
  - halides_bonded_to_sb agrees with the halogens actually present in
    compound_formula_reported (e.g. a chloride formula should not list Br
    as bonded, and vice versa);
  - a compound whose formula is a simple neutral complex (one bracketed
    unit, no separate ionic counterpart) should not carry cation fields
    describing the compound itself as if it were the cation — cation fields
    should be null in that case, not a restatement of the compound name;
  - a compound whose formula is clearly an ion pair (two distinct bracketed
    units, e.g. [cation][anion]) should have cation fields populated.
Fail `row_sanity` if any of these look wrong, and explain which cell and why.

Check that the evidence concerns the correct compound, directly supports the
claim, and does not contain an omitted qualification or contradiction. Pay
special attention to:

- inferred cation formulas presented as if explicitly reported;
- long Sb...halide distances, secondary contacts, and pseudo-chains that may
  have been miscounted as direct bonds;
- packing, hydrogen-bonded, or halide-halide networks counted as Sb-halide
  connectivity; and
- synthesis text that describes a precursor rather than the target compound.

For every field, return `pass`, `fail`, or `uncertain` in `field_checks`.
Set `overall_status` to `supported` only if no field failed. For every
failed or uncertain field, return a targeted `recovery_queries` entry (field
name -> a short retrieval query that could resolve the problem) — do not
rewrite supported fields yourself. Separately report whether counter-evidence
language (e.g. "not considered a chemical bond", "nonbonding", "secondary
contact", "weak contact", "discrete anion", "isolated unit", "pseudo-chain",
"packing interaction") was found and whether it changes the assessment.
