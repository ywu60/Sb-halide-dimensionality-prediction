Construct a complete evidence text for the specified compound and category
using only the supplied evidence units.

You are given `compound_aliases` for the target compound. Papers in this
dataset frequently report several closely related compounds (different
halides, different numbering, sharing the same ligand or scaffold). Some
supplied evidence units may actually describe a *different* compound from
the same paper — before using any sentence, confirm it is tied to one of the
target compound's aliases (its number/label, name, or formula) rather than a
different numbered compound. When a unit does not specify which compound it
concerns, use surrounding context (preceding/following units, table rows) to
decide; if it cannot be confidently attributed to the target compound, leave
it out rather than guessing.

Combine information that is distributed across passages, tables, and
captions. Keep the target compound explicit. Preserve qualifications,
uncertainty, and contradictions rather than smoothing them away. Do not
introduce chemical knowledge that is not stated in the sources. Attach
source IDs to every factual sentence, inline, e.g. "Sources: S4 and S11."
If no explicit evidence supports a requested item, state plainly that it was
not reported rather than omitting it silently.

Category-specific guidance:

- **cation**: reported cation name, alternative name, abbreviation,
  protonation information, explicitly stated cation formula, and any
  ambiguity about whether a formula represents the cation or the complete
  compound. Many compounds here are neutral coordination complexes with NO
  discrete cation at all (the metal and its ligands form one neutral
  molecule, not an ion pair). If the retrieved units only describe the
  neutral compound itself, with no distinct cationic species paired against
  a counter-anion, state explicitly that this compound has no discrete
  cation reported — do not restate the compound's own name/formula as if it
  were a cation.
- **connectivity**: reported Sb-halide building unit; isolated/discrete
  units; monomers, dimers, finite clusters; bridging halides; corner-,
  edge-, or face-sharing polyhedra; infinite chains, layers, or frameworks;
  crystallographic propagation directions; relevant Sb-halide bond
  distances; long Sb...halide contacts; secondary bonding; explicitly
  nonbonding contacts; pseudo-chains or packing descriptions. Do not assign
  0D/1D/2D/3D at this stage — that happens in a separate reasoning step.
- **synthesis**: target compound name or label, reagents and quantities,
  solvents, reaction conditions, temperature and time, order of addition,
  cooling or evaporation, crystallization, washing and drying, yield, and
  other relevant procedural information.

Return only the required JSON schema: `evidence_text`, `source_ids`, and
`not_reported` (true only if literally nothing relevant was found).
