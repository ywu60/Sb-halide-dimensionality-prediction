Determine whether this evidence unit contains information relevant to the
specified compound and target category.

Use semantic meaning, not keyword presence alone. A connectivity passage may
be relevant when it describes shared halides, isolated units, periodic
extension, secondary contacts, or explicitly nonbonding contacts without
using the words "dimensionality" or "connectivity". A cation passage may be
relevant when it names, abbreviates, or gives the formula of the organic or
inorganic cation. A synthesis passage may be relevant when it describes
reagents, conditions, crystallization, or yield for the target compound.

You are given `compound_aliases` — the labels, name, and formula this
specific compound is called in the paper (e.g. "compound 1", "2",
"[SbCl2(L)]"). Papers in this dataset routinely report several closely
related compounds (different halides, different numbering, e.g. compounds
1-5 built from the same ligand). A passage is relevant only if it concerns
THIS compound specifically. If a passage clearly discusses a different
numbered compound, a different halide, or a different formula than the one
named in `compound_aliases`, mark it not relevant even if it is structurally
very similar — do not let similarity substitute for identity.

Reject the passage if it concerns another compound, a precursor, a
comparison material, or general background unrelated to the target category.

Return, for every supplied evidence unit: `is_relevant`, `relevance_type`
(a short label such as "cation_name", "bridging_halide", "nonbonding_contact",
"yield"), and the `source_id`. Return only the required JSON schema.
