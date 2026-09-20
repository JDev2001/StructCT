# Trial eligibility graphs

This dataset contains 103,878 graph rows derived from clinical-trial eligibility
criteria. Inclusion and exclusion criteria are stored as separate rows. It is the
complete dataset after applying the graph-generation/filtering process described
in the accompanying paper artifact.

## Schema

| Column | Type | Description |
|---|---|---|
| `nct_id` | string | ClinicalTrials.gov trial identifier |
| `type` | string | `inclusion` or `exclusion` |
| `entities` | string (JSON) | Extracted eligibility entities |
| `relations` | string (JSON) | Typed relations between entity identifiers |

The graph outputs are automatically extracted and may contain omissions or
incorrect entities, relations, numeric values, or negations.
