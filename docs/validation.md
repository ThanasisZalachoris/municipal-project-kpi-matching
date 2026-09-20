# Evaluation and evidence boundaries

## What was recovered

The reference notebooks formed a joint text corpus from KPI descriptions and
project text, fitted TF-IDF, computed cosine similarities, and exported top-one
or top-five candidates. They included basic lowercase/trim preprocessing and
an optional English stopword/stemming experiment. The public preprocessing uses
an explicit Unicode-safe contract and no stemming, avoiding an ASCII-only filter
that could discard useful non-English text.

Later experiments explored supervised classification. Scoped code inspection
established several validity problems:

1. Some binary labels were defined by a similarity-score cutoff. Supplying that
   same score as a classifier feature makes reproducing the label circular.
2. Oversampling appeared before the train/test split or CV, allowing related
   original/resampled examples to cross evaluation boundaries.
3. Some text transformations were fitted before CV, exposing held-out corpus
   statistics to the feature construction.
4. Candidate rows sharing one project were not established as independent
   evaluation units. A pair-level split can leak project text across folds.

All legacy performance artifacts are unverified. No selection of a convenient
successful run is reported here, and no claim is made to have corrected and
rerun a defensible classifier on independently labelled private data.

## Current executable validation

This is catalogue-based retrieval. Only catalogue name/description text fits
document frequencies. Query text is transformed through that fixed index and
never changes it. The catalogue is the retrieval reference set, not labelled
classifier training data. Synthetic expectations do not enter feature fitting,
threshold selection code, or ranking.

The fixture deliberately exercises obvious lexical mappings and adverse cases.
Its ambiguous `access` query shares vocabulary with multiple indicators. The
cutoffs are illustrative policy settings; authoring cases to exercise those
branches is not threshold tuning on a representative validation population.

Evaluation exports compare the TF-IDF top candidate with an unweighted Jaccard
baseline and the declared expected candidate. Each case also checks abstention.
Expected-category counts expose imbalance. Identical token-frequency signatures
identify duplicate unigram feature groups (not every semantic near-duplicate).
The duplicate project is intentionally kept to test stable behavior, not counted
as independent empirical evidence.

The report sets supervised metrics and cross-validation to null and marks the
classifier as untrained. No precision/recall/F1 or confusion matrix is justified
by this small, hand-authored, synthetic fixture. Baseline comparison is a
diagnostic, not an estimated advantage over a population of projects.

## Algorithms and decision policy

For catalogue size N, IDF is `log((1+N)/(1+df))+1`. Raw term counts multiply IDF;
each nonempty vector is normalized to Euclidean length one. A query uses only
catalogue vocabulary; unseen terms are reported separately. Cosine is the dot
product of query and KPI vectors. Shared-term products explain its contributions.

Scores are rounded to 12 decimal places for exported ranking and policy decisions.
Ties sort by KPI ID. Margin uses the two highest scores over the full catalogue,
even if the caller requests only one displayed candidate. The configured score
and margin cutoffs are not probabilities or accuracy guarantees. Empty and
out-of-vocabulary queries always abstain. A retained top-k list may contain zero
scores; stable tie order does not turn those entries into meaningful matches.

The baseline is intersection size divided by union size for project/KPI token
sets, without term counts or IDF. A zero-overlap baseline has no positive top
candidate. Neither method understands negation or semantic equivalence.

## Conditions for any future supervised evaluation

Obtain independent project-to-KPI relevance judgments, including legitimate
multi-label mappings and disagreement handling. Keep projects and related text
families together when separating development and held-out evaluation. Split
original examples first. Fit text transformations only inside each training fold;
put any resampling inside that same fold-specific pipeline. Do not create labels
from features used to predict them.

Check class and group support before selecting folds. Report raw and unique-group
support and imbalance; use class weighting or training-fold resampling only when
justified. Compare suitable simple baselines on the identical held-out groups.
Publish per-class metrics and confusion matrices only with adequate, independently
labelled support, and retain abstentions/multi-label behavior in the evaluation
definition. If that support is absent, withhold model/CV metrics as this release
does. None of these future supervised steps is claimed as executed here.
