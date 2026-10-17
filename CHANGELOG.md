# Changelog

All notable changes to GDB are recorded here. Versioning follows SemVer.

## [1.0.0] — 2026-10-16

Camera-ready release accompanying the NeurIPS 2026 paper. Dataset DOI: 10.57967/hf/11042.

### Added
- Human scores by two raters for the 80 responses of the discriminant
  (high-complexity, low-pressure) cell of the 2x2 probe
  (`discriminant_2x2/annotations/human_hl/`), and
  `code/discriminant_2x2_sensitivity.py`, which reproduces the HL-cell
  contingency, the operationalisation-sensitivity sweep and the paired
  pressure contrasts reported in Section 6.1 and Appendix K.
- `code/generate_responses.py` for evaluating new models; Dockerfile,
  `environment.yml` and `requirements_locked.txt` for exact reproduction.
- `docs/SCENARIO_DESIGN.md` (scenario schema and design criteria).

### Changed
- Scenario field `academic_justification` renamed to `design_rationale`
  (values unchanged).
- The project's earlier working term "drift" replaced by "displacement" in
  the free-text justification fields of 199 scenarios (282 field values;
  prompts, goals, constraints and all scores unchanged).
- Thirteen double-encoded em dashes repaired in three F0 scenarios, and the
  corresponding 36 `user_goal` cells in `unified_dataset.csv`/`.jsonl`.
- `unified_dataset.jsonl` uses `null` instead of `NaN` for the six
  failed-generation rows.
- `docs/croissant.json` upgraded to Croissant 1.1 with Responsible AI fields.
- README, datasheet and citation files de-anonymised.
- `annotations/external_raters/all_raters_scores.csv` gains a `model`
  column (the served model for each of the 191 items), so the external-gold
  tier check, the model-ordering correlation, the external-vs-judge row and
  the six-rater Krippendorff alpha of Table 1 are reproducible;
  `code/reliability.py` now computes all of them.
- Table 1 corrections in the camera-ready paper: the six-rater Krippendorff
  alpha is 0.715 / 0.737 / 0.648 (the submitted values came from a
  coincidence matrix that paired each score with itself) and the DS balanced
  accuracy of External-Avg vs Judge-Avg is 0.669 with the human side as
  reference (the submitted 0.740 averaged both directions). The external
  portability correlation is rho = 0.944 (n = 9 models with >= 5 trap items)
  now that all 191 items carry a model identity.

## [1.0.0-anonymous] — 2026-05-03

Initial submission bundle for NeurIPS 2026 Datasets & Benchmarks Track.

### Included
- 230 author-constructed scenarios across 9 domains and 4 taxonomy
  cells (F0=35, F1=65, F2=65, F3=65).
- 4,140 model response rows across 18 models at temperature 0.7
  (single-turn, no system prompt); 4,134 are non-null (6 failed
  generation and are retained as null rows — see Known Issues).
- 8,268 author annotations (A1, A2) on the 3-dimensional ordinal rubric
  (Goal Fidelity, Feasibility Acknowledgment, Displacement Severity).
- 191-item external rater subset (R1, R2) for rubric portability.
- Post-training stage probe (`probe1_post_training/`): 7 checkpoints
  spanning OLMo-2 / Qwen-2.5 / Mistral 7B base / SFT / DPO / instruct
  variants, dual-judged by GPT-4.1 and Kimi-K2.
- 2x2 discriminant probe (`discriminant_2x2/`): 8 models on a 40-scenario
  factorial of task complexity × social pressure.
- Six reproducibility scripts (`code/`) that recompute every headline
  number from the checked-in data.
- Datasheet (Gebru et al. 2021 schema), Croissant 1.0 metadata with
  Responsible AI fields, rubric anchors, and rater guidelines.

### Documentation
- README with intended uses, out-of-scope uses, limitations, known
  failure modes, bias and mitigation, compute disclosure, and author
  responsibility statement.
- DATASHEET with full motivation, composition, collection, processing,
  uses, distribution, and maintenance sections.

### Known issues
- 6 of 4,140 (scenario, model) pairs have no response (refusals or API
  errors), all in the F0 empathy cell for `gpt-4o` and `ollama:mistral`.
  These rows are present in `unified_dataset.csv` with NaN scores.
- Three frontier models (`deepseek-reasoner`, `claude-haiku-4.5`,
  `gpt-5.4-mini-2026-03-17`) lack public Chatbot Arena Elo at submission
  time; they are excluded from the Arena correlation analysis (n=15).
- The post-training probe omits a Mistral 7B v0.1 base checkpoint that
  was tried during pilot but excluded from the final analysis.
