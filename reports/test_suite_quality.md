# Test Suite Quality Review

**Date:** 2026-05-08
**Scope:** All 8 JSON files under `tests/test_data/` (88 test cases + 1 prompt-definition file).
**Focus:** Coverage & diversity, and test-case correctness.

This is a one-time qualitative review. Findings cite test `id` and source file so each issue can be opened and inspected directly.

---

## Executive summary

The suites are **functional and broadly schema-valid**, but they have meaningful quality gaps that will distort eval results today:

1. **Several voice-command tests have wrong or arbitrary expected outputs** — most concerning in `sork_voice_commands_suite.json` (e.g. `sork_032` expects `go_up` for "Pick up the welding torch"). These bake noise into pass/fail rates.
2. **A test in the Anki suite references a non-existent tool** (`guiDeckOverview` in `anki_precision_001`) and another (`anki_expert_001`) requires the model to predict an unknowable argument value — both are unfair tests that no model can pass legitimately.
3. **The `unit: "fahrenheit"` expectation in temperature tests will fail correct models** that omit a non-required field (`example_test_suite test_002`, `latency_benchmark_suite latency_baseline_003`). Per `defaults.py`, only `temperature` is required.
4. **Latency-suite metadata (`version`, `target_latency_ms`, `category`, `expected_token_count`, `system_prefix`, `field_mapping`, ...) is silently dropped** by `TestSuite`/`VoiceCommandTest` Pydantic models in `evaluator.py`. If this metadata matters at evaluation time, a separate loader is required.
5. **Tag taxonomy is inconsistent across suites** (`basic` vs none, `ambiguous` vs `ambiguity`, three different "complex" tags), so the cross-tag aggregation in `EvaluationReport.tag_performance` produces noisy comparisons.
6. **Tool coverage gaps**: 5 of 16 Sork tools and 1 of 13 Anki tools are never exercised. Argument-value coverage for the 5 default voice-command tools is shallow (most enums tested at one value).

The suites that are clean and ready to use as-is: `theory_of_mind_suite.json`, `letter_counting_suite.json`, `comprehensive_text_suite.json`, `example_prompts.json`.

---

## Cross-suite findings

### Schema validity
All files load cleanly under the Pydantic models in `src/sledge_eval/evaluator.py` (`TestSuite` for voice-command suites, `TextEvaluationSuite` for text suites, `PromptsFile` for `example_prompts.json`). However:

- **`latency_benchmark_suite.json` carries fields the schema does not declare** — top-level `version`, `target_latency_ms`, `metadata`, and per-test `category`, `expected_token_count`, `system_prefix`, `optimization_savings_tokens`, `field_mapping`, `speculative_notes`. Pydantic discards these silently. If `latency_evaluator.py` doesn't load via a separate model, this metadata is lost.
- **`evaluation_type` strings `"letter_count"` and `"theory_of_mind"`** appear in the text suites. The `TextEvaluationTest.evaluation_type` docstring lists only `contains`, `exact`, `custom`. Confirm `text_evaluator.py` handles the extended values (it appears to, based on suite design); consider updating the docstring/enum to make the contract explicit.

### ID collisions across suites
`letter_count_001`, `letter_count_002`, `letter_count_003`, `theory_of_mind_001`, `theory_of_mind_002`, `theory_of_mind_003` exist in both their dedicated suites and in `comprehensive_text_suite.json`. If both files are ever loaded into the same session, there is no per-suite namespace for IDs in `EvaluationReport`, so per-test cross-referencing breaks down.

### Tag taxonomy inconsistencies
`tag_performance` in `EvaluationReport` aggregates by literal tag string, so tag drift fragments the breakdown:

- **Difficulty tags vary by suite**: `example_test_suite` uses `basic`; `anki_large_toolset_suite` uses `basic`/`intermediate`/`advanced`/`expert`; `sork_voice_commands_suite` has none; `comprehensive_text_suite` mixes `basic`/`medium`/`advanced`.
- **"Complex" appears three ways**: `complex_parameters` (anki_advanced_001), `complex_workflow` (anki_expert_001), `complex` (latency_multi_001).
- **`ambiguous` vs `ambiguity`**: `anki_ambiguity_001` uses tag `ambiguous`; the test ID uses `ambiguity`. Singular use, can't aggregate.
- **`baseline` tag is overloaded in latency suite**: `latency_multi_001` is tagged both `baseline` and `multi_tool`, but its `category` field is `multi_tool` — the tag contradicts the category.

### Tool coverage gaps (relative to tool definitions)
- **Sork** (`src/sledge_eval/tools/sork_tools.py`, 16 tools): never exercised — `go_south`, `go_down`, `use`, `attack`, `flee`. 5/16 tools (31%) untested.
- **Anki** (`src/sledge_eval/tools/defaults.py` `ANKI_TOOLS`, 13 tools): never exercised — `rate_card`. Plus `anki_precision_001` calls a tool that doesn't exist (see below).
- **Default voice-command** (5 tools): all 5 used, but enum values are barely sampled — `control_lights.action` only tests `turn_on` (`turn_off`, `dim`, `brighten` untested), `adjust_volume.action` only tests `increase`, `set_temperature.unit` only tests `fahrenheit`, `get_weather.timeframe` only tests `today`.

---

## Per-suite review

### 1. `tests/test_data/example_test_suite.json` — 4 tests

**Purpose:** Smoke-test the 5 default smart-home tools.

**Coverage & diversity**
- All 5 default tools called at least once. Good baseline coverage.
- Enum coverage thin (see cross-suite section above).
- No prompt-phrasing variation (single phrasing per intent). Fine for a smoke suite, weak for robustness measurement.

**Correctness**
- **`test_002` "Set the thermostat to 72 degrees"** expects `unit: "fahrenheit"`, but the user did not specify a unit. Per `defaults.py`, `unit` is **not** in `required`. A model that correctly omits `unit` will fail under the flexible comparator (which requires every expected key to be present). Either drop `unit` from the expectation or change the prompt to "72 degrees fahrenheit".
- `test_001`, `test_003`, `test_004` are correct.

**Recommended fixes**
- Drop `unit` from `test_002` expected args (or rephrase prompt).
- Add 4–6 more tests to cover `turn_off`/`dim`, `decrease`/`mute`, `celsius`, and weather `location` + non-`today` timeframes.

---

### 2. `tests/test_data/sork_voice_commands_suite.json` — 43 tests

**Purpose:** Voice-to-tool conversion for a Zork-style game, generated from gameplay logs.

**Coverage & diversity**
- 11 of 16 tools used; **`go_south`, `go_down`, `use`, `attack`, `flee` never tested**. Combat and downward navigation are blind spots.
- Strong phrasing variation: 6 different phrasings of "go east" (`Move East.`, `MOVE EAST.`, `East.`, `Move eastward.`, etc.). Good for stress-testing.
- Multi-word item handling tested well: `emergency suit` vs `emergency space suit` vs `emergency spacesuit` (sork_002/003/006).
- No difficulty tags; tags are flat (`inspection`, `items`, `navigation`, `meta`, `edge_case`, `unrecognized`).

**Correctness — high-confidence bugs**
- **`sork_032` "Pick up the welding torch."** expected `go_up`. Clearly wrong — should be `take` with `item: "welding torch"`. Looks like a copy/paste error.
- **`sork_033` "Grab the welding torch."** expected `take` with `item: "welding"` (truncated). Should be `"welding torch"`.
- **`sork_034` "Drop the welding torch."** expected `drop` with `item: "welding"` (truncated). Should be `"welding torch"`.
- **`sork_005` "See you at emergency space soon."** expected `help`. There is no help-intent in this command (looks like ASR garbage); `[]` (unrecognized) is the only defensible expectation.
- **`sork_011` "Fully from the room."** expected `inventory`. Arbitrary — no inventory intent. Should be `[]`.
- **`sork_025` "MOVE EAST."** expected `[]` (unrecognized) while **`sork_028` "Move East."** expects `go_east`. The only difference is letter case. The Sork tool docstring lists `east` as a trigger and is case-agnostic. Same input semantically — pick one.
- **`sork_039` "room."** expected `examine` with `target: "room"`, but `sork_014` ("the room."), `sork_020` ("of the room."), `sork_037` ("The room.") all expect `look`. Inconsistent ground truth for near-identical inputs.
- **`sork_004` "Inventory."** and **`sork_010` "inventory."** expected `[]` (unrecognized), but `sork_018` ("What's in my inventory?") expects `inventory`. The bare word "inventory" is the canonical example in the tool docstring — it should fire `inventory`, not be unrecognized.

**Correctness — ambiguous (defensible expectation, but should be documented)**
- `sork_007` "There's the room." → `look`: defensible but borderline.
- `sork_022` "the pocket knife." → `take`: no verb in command; could be `examine`/`drop`. Pick a convention and apply it consistently.
- `sork_040` "The oxygen tank." → `take`: same issue.
- `sork_042` "around the room." → `[]`: arguable vs `look`.

**Recommended fixes**
- Fix the four high-confidence bugs (`sork_032`, `sork_033`, `sork_034`, plus `sork_005` or `sork_011`).
- Resolve the case-sensitivity inconsistency (`sork_025` vs `sork_028`).
- Pick a single convention for "bare item, no verb" (`sork_022`, `sork_040`) and "bare 'room'" (`sork_014`/`020`/`037`/`039`) — document it in suite description.
- Decide whether bare "inventory" should fire `inventory` or be unrecognized; align `sork_004`/`010` with `sork_018`.
- Add tests for `go_south`, `go_down`, `use`, `attack`, `flee`.
- Consider adding difficulty tags (`basic`/`ambiguous`/`edge_case`) so tag aggregation matches other suites.

---

### 3. `tests/test_data/anki_large_toolset_suite.json` — 12 tests

**Purpose:** Stress-test tool selection and parameter construction with a 13-tool MCP-style toolset.

**Coverage & diversity**
- 12 of 13 tools used; `rate_card` is never tested despite `anki_context_001` setting up the workflow that would need it.
- Good complexity gradient: basic listing → query-string construction → nested parameters → multi-step.
- Difficulty tagging is consistent inside this suite (`basic`/`intermediate`/`advanced`/`expert`/`edge_case`).

**Correctness — high-confidence bugs**
- **`anki_precision_001`** expects a call to **`guiDeckOverview`**, which is **not defined** in `ANKI_TOOLS` (`src/sledge_eval/tools/defaults.py`). No model can pass this test legitimately — it would have to hallucinate a tool name. Either add the tool to `ANKI_TOOLS` or change the expected call to a tool that exists.
- **`anki_expert_001`** expects `notesInfo` with `notes: []` (empty array). The `notesInfo` schema in `defaults.py` requires `minItems: 1`. The model has no way to know real note IDs without first seeing `findNotes` results. The expectation is unfair and contradicts the schema. Restructure as a single-call test (just `findNotes`), or accept any non-empty list, or evaluate the two calls in separate turns.
- **`anki_context_001`** expects `present_card` with `card_id: 0`. Same problem — the model can't know the ID. This test will only pass by accident.

**Correctness — schema-fragile**
- **`anki_advanced_001`** uses `deckName` / `modelName` / `fields` (camelCase). The comparator in `utils/comparison.py` lowercases keys recursively, so a model returning `deckname`/`modelname` would still pass — fine. But a model returning the OpenAI-conventional snake_case (`deck_name`) would fail. This is a real discrimination, but worth being aware of.
- **`anki_context_001`** expects `show_answer: false` — flexible comparator requires this key be present even though the schema says it defaults to `false`. A model that omits it (relying on the default) fails.

**Recommended fixes**
- Add `guiDeckOverview` to `ANKI_TOOLS` (and a `present_card`-style schema), or rewrite `anki_precision_001` to use an existing tool.
- Restructure `anki_expert_001` and `anki_context_001` so the model isn't asked to predict argument values that depend on prior tool output.
- Add at least one `rate_card` test (the only untested tool in the set).

---

### 4. `tests/test_data/latency_benchmark_suite.json` — 18 tests

**Purpose:** Benchmark inference latency under various optimization strategies (prefix injection, short field names, combined, multi-tool, speculative-friendly).

**Coverage & diversity**
- 6 categories well-represented: `baseline` (5), `prefix_injection` (3), `short_fields` (3), `combined_optimizations` (2), `multi_tool` (2), `speculative_friendly` (2). Reasonable spread.
- Same 4–5 intents (lights, weather, temperature, music, volume) recycled across optimization variants — appropriate for an A/B latency comparison.

**Correctness**
- **Schema-undeclared fields silently dropped**: `version`, `target_latency_ms`, `metadata`, `category`, `expected_token_count`, `system_prefix`, `optimization_savings_tokens`, `field_mapping`, `speculative_notes` are not in the Pydantic models (`TestSuite`, `VoiceCommandTest`). Verify `latency_evaluator.py` reads this file via a custom loader, not `Evaluator.load_test_suite()`. If it doesn't, every "optimized" test silently degrades to a baseline test.
- **Short-field tests use undefined tool names** (`lt`, `wx`, `tmp`). These do not exist in `DEFAULT_VOICE_COMMAND_TOOLS`. If the latency runner injects an alternate tool definition (using `field_mapping`), this is fine — but verify; otherwise the model is being asked to call hallucinated tools and "passing" only by happy coincidence.
- **`latency_baseline_003` "Set the thermostat to 72 degrees"** expects `unit: "fahrenheit"`. Same bug as `example_test_suite test_002` — `unit` is not required and not specified by the user.
- **`latency_multi_short_001` "Lights on, temp 70"** expects `lt` with `{"a": "on"}` (no room) and `tmp` with `{"t": 70}`. Defensible (no room specified) but worth noting that the corresponding baseline `latency_baseline_001` always uses `living room` as the room — silent inconsistency between baseline and short variants.
- **Tag drift inside the suite**: `latency_multi_001` is tagged `baseline` while categorized `multi_tool`; `latency_speculative_001`/`002` are missing the `optimized` tag that other optimization tests carry. Pick a tag scheme and apply it uniformly.

**Recommended fixes**
- Confirm `latency_evaluator.py` has a loader that preserves the extra fields. If it uses the generic loader, write a `LatencyTestSuite` Pydantic model that declares them.
- Drop `unit` from `latency_baseline_003` (or rephrase prompt).
- Document the convention for short-field tool definitions (and where they're injected) in the suite description.
- Normalize tags: every test should carry exactly its `category` value as a tag, plus a single domain tag (`lights`/`weather`/etc.).

---

### 5. `tests/test_data/comprehensive_text_suite.json` — 6 tests

**Purpose:** Combined letter-counting + theory-of-mind text-evaluation suite.

(Note: the file contains **6 tests, not 7** as listed in `CLAUDE.md`. Update the entry-points doc.)

**Coverage & diversity**
- 3 letter-counting + 3 theory-of-mind. Balanced, but tiny absolute count.
- Tags consistent within each evaluation type. `evaluation_type` field used appropriately.

**Correctness**
- All 3 letter-count answers verified manually:
  - "strawberry" has 3 r's. ✓
  - "parallel" has 3 l's. ✓
  - "development" has 3 e's. ✓
- All 3 theory-of-mind answers correct (classic false-belief structure).
- `evaluation_type: "letter_count"` and `"theory_of_mind"` are not in the docstring enum on `TextEvaluationTest.evaluation_type` — verify they map to working evaluators.

**Recommended fixes**
- Update `CLAUDE.md` test count from 7 to 6.
- Update `TextEvaluationTest.evaluation_type` docstring/enum to include `letter_count` and `theory_of_mind`.
- Decide whether the standalone suites (`theory_of_mind_suite.json`, `letter_counting_suite.json`) are kept; if so, keep IDs the same and treat this file as a convenience superset (current behavior). Otherwise delete duplicates.

---

### 6. `tests/test_data/theory_of_mind_suite.json` — 3 tests

**Purpose:** Standalone false-belief reasoning suite (Sally-Anne variations).

**Coverage & diversity**
- All 3 tests are classic false-belief structure with rotated actors/objects/locations. Good for what it tests.

**Correctness**
- All 3 expected answers are correct.

**Duplication**
- 100% overlap with the theory-of-mind portion of `comprehensive_text_suite.json` (same IDs, same content). Either delete this file or delete the duplicates from `comprehensive_text_suite.json`.

**Recommended fixes**
- Pick one canonical home for these tests.
- If kept, expand to second-order false belief (e.g. "What does Anne think Sally thinks?") to differentiate it from the comprehensive suite.

---

### 7. `tests/test_data/letter_counting_suite.json` — 2 tests

**Purpose:** Standalone letter-counting suite.

**Coverage & diversity**
- 2 tests is too few to draw conclusions. Both are 3-of-letter answers, so the model could pass by always answering "3".

**Correctness**
- Both expected answers verified correct.

**Duplication**
- 100% overlap with the first 2 letter-counting tests in `comprehensive_text_suite.json`.

**Recommended fixes**
- Resolve duplication (same as ToM suite).
- Diversify expected answers: add tests that should answer 0, 1, 2, 4+. Currently a model answering "3" passes both tests trivially.

---

### 8. `tests/test_data/example_prompts.json` — 3 prompt definitions (not a test suite)

**Purpose:** System-prompt comparison input for `PromptComparisonReport`.

**Findings**
- Schema-valid against `PromptsFile`.
- Three reasonable prompt variants (baseline/detailed/concise). The "detailed" prompt is smart-home-specific, while the "baseline" and "concise" are generic — biases comparisons against the smart-home suite.

**Recommended fixes**
- Either generalize the "detailed" prompt or add a parallel set of prompts per domain (smart home, Sork, Anki).

---

## Recommendations (prioritized)

### P0 — Bugs that produce wrong eval signal (fix before next benchmark run)
1. `sork_032` — change expected from `go_up` to `take {item: "welding torch"}`.
2. `sork_033`, `sork_034` — fix `item` from `"welding"` to `"welding torch"`.
3. `anki_precision_001` — `guiDeckOverview` does not exist in `ANKI_TOOLS`. Add the tool definition, or change the expected call.
4. `anki_expert_001`, `anki_context_001` — drop expectations on argument values the model cannot know (`notes: []`, `card_id: 0`).
5. `example_test_suite test_002` and `latency_baseline_003` — drop `unit: "fahrenheit"` from expected args, or change the voice command to specify the unit.
6. `sork_025` vs `sork_028` — resolve the case-sensitivity contradiction.
7. `sork_004`/`sork_010` vs `sork_018` — resolve the bare-"inventory" contradiction.

### P1 — Quality cleanup
8. `sork_005` "See you at emergency space soon." → `help` and `sork_011` "Fully from the room." → `inventory`: arbitrary mappings, change to `[]`.
9. Adopt one convention for bare-item ("the pocket knife.") and bare-"room" inputs in the Sork suite; apply consistently.
10. Add Sork tests covering `go_south`, `go_down`, `use`, `attack`, `flee`.
11. Add `rate_card` test in the Anki suite.
12. Verify `latency_evaluator.py` preserves `category`, `expected_token_count`, `system_prefix`, `field_mapping` etc.; if not, define a `LatencyTestSuite` Pydantic model.
13. Resolve duplication between `comprehensive_text_suite.json` and the standalone text suites.

### P2 — Coverage expansion
14. Expand `example_test_suite.json` enum coverage for the 5 default tools.
15. Diversify `letter_counting_suite.json` expected answers beyond "3".
16. Add second-order false-belief tests to `theory_of_mind_suite.json` (or a new file).
17. Align tag taxonomy across suites: agree on a difficulty axis (`basic`/`intermediate`/`advanced`/`edge_case`) and apply it uniformly so `tag_performance` aggregations are meaningful.
18. Update `CLAUDE.md` test count for `comprehensive_text_suite.json` from 7 to 6.
19. Update `TextEvaluationTest.evaluation_type` docstring to include `letter_count` and `theory_of_mind`.
