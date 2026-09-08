# 🔭 Vantage Spec: Chimera String Synthesis

## 👤 User Story
**As a** Data Scientist building knowledge graphs,
**I want** to intelligently blend string properties when synthesizing two nodes into a chimera,
**so that** textual attributes like labels, descriptions, and categories are meaningfully combined rather than just discarding one of them.

## 🧐 The "So What?" Ask
**What business problem does this solve?**
Currently, the Chimera hybrid entity synthesis engine handles numeric properties but explicitly defaults to picking the first string for string properties. Real-world graphs rely heavily on textual and categorical data. Without string synthesis, chimeras lose critical context (e.g., merging "Dept A" and "Dept B" should yield a combined name, not just "Dept A"). Fixing this expands Chimera's utility from strictly numerical datasets to general-purpose business graphs.

**Success Metric Definition:**
- **Execution Time:** Blending string properties during synthesis adds < 2ms latency per property.
- **Completeness:** 100% of defined string merge strategies (e.g., Concatenate, PickLongest) are applied correctly when synthesizing two string properties.

## ✅ Acceptance Criteria
- Must introduce a string synthesis strategy configuration (e.g., `Concatenate`, `PickLongest`, `PickFirst`).
- Must integrate this configuration into `SynthesisConfig`.
- Must replace the current fallback logic (which currently picks the first string) with logic that respects the chosen string strategy.
- Must handle missing strings gracefully (if only one node has the property, use it; if neither, skip).

## 🚫 Out of Scope
- Semantic LLM-based string merging (e.g., using an LLM to summarize two descriptions). Phase 1 only uses deterministic string operations.
