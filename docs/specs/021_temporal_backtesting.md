# 🔭 Vantage: Spec for Temporal Backtesting

👤 **User Story:**
As a Trader, I want to backtest against volatile markets using historical temporal snapshots of the knowledge graph, so that I can accurately simulate trading strategies on the exact graph state at any given point in the past without data leakage from the future.

🧐 **The "So What?" (Business Value):**
Current backtesting solutions either rely on static historical snapshots or require complex logic to piece together the state of a market or portfolio at a specific time. AletheiaDB's bi-temporal features inherently support "as-of" queries, but traders need an explicit way to run simulations that traverse time programmatically without manually issuing thousands of discrete `AS OF` queries.
This solves the problem of "look-ahead bias" in quantitative models by guaranteeing the state of the graph used for evaluation was exactly what was known at that moment.

**Success Metric Definition:**
- **Accuracy:** Zero future-data leakage in historical simulations.
- **Performance:** Running a 1-year daily simulation (approx. 250 trading days) with complex graph traversal per day completes in under 5 seconds.

✅ **Acceptance Criteria:**
- Must provide a programmatic API (e.g., `db.backtest(strategy, start_time, end_time, interval)`) to run simulations over a specific time range.
- Must guarantee that for each simulation tick, the graph state exposed to the `strategy` strictly represents the system time as of that tick.
- Must handle NaN data without panicking.
- Must output a CSV report containing the results of the backtest.

🚫 **Out of Scope (Phase 1):**
- Real-time execution (Phase 2).
