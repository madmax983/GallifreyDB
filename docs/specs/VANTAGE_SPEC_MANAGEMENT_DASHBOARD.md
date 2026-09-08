# 🔭 Vantage: Spec for Management Dashboard

## 1. 👤 User Story

> **As a** Database Administrator,
> **I want to** visually inspect the current state of the database (nodes, edges, metrics),
> **So that** I can debug data issues and monitor performance without having to write Cypher queries or use command-line tools.

## 2. 🧐 The "So What?" (Business Value)

Currently, AletheiaDB lacks a built-in visual interface for querying and exploring data. Users have to rely on REST endpoints or the CLI to interact with the database.

**The Gap:**
- **Developer Experience (DX):** Inspecting graph structures purely through JSON responses is tedious and counter-intuitive.
- **Monitoring:** There is no easy way to view system metrics (memory usage, query latency, index sizes) in a single unified view.

**ROI:**
- **Productivity:** Developers can quickly browse their data, reducing debugging time.
- **Adoption:** A polished built-in dashboard lowers the barrier to entry and makes the product feel more mature and complete.

## 3. ✅ Acceptance Criteria

### Functional Requirements
1. **Node and Edge Browser:**
   - The dashboard must provide a tabular or graph-based view to browse nodes and their connected edges.
   - Users must be able to filter nodes by label and property values.
2. **Query Runner:**
   - The dashboard must include a text area to execute Cypher/AQL queries and display the results.
3. **Metrics Overview:**
   - The dashboard must display key database metrics (e.g., active connections, memory usage, transaction throughput).
4. **Architecture:**
   - The dashboard must be served natively from the AletheiaDB HTTP server using the `autumn-web` framework (with `maud` and `htmx`), accessible at `/admin`.

### Non-Functional Requirements
- **Performance:** Rendering the dashboard pages should add minimal overhead to the database server.
- **Zero Configuration:** The dashboard must be available out-of-the-box when the HTTP server is enabled, requiring no separate installation or complex build pipelines.

## 4. 🚫 Out of Scope
- **Advanced Graph Visualization:** Complex interactive network graphs (e.g., d3.js force-directed graphs) are deferred to a later phase. Phase 1 will focus on tabular data and basic inspection.
- **User Authentication:** As with the HTTP API, authentication is handled externally in Phase 1.
- **Modifying Data:** The Phase 1 dashboard is read-only. Editing nodes or edges via the UI is out of scope.
