//! Predicate Pushdown Optimization
//!
//! Moves filter operations as close to data sources as possible,
//! reducing the number of rows processed by expensive operations
//! like traversals and joins.
//!
//! # Theory: Filter Early
//!
//! The "Filter Early" principle is fundamental to query optimization. By applying
//! predicates (filters) as early as possible in the query pipeline, we reduce the
//! cardinality (number of rows) that subsequent operators must process.
//!
//! For example, if a `VectorRank` operation is O(N log k) (with limit k) and a filter
//! removes 90% of the rows, pushing the filter before the rank significantly reduces
//! the workload.
//!
//! # Optimization Strategy
//!
//! This rule recursively traverses the logical plan and attempts to "bubble down"
//! `Filter` operators through other unary operators where safe.
//!
//! ## Capabilities
//!
//! Currently, this rule supports pushing filters through:
//! - **VectorRank (without limit)**: Safe because ranking all items then filtering produces the same result as filtering then ranking.
//! - **Sort**: Sorting is commutative with filtering.
//!
//! ## Limitations
//!
//! - **VectorRank (with limit)**: Cannot push filter past a Top-K operation. Filtering *after* finding top-K is semantically different from finding top-K *after* filtering.
//! - **Traversals**: We conservatively do *not* push filters through traversals yet.
//! - **Scans**: Filters cannot be pushed below scans (they are the leaves).
//! - **Joins**: Filter pushdown through joins is handled by separate logic (future work).
//!
//! # Safety
//!
//! Pushdown is safe when:
//! 1. **No Side Effects**: The operator being swapped doesn't produce side effects that the filter depends on.
//! 2. **Semantic Equivalence**: The result set remains identical.
//!    - **CRITICAL**: Filters must NOT be pushed past `Limit` or `Top-K` operations, as this changes the result set.

use crate::core::error::Result;
use crate::query::plan::{LogicalOp, LogicalPlan, UnaryOp};

use super::{OptimizationRule, Statistics};

/// Predicate pushdown optimization rule.
///
/// This rule moves Filter operations below Traverse and other operations
/// when possible, reducing intermediate result sizes.
///
/// # Example Transformation
///
/// **Before**: Filter is applied *after* sorting (expensive).
/// ```text
/// Filter(name = "Alice")
///   Sort(score DESC)
///     VectorSearch(...)
/// ```
///
/// **After**: Filter is applied *before* sorting (cheaper).
/// ```text
/// Sort(score DESC)
///   Filter(name = "Alice")
///     VectorSearch(...)
/// ```
///
/// # Complex Example
///
/// Pushing through multiple layers:
///
/// ```text
/// Filter(active = true)
///   Sort(created DESC)
///     VectorRank(top_k=None)
///       Scan(...)
/// ```
///
/// Becomes:
///
/// ```text
/// Sort(created DESC)
///   VectorRank(top_k=None)
///     Filter(active = true)  <-- Pushed down
///       Scan(...)
/// ```
///
/// ## Examples
///
/// ```rust
/// use aletheiadb::query::planner::rules::{OptimizationRule, PredicatePushdown};
/// use aletheiadb::query::planner::stats::Statistics;
/// use aletheiadb::query::plan::{LogicalPlan, LogicalOp, UnaryOp, ScanOp, SortKey};
/// use aletheiadb::query::ir::{Predicate, PredicateValue};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // 1. Construct a sub-optimal plan: Filter is applied AFTER sorting
/// let scan = LogicalOp::Scan(ScanOp::NodeScan {
///     label: Some("Person".into()),
///     estimated_rows: Some(100),
/// });
/// let sort = LogicalOp::unary(
///     UnaryOp::Sort { key: SortKey::Score, descending: true },
///     scan
/// );
/// let filter = LogicalOp::unary(
///     UnaryOp::Filter(Predicate::Eq { key: "active".into(), value: PredicateValue::Bool(true) }),
///     sort
/// );
///
/// let plan = LogicalPlan::new(filter);
///
/// // 2. Apply the rule
/// let rule = PredicatePushdown;
/// let stats = Statistics::new();
/// let optimized_plan = rule.apply(&plan, &stats)?.unwrap();
///
/// // 3. The rule pushed the Filter BELOW the Sort
/// let expected_plan = LogicalPlan::new(
///     LogicalOp::unary(
///         UnaryOp::Sort { key: SortKey::Score, descending: true },
///         LogicalOp::unary(
///             UnaryOp::Filter(Predicate::Eq { key: "active".into(), value: PredicateValue::Bool(true) }),
///             LogicalOp::Scan(ScanOp::NodeScan { label: Some("Person".into()), estimated_rows: Some(100) })
///         )
///     )
/// );
/// assert_eq!(optimized_plan, expected_plan);
/// # Ok(())
/// # }
/// ```
pub struct PredicatePushdown;

impl OptimizationRule for PredicatePushdown {
    fn name(&self) -> &str {
        "predicate-pushdown"
    }

    fn apply(&self, plan: &LogicalPlan, _stats: &Statistics) -> Result<Option<LogicalPlan>> {
        let (new_root, changed) = self.push_down(&plan.root)?;

        if changed {
            Ok(Some(LogicalPlan {
                root: new_root,
                temporal_context: plan.temporal_context.clone(),
                hints: plan.hints.clone(),
            }))
        } else {
            Ok(None)
        }
    }
}

impl PredicatePushdown {
    /// Recursively push down filters where possible.
    ///
    /// This method traverses the plan tree. When it encounters a `Filter` operator,
    /// it attempts to move it below its input operator if that operator type allows it.
    fn push_down(&self, op: &LogicalOp) -> Result<(LogicalOp, bool)> {
        match op {
            // Filter operation: this is what we want to push down
            LogicalOp::Unary {
                op: UnaryOp::Filter(predicate),
                input,
            } => {
                // First, recursively optimize the input (bottom-up approach)
                let (optimized_input, input_changed) = self.push_down(input)?;

                // Check if we can push the filter below the optimized input operator
                match &optimized_input {
                    // STOP: Can't push below scans (they are the source)
                    LogicalOp::Scan(_) => Ok((
                        LogicalOp::unary(UnaryOp::Filter(predicate.clone()), optimized_input),
                        input_changed,
                    )),

                    // STOP: For traversal, we generally can't push blindly.
                    // We need to know if the predicate applies to the source or target.
                    // Current implementation is conservative and stops here.
                    LogicalOp::Unary {
                        op: UnaryOp::Traverse { .. },
                        ..
                    } => Ok((
                        LogicalOp::unary(UnaryOp::Filter(predicate.clone()), optimized_input),
                        input_changed,
                    )),

                    // PUSH: VectorRank
                    // Only safe if top_k is None (pure re-scoring).
                    // If top_k is set, pushing filter changes semantics (Top-K then Filter != Filter then Top-K).
                    LogicalOp::Unary {
                        op:
                            UnaryOp::VectorRank {
                                embedding,
                                top_k,
                                property_key,
                                metric,
                                threshold,
                                score_alias,
                            },
                        input: vector_input,
                    } => {
                        if top_k.is_some() {
                            // STOP: Has limit, unsafe to push down
                            Ok((
                                LogicalOp::unary(
                                    UnaryOp::Filter(predicate.clone()),
                                    optimized_input,
                                ),
                                input_changed,
                            ))
                        } else {
                            // SAFE: No limit, just re-scoring
                            let filter_then_rank = LogicalOp::unary(
                                UnaryOp::VectorRank {
                                    embedding: embedding.clone(),
                                    top_k: *top_k,
                                    property_key: property_key.clone(),
                                    metric: *metric,
                                    threshold: *threshold,
                                    score_alias: score_alias.clone(),
                                },
                                LogicalOp::unary(
                                    UnaryOp::Filter(predicate.clone()),
                                    (**vector_input).clone(),
                                ),
                            );
                            Ok((filter_then_rank, true))
                        }
                    }

                    // PUSH: Sort
                    // Filter(Sort(Input)) -> Sort(Filter(Input))
                    // Safe because sorting is purely a reordering operation.
                    LogicalOp::Unary {
                        op: UnaryOp::Sort { key, descending },
                        input: sort_input,
                    } => {
                        let filter_then_sort = LogicalOp::unary(
                            UnaryOp::Sort {
                                key: key.clone(),
                                descending: *descending,
                            },
                            LogicalOp::unary(
                                UnaryOp::Filter(predicate.clone()),
                                (**sort_input).clone(),
                            ),
                        );
                        Ok((filter_then_sort, true))
                    }

                    // Default: STOP. Keep filter where it is.
                    _ => Ok((
                        LogicalOp::unary(UnaryOp::Filter(predicate.clone()), optimized_input),
                        input_changed,
                    )),
                }
            }

            // Not a filter: just recurse down (pass-through)
            LogicalOp::Unary { op, input } => {
                let (optimized_input, changed) = self.push_down(input)?;
                Ok((LogicalOp::unary(op.clone(), optimized_input), changed))
            }

            // Binary op: recurse down both branches
            LogicalOp::Binary { op, left, right } => {
                let (opt_left, left_changed) = self.push_down(left)?;
                let (opt_right, right_changed) = self.push_down(right)?;
                Ok((
                    LogicalOp::binary(op.clone(), opt_left, opt_right),
                    left_changed || right_changed,
                ))
            }

            // Leaf nodes: no change possible
            LogicalOp::Scan(_) | LogicalOp::Empty => Ok((op.clone(), false)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::NodeId;
    use crate::query::ir::Predicate;
    use crate::query::plan::{ScanOp, SortKey};
    use std::sync::Arc;

    fn test_stats() -> Statistics {
        Statistics::default()
    }

    #[test]
    fn test_no_change_on_simple_filter() {
        let rule = PredicatePushdown;
        let stats = test_stats();

        let plan = LogicalPlan::new(LogicalOp::unary(
            UnaryOp::Filter(Predicate::eq("name", "Alice")),
            LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
        ));

        let result = rule.apply(&plan, &stats).unwrap();
        assert!(result.is_none()); // No change needed
    }

    #[test]
    fn test_push_filter_below_vector_rank_no_limit() {
        let rule = PredicatePushdown;
        let stats = test_stats();

        // Filter(VectorRank(top_k=None, Scan)) -> VectorRank(Filter(Scan))
        // Should push down because no limit
        let plan = LogicalPlan::new(LogicalOp::unary(
            UnaryOp::Filter(Predicate::eq("name", "Alice")),
            LogicalOp::unary(
                UnaryOp::VectorRank {
                    embedding: Arc::from([0.1f32; 4].as_slice()),
                    top_k: None,
                    property_key: None,
                    metric: crate::core::vector::DistanceMetric::Cosine,
                    threshold: None,
                    score_alias: None,
                },
                LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
            ),
        ));

        let result = rule.apply(&plan, &stats).unwrap();

        let expected_plan = LogicalPlan::new(LogicalOp::unary(
            UnaryOp::VectorRank {
                embedding: Arc::from([0.1f32; 4].as_slice()),
                top_k: None,
                property_key: None,
                metric: crate::core::vector::DistanceMetric::Cosine,
                threshold: None,
                score_alias: None,
            },
            LogicalOp::unary(
                UnaryOp::Filter(Predicate::eq("name", "Alice")),
                LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
            ),
        ));

        assert_eq!(result, Some(expected_plan));
    }

    #[test]
    fn test_stop_filter_at_traverse() {
        let rule = PredicatePushdown;
        let stats = test_stats();

        // Filter(Traverse(Scan))
        // Should NOT push down because we stop at traversals
        let plan = LogicalPlan::new(LogicalOp::unary(
            UnaryOp::Filter(Predicate::eq("name", "Alice")),
            LogicalOp::unary(
                UnaryOp::Traverse {
                    label: None,
                    direction: crate::query::ir::Direction::Outgoing,
                    depth: crate::query::ir::TraversalDepth::Exact(1),
                },
                LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
            ),
        ));

        let result = rule.apply(&plan, &stats).unwrap();
        // Should return None because pushdown was blocked
        assert!(result.is_none());
    }

    #[test]
    fn test_stop_filter_at_scan() {
        let rule = PredicatePushdown;
        let stats = test_stats();

        // Filter(Scan)
        // Should NOT push down because we stop at scans
        let plan = LogicalPlan::new(LogicalOp::unary(
            UnaryOp::Filter(Predicate::eq("name", "Alice")),
            LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
        ));

        let result = rule.apply(&plan, &stats).unwrap();
        // Should return None because pushdown was blocked by Scan
        assert!(result.is_none());
    }

    #[test]
    fn test_stop_filter_at_vector_rank_with_limit() {
        let rule = PredicatePushdown;
        let stats = test_stats();

        // Filter(VectorRank(top_k=Some(10), Scan))
        // Should NOT push down because limit exists
        let plan = LogicalPlan::new(LogicalOp::unary(
            UnaryOp::Filter(Predicate::eq("name", "Alice")),
            LogicalOp::unary(
                UnaryOp::VectorRank {
                    embedding: Arc::from([0.1f32; 4].as_slice()),
                    top_k: Some(10),
                    property_key: None,
                    metric: crate::core::vector::DistanceMetric::Cosine,
                    threshold: None,
                    score_alias: None,
                },
                LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
            ),
        ));

        let result = rule.apply(&plan, &stats).unwrap();
        // Should return None because pushdown was blocked
        assert!(result.is_none());
    }

    #[test]
    fn test_push_filter_below_sort() {
        let rule = PredicatePushdown;
        let stats = test_stats();

        // Filter(Sort(Scan)) -> Sort(Filter(Scan))
        let plan = LogicalPlan::new(LogicalOp::unary(
            UnaryOp::Filter(Predicate::eq("active", true)),
            LogicalOp::unary(
                UnaryOp::Sort {
                    key: SortKey::Property("created".to_string()),
                    descending: true,
                },
                LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
            ),
        ));

        let result = rule.apply(&plan, &stats).unwrap();

        let expected_plan = LogicalPlan::new(LogicalOp::unary(
            UnaryOp::Sort {
                key: SortKey::Property("created".to_string()),
                descending: true,
            },
            LogicalOp::unary(
                UnaryOp::Filter(Predicate::eq("active", true)),
                LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
            ),
        ));

        assert_eq!(result, Some(expected_plan));
    }

    #[test]
    fn test_multi_level_pushdown() {
        let rule = PredicatePushdown;
        let stats = test_stats();

        // Filter -> Sort -> VectorRank(no limit) -> Scan
        // Should become: Sort -> VectorRank -> Filter -> Scan
        let plan = LogicalPlan::new(LogicalOp::unary(
            UnaryOp::Filter(Predicate::eq("active", true)),
            LogicalOp::unary(
                UnaryOp::Sort {
                    key: SortKey::Property("created".to_string()),
                    descending: true,
                },
                LogicalOp::unary(
                    UnaryOp::VectorRank {
                        embedding: Arc::from([0.1f32; 4].as_slice()),
                        top_k: None,
                        property_key: None,
                        metric: crate::core::vector::DistanceMetric::Cosine,
                        threshold: None,
                        score_alias: None,
                    },
                    LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
                ),
            ),
        ));

        // Simulate planner loop to get full pushdown
        let mut current_plan = plan;
        let mut changed = true;
        let mut iterations = 0;

        while changed && iterations < 10 {
            let result = rule.apply(&current_plan, &stats).unwrap();
            if let Some(new_plan) = result {
                current_plan = new_plan;
                changed = true;
            } else {
                changed = false;
            }
            iterations += 1;
        }

        // Now verify full pushdown: Sort -> VectorRank -> Filter -> Scan
        let expected_plan = LogicalPlan::new(LogicalOp::unary(
            UnaryOp::Sort {
                key: SortKey::Property("created".to_string()),
                descending: true,
            },
            LogicalOp::unary(
                UnaryOp::VectorRank {
                    embedding: Arc::from([0.1f32; 4].as_slice()),
                    top_k: None,
                    property_key: None,
                    metric: crate::core::vector::DistanceMetric::Cosine,
                    threshold: None,
                    score_alias: None,
                },
                LogicalOp::unary(
                    UnaryOp::Filter(Predicate::eq("active", true)),
                    LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
                ),
            ),
        ));

        assert_eq!(current_plan, expected_plan);
    }

    #[test]
    fn test_binary_op_recursion_logic() {
        use crate::query::plan::BinaryOp;

        let rule = PredicatePushdown;
        let stats = test_stats();

        // Binary(Union, Filter(Sort(Scan)), Scan)
        // Left side: Filter(Sort(Scan)) -> Sort(Filter(Scan)) (Optimized, changed=true)
        // Right side: Scan -> Scan (No change, changed=false)
        // Total change: true || false = true

        let left_op = LogicalOp::unary(
            UnaryOp::Filter(Predicate::eq("active", true)),
            LogicalOp::unary(
                UnaryOp::Sort {
                    key: SortKey::Property("created".to_string()),
                    descending: true,
                },
                LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
            ),
        );

        let right_op = LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(2).unwrap()]));

        let plan = LogicalPlan::new(LogicalOp::binary(BinaryOp::Union, left_op, right_op));

        // Expect optimization to occur on the left branch
        let result = rule.apply(&plan, &stats).unwrap();

        let expected_plan = LogicalPlan::new(LogicalOp::binary(
            BinaryOp::Union,
            LogicalOp::unary(
                UnaryOp::Sort {
                    key: SortKey::Property("created".to_string()),
                    descending: true,
                },
                LogicalOp::unary(
                    UnaryOp::Filter(Predicate::eq("active", true)),
                    LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
                ),
            ),
            LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(2).unwrap()])),
        ));

        assert_eq!(
            result,
            Some(expected_plan),
            "Binary op with one changed branch should return Some"
        );
    }
}

#[cfg(test)]
mod sentry_tests {
    use super::*;

    #[test]
    fn test_apply_unchanged() {
        let rule = PredicatePushdown;
        let stats = Statistics::default();
        let plan = LogicalPlan::new(LogicalOp::Scan(ScanOp::NodeLookup(vec![
            NodeId::new(1).unwrap(),
        ])));
        let result = rule.apply(&plan, &stats).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_apply_changed() {
        let rule = PredicatePushdown;
        let stats = Statistics::default();
        let plan = LogicalPlan::new(LogicalOp::unary(
            UnaryOp::Filter(Predicate::eq("a", 1)),
            LogicalOp::unary(
                UnaryOp::Sort {
                    key: SortKey::Property("a".to_string()),
                    descending: true,
                },
                LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
            ),
        ));
        let result = rule.apply(&plan, &stats).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_pushdown_traverse() {
        let rule = PredicatePushdown;
        let stats = Statistics::default();

        let plan = LogicalPlan::new(LogicalOp::unary(
            UnaryOp::Filter(Predicate::eq("a", 1)),
            LogicalOp::unary(
                UnaryOp::Traverse {
                    label: None,
                    direction: crate::query::ir::Direction::Outgoing,
                    depth: crate::query::ir::TraversalDepth::Exact(1),
                },
                LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
            ),
        ));

        let result = rule.apply(&plan, &stats).unwrap();
        // Filter cannot push past Traverse
        assert!(result.is_none());
    }
    #[test]
    fn test_pushdown_unsupported_unary_op() {
        let rule = PredicatePushdown;
        let stats = Statistics::default();

        // Filter inside an unsupported UnaryOp (like Limit) should be passed through
        // but limit itself cannot be pushed down into.
        let plan = LogicalPlan::new(LogicalOp::unary(
            UnaryOp::Limit(10),
            LogicalOp::unary(
                UnaryOp::Filter(Predicate::eq("a", 1)),
                LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
            ),
        ));

        let result = rule.apply(&plan, &stats).unwrap();
        // Since Filter is directly above Scan, it doesn't change.
        // Limit -> Filter -> Scan
        // push_down(Limit(Filter(Scan))) -> Limit(push_down(Filter(Scan))) -> Limit(Filter(Scan))
        assert!(result.is_none());

        let plan2 = LogicalPlan::new(LogicalOp::unary(
            UnaryOp::Filter(Predicate::eq("a", 1)),
            LogicalOp::unary(
                UnaryOp::Limit(10),
                LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
            ),
        ));

        let result2 = rule.apply(&plan2, &stats).unwrap();
        // Filter cannot push past Limit
        assert!(result2.is_none());
    }
    #[test]
    fn test_pushdown_name() {
        let rule = PredicatePushdown;
        assert_eq!(rule.name(), "predicate-pushdown");
    }

    use crate::core::NodeId;
    use crate::query::ir::Predicate;
    use crate::query::plan::{BinaryOp, ScanOp, SortKey};

    #[test]
    fn test_binary_op_partial_optimization() {
        // 🎯 Target: LogicalOp::Binary optimization logic (|| vs &&)
        // 💣 Risk: If changed logic uses &&, partial optimizations (one branch changed) would be lost.

        let rule = PredicatePushdown;
        let stats = Statistics::default();

        // Left: Filter(Sort(Scan)) -> Should change to Sort(Filter(Scan))
        let left = LogicalOp::unary(
            UnaryOp::Filter(Predicate::eq("a", 1)),
            LogicalOp::unary(
                UnaryOp::Sort {
                    key: SortKey::Property("a".to_string()),
                    descending: true,
                },
                LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
            ),
        );

        // Right: Filter(Scan) -> Should NOT change
        let right = LogicalOp::unary(
            UnaryOp::Filter(Predicate::eq("b", 2)),
            LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(2).unwrap()])),
        );

        // Binary Op: Union(Left, Right)
        let root = LogicalOp::binary(BinaryOp::Union, left, right);

        let plan = LogicalPlan::new(root);

        let result = rule.apply(&plan, &stats).unwrap();

        let expected_plan = LogicalPlan::new(LogicalOp::binary(
            BinaryOp::Union,
            LogicalOp::unary(
                UnaryOp::Sort {
                    key: SortKey::Property("a".to_string()),
                    descending: true,
                },
                LogicalOp::unary(
                    UnaryOp::Filter(Predicate::eq("a", 1)),
                    LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
                ),
            ),
            LogicalOp::unary(
                UnaryOp::Filter(Predicate::eq("b", 2)),
                LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(2).unwrap()])),
            ),
        ));

        assert_eq!(
            result,
            Some(expected_plan),
            "Partial optimization (left branch) should trigger change"
        );
    }

    #[test]
    fn test_binary_op_partial_change_right_side() {
        let rule = PredicatePushdown;
        let stats = Statistics::default();

        // Left: Filter(Scan) -> Should NOT change
        let left = LogicalOp::unary(
            UnaryOp::Filter(Predicate::eq("a", 1)),
            LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
        );

        // Right: Filter(Sort(Scan)) -> Should change to Sort(Filter(Scan))
        let right = LogicalOp::unary(
            UnaryOp::Filter(Predicate::eq("b", 2)),
            LogicalOp::unary(
                UnaryOp::Sort {
                    key: SortKey::Property("b".to_string()),
                    descending: false,
                },
                LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(2).unwrap()])),
            ),
        );

        let plan = LogicalPlan::new(LogicalOp::binary(BinaryOp::Union, left, right));

        let result = rule.apply(&plan, &stats).unwrap();

        let expected_plan = LogicalPlan::new(LogicalOp::binary(
            BinaryOp::Union,
            LogicalOp::unary(
                UnaryOp::Filter(Predicate::eq("a", 1)),
                LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
            ),
            LogicalOp::unary(
                UnaryOp::Sort {
                    key: SortKey::Property("b".to_string()),
                    descending: false,
                },
                LogicalOp::unary(
                    UnaryOp::Filter(Predicate::eq("b", 2)),
                    LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(2).unwrap()])),
                ),
            ),
        ));

        assert_eq!(
            result,
            Some(expected_plan),
            "Partial optimization (right branch) should trigger change"
        );
    }
}
