//! Limit Pushdown Optimization
//!
//! Propagates LIMIT operations down through the plan tree where safe,
//! enabling early termination and reducing work.

use crate::core::error::Result;
use crate::query::plan::{LogicalOp, LogicalPlan, UnaryOp};

use super::{OptimizationRule, Statistics};

/// Limit pushdown optimization rule.
///
/// This rule propagates LIMIT operations through certain operators
/// to enable early termination. This is particularly useful for
/// vector search and top-k queries.
///
/// # Example Transformation
///
/// Before:
/// ```text
/// Limit(10)
///   Sort(score DESC)
///     Traverse(KNOWS)
///       NodeLookup([1])
/// ```
///
/// After:
/// ```text
/// Limit(10)
///   Sort(score DESC, limit: 10)  // Top-K sort
///     Traverse(KNOWS)
///       NodeLookup([1])
/// ```
///
/// ## Examples
///
/// ```rust
/// use aletheiadb::query::planner::rules::{OptimizationRule, LimitPushdown};
/// use aletheiadb::query::planner::stats::Statistics;
/// use aletheiadb::query::plan::{LogicalPlan, LogicalOp, UnaryOp, ScanOp, SortKey};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // 1. Construct a plan where LIMIT is separated from VectorRank
/// let scan = LogicalOp::Scan(ScanOp::NodeScan {
///     label: Some("Person".into()),
///     estimated_rows: Some(100),
/// });
/// let rank = LogicalOp::unary(
///     UnaryOp::VectorRank {
///         embedding: vec![0.1, 0.2].into(),
///         top_k: None, // Missing limit!
///         property_key: None,
///         metric: aletheiadb::core::vector::DistanceMetric::Cosine,
///         threshold: None,
///         score_alias: None,
///     },
///     scan
/// );
/// let limit = LogicalOp::unary(UnaryOp::Limit(10), rank);
///
/// let plan = LogicalPlan { root: limit, temporal_context: None, hints: Default::default() };
///
/// // 2. Apply the rule
/// let rule = LimitPushdown;
/// let stats = Statistics::new();
/// let optimized_plan = rule.apply(&plan, &stats)?.unwrap(); // unwraps if `changed == true`
///
/// // 3. The rule preserves the outer Limit but ALSO pushes it down to VectorRank
/// if let LogicalOp::Unary { op: UnaryOp::Limit(10), input } = optimized_plan.root {
///     assert!(matches!(
///         *input,
///         LogicalOp::Unary { op: UnaryOp::VectorRank { top_k: Some(10), .. }, .. }
///     ));
/// } else {
///     panic!("Expected Limit operator at root");
/// }
/// # Ok(())
/// # }
/// ```
pub struct LimitPushdown;

impl OptimizationRule for LimitPushdown {
    fn name(&self) -> &str {
        "limit-pushdown"
    }

    fn apply(&self, plan: &LogicalPlan, _stats: &Statistics) -> Result<Option<LogicalPlan>> {
        let (new_root, changed) = self.push_down(&plan.root, None)?;

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

impl LimitPushdown {
    /// Recursively push down limit information where possible.
    ///
    /// `limit` is Some(n) if there's a LIMIT above that we might propagate.
    fn push_down(&self, op: &LogicalOp, limit: Option<usize>) -> Result<(LogicalOp, bool)> {
        match op {
            // Limit operation: propagate limit value down
            LogicalOp::Unary {
                op: UnaryOp::Limit(n),
                input,
            } => {
                // Use the smaller of the two limits
                let effective_limit = limit.map(|l| l.min(*n)).unwrap_or(*n);
                let (optimized_input, changed) = self.push_down(input, Some(effective_limit))?;

                // If the child is also a Limit, we can combine them
                if let LogicalOp::Unary {
                    op: UnaryOp::Limit(child_limit),
                    input: child_input,
                } = &optimized_input
                {
                    let combined_limit = effective_limit.min(*child_limit);
                    return Ok((
                        LogicalOp::unary(UnaryOp::Limit(combined_limit), (**child_input).clone()),
                        true,
                    ));
                }

                Ok((
                    LogicalOp::unary(UnaryOp::Limit(effective_limit), optimized_input),
                    changed || effective_limit != *n,
                ))
            }

            // VectorRank: can use limit hint for top-k optimization
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
                input,
            } => {
                let (optimized_input, input_changed) = self.push_down(input, None)?;

                // If there's a limit and it's smaller than current top_k, use it
                let new_top_k = match (limit, *top_k) {
                    (Some(l), Some(k)) => Some(l.min(k)),
                    (Some(l), None) => Some(l),
                    (None, k) => k,
                };

                let changed = input_changed || new_top_k != *top_k;

                Ok((
                    LogicalOp::unary(
                        UnaryOp::VectorRank {
                            embedding: embedding.clone(),
                            top_k: new_top_k,
                            property_key: property_key.clone(),
                            metric: *metric,
                            threshold: *threshold,
                            score_alias: score_alias.clone(),
                        },
                        optimized_input,
                    ),
                    changed,
                ))
            }

            // Sort: propagate limit for top-k sort optimization
            // (handled at physical plan level, but we track it)
            LogicalOp::Unary {
                op: UnaryOp::Sort { key, descending },
                input,
            } => {
                let (optimized_input, changed) = self.push_down(input, None)?;
                Ok((
                    LogicalOp::unary(
                        UnaryOp::Sort {
                            key: key.clone(),
                            descending: *descending,
                        },
                        optimized_input,
                    ),
                    changed,
                ))
            }

            // Filter: propagate limit through (filtering might reduce result count)
            LogicalOp::Unary {
                op: UnaryOp::Filter(predicate),
                input,
            } => {
                let (optimized_input, changed) = self.push_down(input, None)?; // A filter reduces the row count. We cannot propagate a strict limit through it.
                Ok((
                    LogicalOp::unary(UnaryOp::Filter(predicate.clone()), optimized_input),
                    changed,
                ))
            }

            // Project: propagate limit through (projection doesn't change row count)
            LogicalOp::Unary {
                op: UnaryOp::Project(props),
                input,
            } => {
                let (optimized_input, changed) = self.push_down(input, limit)?;
                Ok((
                    LogicalOp::unary(UnaryOp::Project(props.clone()), optimized_input),
                    changed,
                ))
            }

            // Other unary ops: recursively optimize
            LogicalOp::Unary { op, input } => {
                let (optimized_input, changed) = self.push_down(input, None)?;
                Ok((LogicalOp::unary(op.clone(), optimized_input), changed))
            }

            // Binary: recursively optimize both branches (don't propagate limit)
            LogicalOp::Binary { op, left, right } => {
                let (opt_left, left_changed) = self.push_down(left, None)?;
                let (opt_right, right_changed) = self.push_down(right, None)?;
                Ok((
                    LogicalOp::binary(op.clone(), opt_left, opt_right),
                    left_changed || right_changed,
                ))
            }

            // Leaf nodes: no change
            LogicalOp::Scan(_) | LogicalOp::Empty => Ok((op.clone(), false)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::NodeId;
    use crate::query::plan::ScanOp;
    use std::sync::Arc;

    fn test_stats() -> Statistics {
        Statistics::default()
    }

    #[test]
    fn test_combine_consecutive_limits() {
        let rule = LimitPushdown;
        let stats = test_stats();

        // Limit(5, Limit(10, Scan)) -> Limit(5, Scan)
        let plan = LogicalPlan::new(LogicalOp::unary(
            UnaryOp::Limit(5),
            LogicalOp::unary(
                UnaryOp::Limit(10),
                LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
            ),
        ));

        let result = rule.apply(&plan, &stats).unwrap();
        assert!(result.is_some());

        let new_plan = result.unwrap();
        // Should be Limit(5, Scan) - no nested limit
        match &new_plan.root {
            LogicalOp::Unary {
                op: UnaryOp::Limit(n),
                input,
            } => {
                assert_eq!(*n, 5);
                assert!(matches!(input.as_ref(), LogicalOp::Scan(_)));
            }
            _ => panic!("Expected Limit"),
        }
    }

    #[test]
    fn test_propagate_limit_to_vector_rank() {
        let rule = LimitPushdown;
        let stats = test_stats();

        // Limit(5, VectorRank(top_k=10, Scan)) -> Limit(5, VectorRank(top_k=5, Scan))
        let plan = LogicalPlan::new(LogicalOp::unary(
            UnaryOp::Limit(5),
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
        assert!(result.is_some());

        let new_plan = result.unwrap();
        // VectorRank should have top_k=5 now
        match &new_plan.root {
            LogicalOp::Unary {
                op: UnaryOp::Limit(_),
                input,
            } => match input.as_ref() {
                LogicalOp::Unary {
                    op: UnaryOp::VectorRank { top_k, .. },
                    ..
                } => {
                    assert_eq!(*top_k, Some(5));
                }
                _ => panic!("Expected VectorRank"),
            },
            _ => panic!("Expected Limit"),
        }
    }

    #[test]
    fn test_no_change_for_simple_limit() {
        let rule = LimitPushdown;
        let stats = test_stats();

        let plan = LogicalPlan::new(LogicalOp::unary(
            UnaryOp::Limit(10),
            LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
        ));

        let result = rule.apply(&plan, &stats).unwrap();
        assert!(result.is_none()); // No change needed
    }

    #[test]
    fn test_propagate_limit_through_filter() {
        use crate::query::ir::{Predicate, PredicateValue};
        let rule = LimitPushdown;
        let stats = test_stats();
        let plan = LogicalPlan::new(LogicalOp::unary(
            UnaryOp::Limit(5),
            LogicalOp::unary(
                UnaryOp::Filter(Predicate::eq(
                    "name".to_string(),
                    PredicateValue::String("Alice".to_string()),
                )),
                LogicalOp::unary(
                    UnaryOp::Limit(10),
                    LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
                ),
            ),
        ));

        let result = rule.apply(&plan, &stats).unwrap();
        assert!(result.is_none()); // We shouldn't push the limit down through a filter
    }

    #[test]
    fn test_propagate_limit_through_project() {
        let rule = LimitPushdown;
        let stats = test_stats();
        let plan = LogicalPlan::new(LogicalOp::unary(
            UnaryOp::Limit(5),
            LogicalOp::unary(
                UnaryOp::Project(vec!["name".to_string()]),
                LogicalOp::unary(
                    UnaryOp::Limit(10),
                    LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
                ),
            ),
        ));

        let result = rule.apply(&plan, &stats).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_binary_op_limit_pushdown() {
        let rule = LimitPushdown;
        let stats = test_stats();
        let plan = LogicalPlan::new(LogicalOp::unary(
            UnaryOp::Limit(5),
            LogicalOp::binary(
                crate::query::plan::BinaryOp::Union,
                LogicalOp::unary(
                    UnaryOp::Limit(10),
                    LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
                ),
                LogicalOp::unary(
                    UnaryOp::Limit(15),
                    LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(2).unwrap()])),
                ),
            ),
        ));
        let result = rule.apply(&plan, &stats).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_binary_op_limit_pushdown_children() {
        let rule = LimitPushdown;
        let stats = test_stats();
        let plan = LogicalPlan::new(LogicalOp::binary(
            crate::query::plan::BinaryOp::Union,
            LogicalOp::unary(
                UnaryOp::Limit(10),
                LogicalOp::unary(
                    UnaryOp::Limit(20),
                    LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
                ),
            ),
            LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(2).unwrap()])),
        ));
        let result = rule.apply(&plan, &stats).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_propagate_limit_to_vector_rank_equal_limit() {
        let rule = LimitPushdown;
        let stats = test_stats();
        let plan = LogicalPlan::new(LogicalOp::unary(
            UnaryOp::Limit(10),
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
        assert!(result.is_none());
    }

    #[test]
    fn test_propagate_limit_through_sort() {
        let rule = LimitPushdown;
        let stats = test_stats();
        let plan = LogicalPlan::new(LogicalOp::unary(
            UnaryOp::Limit(5),
            LogicalOp::unary(
                UnaryOp::Sort {
                    key: crate::query::plan::SortKey::Property("age".into()),
                    descending: true,
                },
                LogicalOp::unary(
                    UnaryOp::Limit(10),
                    LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
                ),
            ),
        ));

        let result = rule.apply(&plan, &stats).unwrap();
        assert!(result.is_none()); // Sort requires all elements to sort them before limiting
    }
}

#[cfg(test)]
mod sentry_tests {
    use super::*;
    use crate::core::NodeId;
    use crate::query::plan::{BinaryOp, ScanOp};

    fn test_stats() -> Statistics {
        Statistics::default()
    }

    #[test]
    fn test_pushdown_binary_partial_change() {
        let rule = LimitPushdown;
        let stats = test_stats();

        // Left: Limit(10, Limit(20, Scan)) -> Limit(10, Scan) [changed = true]
        let left = LogicalOp::unary(
            UnaryOp::Limit(10),
            LogicalOp::unary(
                UnaryOp::Limit(20),
                LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
            ),
        );

        // Right: Scan -> Scan [changed = false]
        let right = LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(2).unwrap()]));

        let plan = LogicalPlan::new(LogicalOp::binary(BinaryOp::Union, left, right));

        let result = rule.apply(&plan, &stats).unwrap();
        let expected_plan = LogicalPlan::new(LogicalOp::binary(
            BinaryOp::Union,
            LogicalOp::unary(
                UnaryOp::Limit(10),
                LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
            ),
            LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(2).unwrap()])),
        ));

        assert_eq!(
            result,
            Some(expected_plan),
            "Partial optimization in left branch should propagate with exact limit structure"
        );
    }

    #[test]
    fn test_pushdown_limit_neq_child_limit() {
        let rule = LimitPushdown;

        let op = LogicalOp::unary(
            UnaryOp::Limit(10),
            LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
        );
        let (new_op, changed) = rule.push_down(&op, Some(5)).unwrap();

        assert!(changed, "Expected limit to shrink from 10 to 5");
        if let LogicalOp::Unary {
            op: UnaryOp::Limit(n),
            ..
        } = new_op
        {
            assert_eq!(n, 5);
        } else {
            panic!("Expected limit");
        }
    }

    #[test]
    fn test_pushdown_vector_rank_neq() {
        let rule = LimitPushdown;
        // Vector rank with top_k = None, pass down limit = Some(5)
        let op = LogicalOp::unary(
            UnaryOp::VectorRank {
                embedding: vec![0.1f32].into(),
                top_k: None,
                property_key: None,
                metric: crate::core::vector::DistanceMetric::Cosine,
                threshold: None,
                score_alias: None,
            },
            LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
        );
        let (new_op, changed) = rule.push_down(&op, Some(5)).unwrap();
        assert!(changed, "Vector rank should apply new top_k");
        if let LogicalOp::Unary {
            op: UnaryOp::VectorRank { top_k, .. },
            ..
        } = new_op
        {
            assert_eq!(top_k, Some(5));
        } else {
            panic!("Expected VectorRank");
        }
    }

    #[test]
    fn test_pushdown_unary_project_partial_change() {
        let rule = LimitPushdown;
        let stats = test_stats();

        // Limit(5, Project(["a"], Limit(10, Scan))) -> Limit(5, Project(["a"], Limit(5, Scan)))
        let plan = LogicalPlan::new(LogicalOp::unary(
            UnaryOp::Limit(5),
            LogicalOp::unary(
                UnaryOp::Project(vec!["a".to_string()]),
                LogicalOp::unary(
                    UnaryOp::Limit(10),
                    LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
                ),
            ),
        ));

        let result = rule.apply(&plan, &stats).unwrap();

        let expected_plan = LogicalPlan::new(LogicalOp::unary(
            UnaryOp::Limit(5),
            LogicalOp::unary(
                UnaryOp::Project(vec!["a".to_string()]),
                LogicalOp::unary(
                    UnaryOp::Limit(5),
                    LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
                ),
            ),
        ));

        assert_eq!(
            result,
            Some(expected_plan),
            "Limit pushdown through project should propagate changed = true"
        );
    }

    #[test]
    fn test_binary_partial_change_right() {
        let rule = LimitPushdown;
        let stats = test_stats();

        // Left: Scan -> Scan [changed = false]
        let left = LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()]));

        // Right: Limit(10, Limit(20, Scan)) -> Limit(10, Scan) [changed = true]
        let right = LogicalOp::unary(
            UnaryOp::Limit(10),
            LogicalOp::unary(
                UnaryOp::Limit(20),
                LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(2).unwrap()])),
            ),
        );

        let plan = LogicalPlan::new(LogicalOp::binary(BinaryOp::Union, left, right));

        let result = rule.apply(&plan, &stats).unwrap();
        let expected_plan = LogicalPlan::new(LogicalOp::binary(
            BinaryOp::Union,
            LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(1).unwrap()])),
            LogicalOp::unary(
                UnaryOp::Limit(10),
                LogicalOp::Scan(ScanOp::NodeLookup(vec![NodeId::new(2).unwrap()])),
            ),
        ));

        assert_eq!(
            result,
            Some(expected_plan),
            "Partial optimization in right branch should propagate changed = true"
        );
    }
}
