//! Operation Reordering Optimization
//!
//! Reorders operations based on estimated costs and selectivity to minimize
//! overall query execution cost.
//!
//! # Optimization Strategy
//!
//! This rule applies three main optimizations:
//!
//! 1. **Filter Reordering**: Push more selective filters deeper into the plan (closer to the data source).
//!    This reduces the number of rows that subsequent operations need to process.
//! 2. **Join Reordering**: Ensure the smaller relation is on the left side of a hash join (build side).
//!    This minimizes the memory footprint of the hash table.
//! 3. **Cost-Based Ordering**: Use cardinality estimates to choose the cheapest operation sequence.
//!
//! # Example: Join Optimization
//!
//! ```text
//! BEFORE: Large Table (Left) JOIN Small Table (Right)
//!         (Requires building hash table on Large Table)
//!
//!         Join
//!        /    \
//!   Large      Small
//!
//! AFTER:  Small Table (Left) JOIN Large Table (Right)
//!         (Builds hash table on Small Table - faster & less memory)
//!
//!         Join
//!        /    \
//!   Small      Large
//! ```

use crate::core::error::Result;
use crate::query::ir::Predicate;
use crate::query::plan::{BinaryOp, LogicalOp, LogicalPlan, ScanOp, UnaryOp};

use super::{OptimizationRule, Statistics};

/// Default cardinality estimate for node scans without statistics.
const DEFAULT_NODE_SCAN_CARDINALITY: usize = 1000;

/// Default cardinality estimate for edge scans without statistics.
const DEFAULT_EDGE_SCAN_CARDINALITY: usize = 1000;

/// Default cardinality estimate for property-indexed scans (~10% of full scan).
const DEFAULT_PROPERTY_SCAN_CARDINALITY: usize = 100;

/// Default selectivity estimate for filter predicates (10%).
const DEFAULT_FILTER_SELECTIVITY: f64 = 0.1;

/// Default selectivity for join operations (10% of cross product).
const DEFAULT_JOIN_SELECTIVITY: f64 = 0.1;

// Selectivity estimates for different predicate types
// (0.0 = filters everything, 1.0 = filters nothing)
/// Selectivity for `IS NULL` checks (0.1). Assumes nulls are relatively rare.
const NULL_CHECK_SELECTIVITY: f64 = 0.1;
/// Selectivity for existence checks (0.1). Assumes checking for a specific property filters well.
const EXISTENCE_CHECK_SELECTIVITY: f64 = 0.1;
/// Selectivity for `IN` predicates (0.15).
const IN_PREDICATE_SELECTIVITY: f64 = 0.15;
/// Selectivity for string matching (Contains/StartsWith/EndsWith) (0.2).
const STRING_PREDICATE_SELECTIVITY: f64 = 0.2;
/// Selectivity for range predicates (Gt/Lt/Gte/Lte) (0.3).
const RANGE_PREDICATE_SELECTIVITY: f64 = 0.3;
/// Selectivity for not-equals (!=) (0.9). These typically filter very little.
const NOT_EQUALS_SELECTIVITY: f64 = 0.9;
/// Selectivity for `TRUE` (1.0). Filters nothing.
const TRUE_SELECTIVITY: f64 = 1.0;
/// Selectivity for `FALSE` (0.0). Filters everything.
const FALSE_SELECTIVITY: f64 = 0.0;
// Note: AND/OR/NOT selectivity is computed dynamically based on child predicates

/// Operation reordering optimization rule.
///
/// This rule reorders operations based on cost estimates to minimize
/// total execution time.
///
/// # Examples
///
/// ```text
/// Input Plan:
/// Filter(B) -> Filter(A) -> Scan
/// (Where Filter A is very selective, and Filter B is not)
///
/// Optimized Plan:
/// Filter(A) -> Filter(B) -> Scan
/// (Filter A is applied first, reducing rows for Filter B)
/// ```
///
/// ## Examples
///
/// ```rust
/// use aletheiadb::query::planner::rules::{OptimizationRule, OperationReordering};
/// use aletheiadb::query::planner::stats::Statistics;
/// use aletheiadb::query::plan::{LogicalPlan, LogicalOp, UnaryOp, ScanOp};
/// use aletheiadb::query::ir::{Predicate, PredicateValue};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // 1. Construct a sub-optimal plan: A very selective filter (Eq) is ABOVE a less selective one (NotEq)
/// let scan = LogicalOp::Scan(ScanOp::NodeScan {
///     label: Some("Person".into()),
///     estimated_rows: Some(100),
/// });
/// let filter_bad = LogicalOp::unary(
///     UnaryOp::Filter(Predicate::Ne {
///         key: "status".into(),
///         value: PredicateValue::String("deleted".into())
///     }),
///     scan
/// );
/// let filter_good = LogicalOp::unary(
///     UnaryOp::Filter(Predicate::Eq {
///         key: "id".into(),
///         value: PredicateValue::Int(42)
///     }),
///     filter_bad
/// );
///
/// let plan = LogicalPlan { root: filter_good, temporal_context: None, hints: Default::default() };
///
/// // 2. Apply the rule
/// let rule = OperationReordering;
/// let stats = Statistics::new();
/// let optimized_plan = rule.apply(&plan, &stats)?.unwrap(); // unwraps if `changed == true`
///
/// // 3. The rule reorders them so the more selective Filter(Eq) is applied first (deeper in the tree)
/// if let LogicalOp::Unary { op: UnaryOp::Filter(Predicate::Ne { .. }), input } = optimized_plan.root {
///     assert!(matches!(
///         *input,
///         LogicalOp::Unary { op: UnaryOp::Filter(Predicate::Eq { .. }), .. }
///     ));
/// } else {
///     panic!("Expected Ne filter at root after reordering");
/// }
/// # Ok(())
/// # }
/// ```
pub struct OperationReordering;

impl OptimizationRule for OperationReordering {
    fn name(&self) -> &str {
        "operation-reordering"
    }

    fn apply(&self, plan: &LogicalPlan, stats: &Statistics) -> Result<Option<LogicalPlan>> {
        let (new_root, changed) = self.reorder(&plan.root, stats)?;

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

impl OperationReordering {
    /// Recursively reorder operations in the plan tree.
    fn reorder(&self, op: &LogicalOp, stats: &Statistics) -> Result<(LogicalOp, bool)> {
        match op {
            // Check if this is a sequence of filters that can be reordered
            LogicalOp::Unary {
                op: UnaryOp::Filter(_),
                ..
            } => {
                // Collect all consecutive filters
                let (filters, base) = self.collect_filters(op);

                if filters.len() > 1 {
                    // Reorder filters by selectivity (most selective first = deepest)
                    let reordered = self.reorder_filters(filters, base, stats)?;
                    // Check if order actually changed
                    let changed = !self.filters_equal(op, &reordered);
                    Ok((reordered, changed))
                } else {
                    // Single filter or no filter - just recurse
                    if let LogicalOp::Unary {
                        op: filter_op,
                        input,
                    } = op
                    {
                        let (new_input, changed) = self.reorder(input, stats)?;
                        Ok((LogicalOp::unary(filter_op.clone(), new_input), changed))
                    } else {
                        Ok((op.clone(), false))
                    }
                }
            }

            // Join reordering: put smaller table on left (build side)
            LogicalOp::Binary {
                op:
                    BinaryOp::Join {
                        left_key,
                        right_key,
                    },
                left,
                right,
            } => {
                // First, optimize children
                let (opt_left, left_changed) = self.reorder(left, stats)?;
                let (opt_right, right_changed) = self.reorder(right, stats)?;

                // Estimate cardinalities
                let left_card = self.estimate_cardinality(&opt_left);
                let right_card = self.estimate_cardinality(&opt_right);

                // If right side is smaller, swap them
                let (final_left, final_right, final_left_key, final_right_key, swapped) =
                    if right_card < left_card {
                        (
                            opt_right,
                            opt_left,
                            right_key.clone(),
                            left_key.clone(),
                            true,
                        )
                    } else {
                        (
                            opt_left,
                            opt_right,
                            left_key.clone(),
                            right_key.clone(),
                            false,
                        )
                    };

                Ok((
                    LogicalOp::binary(
                        BinaryOp::Join {
                            left_key: final_left_key,
                            right_key: final_right_key,
                        },
                        final_left,
                        final_right,
                    ),
                    left_changed || right_changed || swapped,
                ))
            }

            // Other unary operations: recursively optimize
            LogicalOp::Unary { op, input } => {
                let (new_input, changed) = self.reorder(input, stats)?;
                Ok((LogicalOp::unary(op.clone(), new_input), changed))
            }

            // Binary operations (non-join): recursively optimize both sides
            LogicalOp::Binary { op, left, right } => {
                let (opt_left, left_changed) = self.reorder(left, stats)?;
                let (opt_right, right_changed) = self.reorder(right, stats)?;
                Ok((
                    LogicalOp::binary(op.clone(), opt_left, opt_right),
                    left_changed || right_changed,
                ))
            }

            // Leaf nodes: no change
            LogicalOp::Scan(_) | LogicalOp::Empty => Ok((op.clone(), false)),
        }
    }

    /// Collect all consecutive filters from a filter chain.
    /// Returns (filters, base) where filters are in top-to-bottom order.
    fn collect_filters(&self, op: &LogicalOp) -> (Vec<Predicate>, LogicalOp) {
        let mut filters = Vec::with_capacity(4); // ⚡ Bolt Optimization: Pre-allocate capacity for query filter lists to prevent multiple reallocations during logical plan extraction.
        let mut current = op;

        while let LogicalOp::Unary {
            op: UnaryOp::Filter(predicate),
            input,
        } = current
        {
            filters.push(predicate.clone());
            current = input;
        }

        (filters, current.clone())
    }

    /// Reorder filters by selectivity (most selective applied first = deepest in tree).
    fn reorder_filters(
        &self,
        mut filters: Vec<Predicate>,
        base: LogicalOp,
        stats: &Statistics,
    ) -> Result<LogicalOp> {
        // Sort filters by selectivity (ascending = most selective first)
        filters.sort_by(|a, b| {
            let sel_a = self.estimate_filter_selectivity(a, stats);
            let sel_b = self.estimate_filter_selectivity(b, stats);
            sel_a
                .partial_cmp(&sel_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Build filter chain with most selective at bottom (applied first)
        let mut result = base;
        for filter in filters {
            result = LogicalOp::unary(UnaryOp::Filter(filter), result);
        }

        Ok(result)
    }

    /// Estimate filter selectivity (0.0 = filters everything, 1.0 = filters nothing).
    ///
    /// # Heuristics
    ///
    /// - **Equality**: Uses statistical histograms if available.
    /// - **Range**: Assumes 30% selectivity (`RANGE_PREDICATE_SELECTIVITY`).
    /// - **String**: Assumes 20% selectivity (`STRING_PREDICATE_SELECTIVITY`).
    /// - **Logic**:
    ///   - `AND`: Product of probabilities (intersection).
    ///   - `OR`: 1 - Product of complement probabilities (union).
    ///   - `NOT`: 1 - Probability (complement).
    fn estimate_filter_selectivity(&self, predicate: &Predicate, stats: &Statistics) -> f64 {
        match predicate {
            Predicate::Eq { key, value } => {
                // Special case: Null checks are typically selective
                if matches!(value, crate::query::ir::PredicateValue::Null) {
                    return NULL_CHECK_SELECTIVITY;
                }

                // Convert value to string for selectivity estimation
                let value_str = match value {
                    crate::query::ir::PredicateValue::String(s) => s.clone(),
                    crate::query::ir::PredicateValue::Int(i) => i.to_string(),
                    crate::query::ir::PredicateValue::Float(f) => f.to_string(),
                    crate::query::ir::PredicateValue::Bool(b) => b.to_string(),
                    crate::query::ir::PredicateValue::Null => unreachable!(), // Already handled above
                };
                stats.estimate_selectivity(key, &value_str)
            }
            Predicate::Gt { .. }
            | Predicate::Lt { .. }
            | Predicate::Gte { .. }
            | Predicate::Lte { .. } => RANGE_PREDICATE_SELECTIVITY,
            Predicate::Contains { .. }
            | Predicate::StartsWith { .. }
            | Predicate::EndsWith { .. } => STRING_PREDICATE_SELECTIVITY,
            Predicate::And(predicates) => {
                // For AND: multiply selectivities (intersection rule)
                // Empty AND defaults to high selectivity (filters nothing)
                if predicates.is_empty() {
                    return TRUE_SELECTIVITY;
                }
                predicates
                    .iter()
                    .map(|p| self.estimate_filter_selectivity(p, stats))
                    .product()
            }
            Predicate::Or(predicates) => {
                // For OR: use union rule: 1 - (1-sel1) * (1-sel2) * ...
                // Empty OR defaults to low selectivity (filters everything)
                if predicates.is_empty() {
                    return FALSE_SELECTIVITY;
                }
                let complement_product: f64 = predicates
                    .iter()
                    .map(|p| 1.0 - self.estimate_filter_selectivity(p, stats))
                    .product();
                1.0 - complement_product
            }
            Predicate::Not(inner) => {
                // For NOT: complement of inner selectivity
                1.0 - self.estimate_filter_selectivity(inner, stats)
            }
            // Edge-scoped leaves (Issue #3622) are evaluated against the row's
            // traversed edge; estimate from the wrapped predicate so an edge
            // equality still orders ahead of an edge range, etc.
            Predicate::EdgeScoped(inner) => self.estimate_filter_selectivity(inner, stats),
            Predicate::Ne { .. } => NOT_EQUALS_SELECTIVITY,
            Predicate::In { .. } => IN_PREDICATE_SELECTIVITY,
            Predicate::Exists(_) | Predicate::NotExists(_) => EXISTENCE_CHECK_SELECTIVITY,
            // Provenance predicates (Issue #3354a) require a per-version
            // metadata lookup with no column statistics available; treat them
            // like a range predicate for ordering purposes (moderately
            // selective, so they run after cheap equality filters).
            Predicate::Provenance(_) => RANGE_PREDICATE_SELECTIVITY,
            Predicate::True => TRUE_SELECTIVITY,
            Predicate::False => FALSE_SELECTIVITY,
        }
    }

    /// Estimate cardinality of a logical operation's output.
    fn estimate_cardinality(&self, op: &LogicalOp) -> usize {
        match op {
            LogicalOp::Scan(scan) => match scan {
                ScanOp::NodeLookup(ids) => ids.len(),
                ScanOp::NodeScan { estimated_rows, .. } => {
                    estimated_rows.unwrap_or(DEFAULT_NODE_SCAN_CARDINALITY)
                }
                ScanOp::VectorSearch { k, .. } => *k,
                ScanOp::TemporalNodeLookup { node_ids, .. } => node_ids.len(),
                ScanOp::TemporalVectorSearch { k, .. } => *k,
                ScanOp::SimilarToNode { k, .. } => *k,
                ScanOp::PropertyScan { .. } => DEFAULT_PROPERTY_SCAN_CARDINALITY,
                ScanOp::EdgeScan { estimated_rows, .. } => {
                    estimated_rows.unwrap_or(DEFAULT_EDGE_SCAN_CARDINALITY)
                }
            },
            LogicalOp::Unary {
                op: UnaryOp::Filter(_),
                input,
            } => (self.estimate_cardinality(input) as f64 * DEFAULT_FILTER_SELECTIVITY) as usize,
            LogicalOp::Unary {
                op: UnaryOp::Limit(n),
                ..
            } => *n,
            LogicalOp::Unary { input, .. } => self.estimate_cardinality(input),
            LogicalOp::Binary { left, right, .. } => {
                let left_card = self.estimate_cardinality(left);
                let right_card = self.estimate_cardinality(right);
                (left_card as f64 * right_card as f64 * DEFAULT_JOIN_SELECTIVITY) as usize
            }
            LogicalOp::Empty => 0,
        }
    }

    /// Check if two filter chains are equal (same filters in same order).
    ///
    /// `LogicalOp` and `Predicate` both derive `PartialEq`, so a direct `==`
    /// comparison is sufficient.
    fn filters_equal(&self, a: &LogicalOp, b: &LogicalOp) -> bool {
        a == b
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::NodeId;
    use crate::query::ir::Predicate;
    use crate::query::plan::{BinaryOp, ScanOp, UnaryOp};

    fn test_stats() -> Statistics {
        let stats = Statistics::default();
        // Set up some statistics for testing
        stats.refresh(1000, 5000, 100, vec![], 5.0);

        // Set up property statistics for selectivity estimation
        stats.update_property_stats("rare_property", 10); // Very selective (1/10)
        stats.update_property_stats("common_property", 500); // Less selective (1/500)

        stats
    }

    // ==================== Filter Reordering Tests ====================

    #[test]
    fn test_reorder_filters_by_selectivity() {
        let rule = OperationReordering;
        let stats = test_stats();

        // Original: Filter(common) -> Filter(rare) -> Scan
        // common_property: 500 distinct values → 0.2% selectivity (MORE selective)
        // rare_property: 10 distinct values → 10% selectivity (LESS selective)
        // Should be reordered to: Filter(rare) -> Filter(common) -> Scan
        // (most selective at bottom/applied first)
        let plan = LogicalPlan::new(LogicalOp::unary(
            UnaryOp::Filter(Predicate::eq("common_property", "value")),
            LogicalOp::unary(
                UnaryOp::Filter(Predicate::eq("rare_property", "value")),
                LogicalOp::Scan(ScanOp::NodeScan {
                    label: None,
                    estimated_rows: Some(1000),
                }),
            ),
        ));

        let result = rule.apply(&plan, &stats).unwrap();

        let expected_plan = LogicalPlan::new(LogicalOp::unary(
            UnaryOp::Filter(Predicate::eq("rare_property", "value")),
            LogicalOp::unary(
                UnaryOp::Filter(Predicate::eq("common_property", "value")),
                LogicalOp::Scan(ScanOp::NodeScan {
                    label: None,
                    estimated_rows: Some(1000),
                }),
            ),
        ));

        assert_eq!(result, Some(expected_plan), "Should reorder filters by selectivity");
    }


    #[test]
    fn test_no_reorder_when_filters_already_optimal() {
        let rule = OperationReordering;
        let stats = test_stats();

        // Filter(rare - less selective) -> Filter(common - more selective) -> Scan
        // Already optimal: most selective at bottom
        let plan = LogicalPlan::new(LogicalOp::unary(
            UnaryOp::Filter(Predicate::eq("rare_property", "value")),
            LogicalOp::unary(
                UnaryOp::Filter(Predicate::eq("common_property", "value")),
                LogicalOp::Scan(ScanOp::NodeScan {
                    label: None,
                    estimated_rows: Some(1000),
                }),
            ),
        ));

        let result = rule.apply(&plan, &stats).unwrap();
        // Should return None since already optimal
        assert!(
            result.is_none(),
            "Should not reorder already optimal filters"
        );
    }

    #[test]
    fn test_reorder_three_filters() {
        let rule = OperationReordering;
        let stats = test_stats();

        // Add a third property with medium selectivity
        stats.update_property_stats("medium_property", 100); // Medium (1/100)

        // Filter(common) -> Filter(medium) -> Filter(rare) -> Scan
        // Should become: Filter(rare) -> Filter(medium) -> Filter(common) -> Scan
        let plan = LogicalPlan::new(LogicalOp::unary(
            UnaryOp::Filter(Predicate::eq("common_property", "value")),
            LogicalOp::unary(
                UnaryOp::Filter(Predicate::eq("medium_property", "value")),
                LogicalOp::unary(
                    UnaryOp::Filter(Predicate::eq("rare_property", "value")),
                    LogicalOp::Scan(ScanOp::NodeScan {
                        label: None,
                        estimated_rows: Some(1000),
                    }),
                ),
            ),
        ));

        let result = rule.apply(&plan, &stats).unwrap();

        let expected_plan = LogicalPlan::new(LogicalOp::unary(
            UnaryOp::Filter(Predicate::eq("rare_property", "value")),
            LogicalOp::unary(
                UnaryOp::Filter(Predicate::eq("medium_property", "value")),
                LogicalOp::unary(
                    UnaryOp::Filter(Predicate::eq("common_property", "value")),
                    LogicalOp::Scan(ScanOp::NodeScan {
                        label: None,
                        estimated_rows: Some(1000),
                    }),
                ),
            ),
        ));
        assert_eq!(result, Some(expected_plan), "Should reorder three filters");
    }

    // ==================== Join Reordering Tests ====================

    #[test]
    fn test_reorder_join_operands_by_size() {
        let rule = OperationReordering;
        let stats = test_stats();

        // Join with large left side and small right side
        // Should be reordered to put small side first (for hash join build)
        let large_scan = LogicalOp::Scan(ScanOp::NodeScan {
            label: Some("LargeTable".to_string()),
            estimated_rows: Some(10000),
        });

        let small_scan = LogicalOp::Scan(ScanOp::NodeScan {
            label: Some("SmallTable".to_string()),
            estimated_rows: Some(100),
        });

        let plan = LogicalPlan::new(LogicalOp::binary(
            BinaryOp::Join {
                left_key: "id".to_string(),
                right_key: "ref_id".to_string(),
            },
            large_scan,
            small_scan,
        ));

        let result = rule.apply(&plan, &stats).unwrap();

        let expected_plan = LogicalPlan::new(LogicalOp::binary(
            BinaryOp::Join {
                left_key: "ref_id".to_string(),
                right_key: "id".to_string(),
            },
            LogicalOp::Scan(ScanOp::NodeScan {
                label: Some("SmallTable".to_string()),
                estimated_rows: Some(100),
            }),
            LogicalOp::Scan(ScanOp::NodeScan {
                label: Some("LargeTable".to_string()),
                estimated_rows: Some(10000),
            }),
        ));
        assert_eq!(result, Some(expected_plan), "Should reorder join operands");
    }


    #[test]
    fn test_no_reorder_when_join_already_optimal() {
        let rule = OperationReordering;
        let stats = test_stats();

        let small_scan = LogicalOp::Scan(ScanOp::NodeScan {
            label: Some("SmallTable".to_string()),
            estimated_rows: Some(100),
        });

        let large_scan = LogicalOp::Scan(ScanOp::NodeScan {
            label: Some("LargeTable".to_string()),
            estimated_rows: Some(10000),
        });

        // Small already on left - optimal
        let plan = LogicalPlan::new(LogicalOp::binary(
            BinaryOp::Join {
                left_key: "id".to_string(),
                right_key: "ref_id".to_string(),
            },
            small_scan,
            large_scan,
        ));

        let result = rule.apply(&plan, &stats).unwrap();
        assert!(result.is_none(), "Should not reorder optimal join");
    }

    // ==================== Complex Query Reordering Tests ====================

    #[test]
    fn test_reorder_complex_query() {
        let rule = OperationReordering;
        let stats = test_stats();

        // Complex query with multiple opportunities for reordering:
        // Join(large, small) with filters in non-optimal order
        let plan = LogicalPlan::new(LogicalOp::binary(
            BinaryOp::Join {
                left_key: "id".to_string(),
                right_key: "ref_id".to_string(),
            },
            // Left side: large scan with non-optimal filter order
            LogicalOp::unary(
                UnaryOp::Filter(Predicate::eq("common_property", "value")),
                LogicalOp::Scan(ScanOp::NodeScan {
                    label: Some("LargeTable".to_string()),
                    estimated_rows: Some(10000),
                }),
            ),
            // Right side: small scan
            LogicalOp::Scan(ScanOp::NodeScan {
                label: Some("SmallTable".to_string()),
                estimated_rows: Some(100),
            }),
        ));

        let result = rule.apply(&plan, &stats).unwrap();

        let expected_plan = LogicalPlan::new(LogicalOp::binary(
            BinaryOp::Join {
                left_key: "ref_id".to_string(),
                right_key: "id".to_string(),
            },
            LogicalOp::Scan(ScanOp::NodeScan {
                label: Some("SmallTable".to_string()),
                estimated_rows: Some(100),
            }),
            LogicalOp::unary(
                UnaryOp::Filter(Predicate::eq("common_property", "value")),
                LogicalOp::Scan(ScanOp::NodeScan {
                    label: Some("LargeTable".to_string()),
                    estimated_rows: Some(10000),
                }),
            ),
        ));
        assert_eq!(result, Some(expected_plan), "Should optimize complex query");
    }

    // ==================== Edge Cases ====================

    #[test]
    fn test_no_change_for_simple_scan() {
        let rule = OperationReordering;
        let stats = test_stats();

        let plan = LogicalPlan::new(LogicalOp::Scan(ScanOp::NodeLookup(vec![
            NodeId::new(1).unwrap(),
        ])));

        let result = rule.apply(&plan, &stats).unwrap();
        assert!(result.is_none(), "Should not change simple scan");
    }

    #[test]
    fn test_single_filter_no_reorder() {
        let rule = OperationReordering;
        let stats = test_stats();

        let plan = LogicalPlan::new(LogicalOp::unary(
            UnaryOp::Filter(Predicate::eq("name", "Alice")),
            LogicalOp::Scan(ScanOp::NodeScan {
                label: None,
                estimated_rows: Some(1000),
            }),
        ));

        let result = rule.apply(&plan, &stats).unwrap();
        assert!(result.is_none(), "Single filter cannot be reordered");
    }

    #[test]
    fn test_selectivity_and_predicate() {
        let rule = OperationReordering;
        let stats = test_stats();

        // AND of two predicates: selectivity should be product
        // Use predicates with fixed selectivity (not dependent on property stats)
        let pred = Predicate::And(vec![
            Predicate::gt("score", 50i64), // RANGE_PREDICATE_SELECTIVITY = 0.3
            Predicate::contains("name", "test"), // STRING_PREDICATE_SELECTIVITY = 0.2
        ]);

        let sel = rule.estimate_filter_selectivity(&pred, &stats);
        // Expected: 0.3 * 0.2 = 0.06
        assert!(
            (sel - 0.06).abs() < 0.001,
            "AND selectivity should be product, got {}",
            sel
        );
    }

    #[test]
    fn test_selectivity_or_predicate() {
        let rule = OperationReordering;
        let stats = test_stats();

        // OR of two predicates: selectivity = 1 - (1-sel1) * (1-sel2)
        // Use predicates with fixed selectivity
        let pred = Predicate::Or(vec![
            Predicate::gt("score", 50i64), // RANGE_PREDICATE_SELECTIVITY = 0.3
            Predicate::contains("name", "test"), // STRING_PREDICATE_SELECTIVITY = 0.2
        ]);

        let sel = rule.estimate_filter_selectivity(&pred, &stats);
        // Expected: 1 - (1-0.3) * (1-0.2) = 1 - 0.7 * 0.8 = 1 - 0.56 = 0.44
        assert!(
            (sel - 0.44).abs() < 0.001,
            "OR selectivity should use union rule, got {}",
            sel
        );
    }

    #[test]
    fn test_selectivity_not_predicate() {
        let rule = OperationReordering;
        let stats = test_stats();

        // NOT of a predicate: selectivity = 1 - inner_sel
        // Use predicate with fixed selectivity
        let pred = Predicate::Not(Box::new(Predicate::gt("score", 50i64))); // RANGE_PREDICATE_SELECTIVITY = 0.3

        let sel = rule.estimate_filter_selectivity(&pred, &stats);
        // Expected: 1 - 0.3 = 0.7
        assert!(
            (sel - 0.7).abs() < 0.001,
            "NOT selectivity should be complement, got {}",
            sel
        );
    }

    #[test]
    fn test_selectivity_empty_and() {
        let rule = OperationReordering;
        let stats = test_stats();

        // Empty AND should have high selectivity (filters nothing)
        let pred = Predicate::And(vec![]);
        let sel = rule.estimate_filter_selectivity(&pred, &stats);
        assert_eq!(sel, TRUE_SELECTIVITY);
    }

    #[test]
    fn test_selectivity_empty_or() {
        let rule = OperationReordering;
        let stats = test_stats();

        // Empty OR should have low selectivity (filters everything)
        let pred = Predicate::Or(vec![]);
        let sel = rule.estimate_filter_selectivity(&pred, &stats);
        assert_eq!(sel, FALSE_SELECTIVITY);
    }

    #[test]
    fn test_selectivity_nested_predicates() {
        let rule = OperationReordering;
        let stats = test_stats();

        // Nested: AND(OR(a, b), c)
        // Use predicates with fixed selectivity
        let pred = Predicate::And(vec![
            Predicate::Or(vec![
                Predicate::gt("score", 50i64),       // RANGE = 0.3
                Predicate::contains("name", "test"), // STRING = 0.2
            ]),
            Predicate::lt("age", 100i64), // RANGE_PREDICATE_SELECTIVITY = 0.3
        ]);

        let sel = rule.estimate_filter_selectivity(&pred, &stats);
        // OR: 1 - (1-0.3) * (1-0.2) = 1 - 0.7 * 0.8 = 0.44
        // AND: 0.44 * 0.3 = 0.132
        assert!(
            (sel - 0.132).abs() < 0.001,
            "Nested predicates should compute correctly, got {}",
            sel
        );
    }

    #[test]
    fn test_selectivity_all_types() {
        let rule = OperationReordering;
        let stats = test_stats();

        // Test all predicate types have defined selectivity
        assert_eq!(
            rule.estimate_filter_selectivity(&Predicate::True, &stats),
            TRUE_SELECTIVITY
        );
        assert_eq!(
            rule.estimate_filter_selectivity(&Predicate::False, &stats),
            FALSE_SELECTIVITY
        );
        assert_eq!(
            rule.estimate_filter_selectivity(&Predicate::gt("x", 1i64), &stats),
            RANGE_PREDICATE_SELECTIVITY
        );
        assert_eq!(
            rule.estimate_filter_selectivity(&Predicate::lt("x", 1i64), &stats),
            RANGE_PREDICATE_SELECTIVITY
        );
        assert_eq!(
            rule.estimate_filter_selectivity(&Predicate::contains("x", "y"), &stats),
            STRING_PREDICATE_SELECTIVITY
        );
        assert_eq!(
            rule.estimate_filter_selectivity(
                &Predicate::StartsWith {
                    key: "x".to_string(),
                    prefix: "y".to_string()
                },
                &stats
            ),
            STRING_PREDICATE_SELECTIVITY
        );
        assert_eq!(
            rule.estimate_filter_selectivity(
                &Predicate::EndsWith {
                    key: "x".to_string(),
                    suffix: "y".to_string()
                },
                &stats
            ),
            STRING_PREDICATE_SELECTIVITY
        );
        assert_eq!(
            rule.estimate_filter_selectivity(&Predicate::ne("x", 1i64), &stats),
            NOT_EQUALS_SELECTIVITY
        );
        assert_eq!(
            rule.estimate_filter_selectivity(
                &Predicate::In {
                    key: "x".to_string(),
                    values: vec![
                        crate::query::ir::PredicateValue::Int(1),
                        crate::query::ir::PredicateValue::Int(2)
                    ]
                },
                &stats
            ),
            IN_PREDICATE_SELECTIVITY
        );
        assert_eq!(
            rule.estimate_filter_selectivity(&Predicate::exists("x"), &stats),
            EXISTENCE_CHECK_SELECTIVITY
        );
        assert_eq!(
            rule.estimate_filter_selectivity(&Predicate::NotExists("x".to_string()), &stats),
            EXISTENCE_CHECK_SELECTIVITY
        );
    }

    #[test]
    fn test_predicates_equal_basic() {
        // Same predicates should be equal (Predicate derives PartialEq)
        let p1 = Predicate::eq("name", "Alice");
        let p2 = Predicate::eq("name", "Alice");
        assert_eq!(p1, p2);

        // Different values should not be equal
        let p3 = Predicate::eq("name", "Bob");
        assert_ne!(p1, p3);

        // Different keys should not be equal
        let p4 = Predicate::eq("age", "Alice");
        assert_ne!(p1, p4);
    }

    #[test]
    fn test_predicates_equal_different_types() {
        // Different predicate types should not be equal
        let p1 = Predicate::eq("name", "Alice");
        let p2 = Predicate::ne("name", "Alice");
        assert_ne!(p1, p2);

        let p3 = Predicate::gt("age", 30i64);
        let p4 = Predicate::lt("age", 30i64);
        assert_ne!(p3, p4);
    }

    #[test]
    fn test_predicates_equal_and_or() {
        // Same AND predicates
        let p1 = Predicate::And(vec![
            Predicate::eq("name", "Alice"),
            Predicate::gt("age", 30i64),
        ]);
        let p2 = Predicate::And(vec![
            Predicate::eq("name", "Alice"),
            Predicate::gt("age", 30i64),
        ]);
        assert_eq!(p1, p2);

        // Different order in AND should not be equal (structural comparison)
        let p3 = Predicate::And(vec![
            Predicate::gt("age", 30i64),
            Predicate::eq("name", "Alice"),
        ]);
        assert_ne!(p1, p3);

        // Different length AND
        let p4 = Predicate::And(vec![Predicate::eq("name", "Alice")]);
        assert_ne!(p1, p4);

        // AND vs OR
        let p5 = Predicate::Or(vec![
            Predicate::eq("name", "Alice"),
            Predicate::gt("age", 30i64),
        ]);
        assert_ne!(p1, p5);
    }

    #[test]
    fn test_predicates_equal_not() {
        // Same NOT predicates
        let p1 = Predicate::Not(Box::new(Predicate::eq("active", true)));
        let p2 = Predicate::Not(Box::new(Predicate::eq("active", true)));
        assert_eq!(p1, p2);

        // Different inner predicates
        let p3 = Predicate::Not(Box::new(Predicate::eq("active", false)));
        assert_ne!(p1, p3);
    }

    #[test]
    fn test_predicates_equal_all_variants() {
        // Test all variants for coverage (Predicate derives PartialEq)
        assert_eq!(Predicate::True, Predicate::True);
        assert_eq!(Predicate::False, Predicate::False);
        assert_ne!(Predicate::True, Predicate::False);

        // String predicates
        let c1 = Predicate::contains("text", "hello");
        let c2 = Predicate::contains("text", "hello");
        assert_eq!(c1, c2);

        let s1 = Predicate::StartsWith {
            key: "text".to_string(),
            prefix: "hello".to_string(),
        };
        let s2 = Predicate::StartsWith {
            key: "text".to_string(),
            prefix: "hello".to_string(),
        };
        assert_eq!(s1, s2);

        let e1 = Predicate::EndsWith {
            key: "text".to_string(),
            suffix: "world".to_string(),
        };
        let e2 = Predicate::EndsWith {
            key: "text".to_string(),
            suffix: "world".to_string(),
        };
        assert_eq!(e1, e2);

        // In predicate
        let i1 = Predicate::In {
            key: "id".to_string(),
            values: vec![
                crate::query::ir::PredicateValue::Int(1),
                crate::query::ir::PredicateValue::Int(2),
            ],
        };
        let i2 = Predicate::In {
            key: "id".to_string(),
            values: vec![
                crate::query::ir::PredicateValue::Int(1),
                crate::query::ir::PredicateValue::Int(2),
            ],
        };
        assert_eq!(i1, i2);

        // Exists predicates
        let ex1 = Predicate::exists("prop");
        let ex2 = Predicate::exists("prop");
        assert_eq!(ex1, ex2);

        let nex1 = Predicate::NotExists("prop".to_string());
        let nex2 = Predicate::NotExists("prop".to_string());
        assert_eq!(nex1, nex2);
    }

    #[test]
    fn test_selectivity_null_check() {
        let rule = OperationReordering;
        let stats = test_stats();

        // Null checks should use NULL_CHECK_SELECTIVITY
        let pred = Predicate::Eq {
            key: "value".to_string(),
            value: crate::query::ir::PredicateValue::Null,
        };

        let sel = rule.estimate_filter_selectivity(&pred, &stats);
        assert_eq!(sel, NULL_CHECK_SELECTIVITY);
    }

    #[test]
    fn test_predicates_equal_exhaustive_mismatches() {
        // Predicate derives PartialEq, so we use assert_ne! directly.

        // Eq mismatches
        assert_ne!(Predicate::eq("a", 1), Predicate::eq("b", 1));
        assert_ne!(Predicate::eq("a", 1), Predicate::eq("a", 2));

        // Ne mismatches
        assert_ne!(Predicate::ne("a", 1), Predicate::ne("b", 1));
        assert_ne!(Predicate::ne("a", 1), Predicate::ne("a", 2));

        // Gt mismatches
        assert_ne!(Predicate::gt("a", 1), Predicate::gt("b", 1));
        assert_ne!(Predicate::gt("a", 1), Predicate::gt("a", 2));

        // Gte mismatches
        assert_ne!(
            Predicate::Gte {
                key: "a".to_string(),
                value: crate::query::ir::PredicateValue::Int(1)
            },
            Predicate::Gte {
                key: "b".to_string(),
                value: crate::query::ir::PredicateValue::Int(1)
            }
        );
        assert_ne!(
            Predicate::Gte {
                key: "a".to_string(),
                value: crate::query::ir::PredicateValue::Int(1)
            },
            Predicate::Gte {
                key: "a".to_string(),
                value: crate::query::ir::PredicateValue::Int(2)
            }
        );

        // Lt mismatches
        assert_ne!(Predicate::lt("a", 1), Predicate::lt("b", 1));
        assert_ne!(Predicate::lt("a", 1), Predicate::lt("a", 2));

        // Lte mismatches
        assert_ne!(
            Predicate::Lte {
                key: "a".to_string(),
                value: crate::query::ir::PredicateValue::Int(1)
            },
            Predicate::Lte {
                key: "b".to_string(),
                value: crate::query::ir::PredicateValue::Int(1)
            }
        );
        assert_ne!(
            Predicate::Lte {
                key: "a".to_string(),
                value: crate::query::ir::PredicateValue::Int(1)
            },
            Predicate::Lte {
                key: "a".to_string(),
                value: crate::query::ir::PredicateValue::Int(2)
            }
        );

        // In mismatches
        assert_ne!(
            Predicate::In {
                key: "a".to_string(),
                values: vec![crate::query::ir::PredicateValue::Int(1)]
            },
            Predicate::In {
                key: "b".to_string(),
                values: vec![crate::query::ir::PredicateValue::Int(1)]
            }
        );
        assert_ne!(
            Predicate::In {
                key: "a".to_string(),
                values: vec![crate::query::ir::PredicateValue::Int(1)]
            },
            Predicate::In {
                key: "a".to_string(),
                values: vec![crate::query::ir::PredicateValue::Int(2)]
            }
        );

        // Contains mismatches
        assert_ne!(
            Predicate::contains("a", "abc"),
            Predicate::contains("b", "abc")
        );
        assert_ne!(
            Predicate::contains("a", "abc"),
            Predicate::contains("a", "xyz")
        );

        // StartsWith mismatches
        assert_ne!(
            Predicate::StartsWith {
                key: "a".to_string(),
                prefix: "abc".to_string()
            },
            Predicate::StartsWith {
                key: "b".to_string(),
                prefix: "abc".to_string()
            }
        );
        assert_ne!(
            Predicate::StartsWith {
                key: "a".to_string(),
                prefix: "abc".to_string()
            },
            Predicate::StartsWith {
                key: "a".to_string(),
                prefix: "xyz".to_string()
            }
        );

        // EndsWith mismatches
        assert_ne!(
            Predicate::EndsWith {
                key: "a".to_string(),
                suffix: "abc".to_string()
            },
            Predicate::EndsWith {
                key: "b".to_string(),
                suffix: "abc".to_string()
            }
        );
        assert_ne!(
            Predicate::EndsWith {
                key: "a".to_string(),
                suffix: "abc".to_string()
            },
            Predicate::EndsWith {
                key: "a".to_string(),
                suffix: "xyz".to_string()
            }
        );

        // Exists mismatches
        assert_ne!(Predicate::exists("a"), Predicate::exists("b"));

        // NotExists mismatches
        assert_ne!(
            Predicate::NotExists("a".to_string()),
            Predicate::NotExists("b".to_string())
        );

        // And / Or structural mismatches (different sizes)
        assert_ne!(
            Predicate::And(vec![Predicate::eq("a", 1)]),
            Predicate::And(vec![Predicate::eq("a", 1), Predicate::eq("b", 2)])
        );
        assert_ne!(
            Predicate::Or(vec![Predicate::eq("a", 1)]),
            Predicate::Or(vec![Predicate::eq("a", 1), Predicate::eq("b", 2)])
        );
    }
}
