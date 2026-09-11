//! Zero-Allocation Traversal Iterators
//!
//! This module provides high-performance iterators for traversing graph edges
//! without allocating intermediate memory.
//!
//! # The "Zero-Allocation Traversal" Philosophy
//!
//! Graph traversal often requires fetching the neighbors of a node. Naive implementations
//! allocate a `Vec<EdgeId>` to return these neighbors, which introduces significant overhead
//! (100-500ns per traversal) in hot paths, especially for deep graph queries.
//!
//! AletheiaDB avoids this by returning lazy iterators. These iterators hold a lock-free reference
//! ([`MergedAdjacencyGuard`]) to the underlying
//! adjacency data (frozen CSR slice + delta map) and yield edge IDs on demand.
//!
//! This approach provides:
//! - **Zero allocation**: No `Vec` is created during traversal.
//! - **Lazy evaluation**: Only the necessary edges are processed, which is crucial for `LIMIT` queries.
//! - **Cache-friendly**: Edge IDs are read sequentially from contiguous memory (the CSR frozen state).
//!
//! The iterators defined here are generated via macros to ensure consistent behavior
//! across different traversal directions (outgoing vs. incoming) and filtering conditions (with/without labels).

use crate::core::id::EdgeId;
use crate::core::interning::InternedString;
use crate::index::incremental_adjacency::MergedAdjacencyGuard;

/// Macro to generate edge iterator types.
///
/// This eliminates code duplication between outgoing/incoming iterator variants
/// while preserving distinct types for API clarity.
macro_rules! impl_edge_iter {
    (
        $(#[$meta:meta])*
        $name:ident,
        $method_link:literal,
        $direction:literal
    ) => {
        $(#[$meta])*
        #[doc = concat!(
            "\n\nThis iterator provides zero-allocation traversal of ", $direction, " edges,",
            "\navoiding the `Vec` allocation overhead of [`", $method_link, "`](crate::storage::CurrentStorage::", $method_link, ").",
            "\n\n# Performance",
            "\n\n- **Zero allocation**: Holds a `MergedAdjacencyGuard` (merges frozen + delta)",
            "\n- **Lazy evaluation**: Only computes results as you iterate",
            "\n- **Cache-friendly**: Sequential access to CSR adjacency data",
            "\n\n# Issue #187",
            "\n\nThis iterator addresses the performance issue where edge traversal methods",
            "\nunnecessarily allocated a new `Vec<EdgeId>` on every call, adding 100-500ns",
            "\noverhead per traversal."
        )]
        pub struct $name<'a> {
            guard: MergedAdjacencyGuard<'a>,
            /// Bounds of this node's run in `guard.frozen_entries()`, resolved
            /// once at construction (Issue #3813).
            ///
            /// `MergedAdjacencyGuard::frozen_slice()` is an O(log V) binary
            /// search, so calling it per `next()` made a degree-`d` walk cost
            /// O(d * log V). The guard pins the frozen CSR, so these bounds
            /// are stable for the iterator's lifetime.
            frozen_start: usize,
            frozen_end: usize,
            index: usize,
        }

        impl<'a> $name<'a> {
            #[doc = concat!("Create a new iterator over ", $direction, " edges.")]
            #[inline]
            pub(crate) fn new(guard: MergedAdjacencyGuard<'a>) -> Self {
                let (frozen_start, frozen_end) = guard.frozen_range();
                Self {
                    guard,
                    frozen_start,
                    frozen_end,
                    index: 0,
                }
            }
        }

        impl<'a> Iterator for $name<'a> {
            type Item = EdgeId;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                // O(1) slice re-derivation from the bounds resolved at
                // construction -- no binary search per element (Issue #3813).
                // MergedAdjacencyGuard::get() is O(n), so calling it in a loop
                // would be O(n^2); frozen_slice() is O(log V) per call, which
                // was the ~3.8x slowdown vs the Vec path.
                let frozen_slice =
                    &self.guard.frozen_entries()[self.frozen_start..self.frozen_end];
                let delta_slice = self.guard.delta_slice();

                while self.index < frozen_slice.len() + delta_slice.len() {
                    let from_delta = self.index >= frozen_slice.len();
                    let entry = if from_delta {
                        &delta_slice[self.index - frozen_slice.len()]
                    } else {
                        &frozen_slice[self.index]
                    };

                    self.index += 1;

                    // Filter tombstones
                    if self.guard.is_tombstoned(entry.edge_id) {
                        continue;
                    }
                    // Filter the delta half against the frozen slice while a
                    // compaction publish window is open, where an entry is
                    // momentarily in both layers (Issue #3810). This walks the
                    // layers directly instead of through `guard.iter()`, so it
                    // has to apply the same de-duplication that does.
                    if from_delta
                        && self
                            .guard
                            .delta_entry_is_duplicate_in(frozen_slice, entry)
                    {
                        continue;
                    }
                    return Some(entry.edge_id);
                }
                None
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                if let Some(len) = self.guard.fast_len() {
                    let remaining = len.saturating_sub(self.index);
                    (remaining, Some(remaining))
                } else {
                    let upper = self.guard.capacity_hint().saturating_sub(self.index);
                    (0, Some(upper))
                }
            }
        }

        impl<'a> ExactSizeIterator for $name<'a> {
            #[inline]
            fn len(&self) -> usize {
                // Note: O(n) unless in fast path.
                if let Some(len) = self.guard.fast_len() {
                    len.saturating_sub(self.index)
                } else {
                    self.guard.len().saturating_sub(self.index)
                }
            }
        }

        impl<'a> std::iter::FusedIterator for $name<'a> {}
    };
}

/// Macro to generate label-filtered edge iterator types.
macro_rules! impl_edge_iter_with_label {
    (
        $(#[$meta:meta])*
        $name:ident,
        $method_link:literal,
        $direction:literal
    ) => {
        $(#[$meta])*
        #[doc = concat!(
            "\n\nThis iterator provides zero-allocation traversal of ", $direction, " edges",
            "\nthat match a specific label, avoiding the `Vec` allocation overhead of",
            "\n[`", $method_link, "`](crate::storage::CurrentStorage::", $method_link, ").",
            "\n\n# Performance",
            "\n\n- **Zero allocation**: Holds a `MergedAdjacencyGuard` and pre-resolved label ID",
            "\n- **Lazy evaluation**: Filters as you iterate",
            "\n- **O(n) complexity**: Single linear scan regardless of match distribution",
            "\n- **Early termination**: If label doesn't exist, returns empty iterator"
        )]
        pub struct $name<'a> {
            guard: MergedAdjacencyGuard<'a>,
            /// Bounds of this node's run in `guard.frozen_entries()`, resolved
            /// once at construction (Issue #3813) -- see the unlabeled
            /// iterator above for why this must not be a per-`next()`
            /// `frozen_slice()` call.
            frozen_start: usize,
            frozen_end: usize,
            index: usize,
            label_id: Option<InternedString>,
        }

        impl<'a> $name<'a> {
            #[doc = concat!("Create a new iterator over ", $direction, " edges with a specific label.")]
            #[inline]
            pub(crate) fn new(guard: MergedAdjacencyGuard<'a>, label_id: Option<InternedString>) -> Self {
                let (frozen_start, frozen_end) = guard.frozen_range();
                Self {
                    guard,
                    frozen_start,
                    frozen_end,
                    index: 0,
                    label_id,
                }
            }
        }

        impl<'a> Iterator for $name<'a> {
            type Item = EdgeId;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                // Early return if label doesn't exist in interner
                let label_id = self.label_id?;

                // Linear scan with manual indexing - O(n) total complexity.
                // We access frozen and delta slices directly to avoid O(n^2) indexing
                // through the MergedAdjacencyGuard. The frozen bounds were
                // resolved once at construction (Issue #3813) instead of
                // re-running the O(log V) `frozen_slice()` lookup per element.
                let frozen_slice =
                    &self.guard.frozen_entries()[self.frozen_start..self.frozen_end];
                let delta_slice = self.guard.delta_slice();

                while self.index < frozen_slice.len() + delta_slice.len() {
                    let from_delta = self.index >= frozen_slice.len();
                    let entry = if from_delta {
                        &delta_slice[self.index - frozen_slice.len()]
                    } else {
                        &frozen_slice[self.index]
                    };

                    self.index += 1;

                    // Filter by label AND tombstones, and de-duplicate the
                    // delta half against the frozen slice while a compaction
                    // publish window is open (Issue #3810) -- see the unlabeled
                    // iterator above.
                    if entry.label == label_id
                        && !self.guard.is_tombstoned(entry.edge_id)
                        && !(from_delta
                            && self
                                .guard
                                .delta_entry_is_duplicate_in(frozen_slice, entry))
                    {
                        return Some(entry.edge_id);
                    }
                }
                None
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                // Lower bound is 0 (all remaining could be filtered out)
                // Upper bound is remaining entries
                let remaining = self.guard.capacity_hint().saturating_sub(self.index);
                (0, Some(remaining))
            }
        }

        impl<'a> std::iter::FusedIterator for $name<'a> {}
    };
}

impl_edge_iter!(
    /// Iterator over outgoing edge IDs from a node.
    OutgoingEdgesIter,
    "get_outgoing_edges",
    "outgoing"
);

impl_edge_iter!(
    /// Iterator over incoming edge IDs to a node.
    IncomingEdgesIter,
    "get_incoming_edges",
    "incoming"
);

impl_edge_iter_with_label!(
    /// Iterator over outgoing edge IDs from a node, filtered by label.
    OutgoingEdgesWithLabelIter,
    "get_outgoing_edges_with_label",
    "outgoing"
);

impl_edge_iter_with_label!(
    /// Iterator over incoming edge IDs to a node, filtered by label.
    IncomingEdgesWithLabelIter,
    "get_incoming_edges_with_label",
    "incoming"
);
