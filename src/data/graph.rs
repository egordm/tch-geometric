use std::ops::Range;
use crate::utils::types::{DefaultIx, IndexType, NodeIdx, NodeOffset};


#[derive(Debug)]
pub struct CscGraph<'a, O = DefaultIx, Ix = DefaultIx> {
    pub col_ptrs: &'a [NodeOffset<O>],
    pub row_data: &'a [NodeIdx<Ix>],
}

impl<'a, O: IndexType, Ix: IndexType> CscGraph<'a, O, Ix> {
    pub fn node_count(&self) -> usize {
        self.col_ptrs.len() - 1
    }

    pub fn edge_count(&self) -> usize {
        self.row_data.len()
    }

    #[inline(always)]
    fn edge_offset(&self, node: NodeIdx<Ix>) -> NodeOffset<O> {
        self.col_ptrs[node.index()]
    }

    pub fn neighbors_range(&self, x: NodeIdx<Ix>) -> Range<usize> {
        let node = x.index();
        let start = self.col_ptrs[node].index();
        let end = self.col_ptrs[node + 1].index();
        start..end
    }

    pub fn neighbors_slice(&self, x: NodeIdx<Ix>) -> &[NodeIdx<Ix>] {
        &self.row_data[self.neighbors_range(x)]
    }
}

