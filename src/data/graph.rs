use std::ops::Range;
use crate::utils::types::{DefaultIx, DefaultPtr, IndexType, NodeIdx, NodePtr};

#[derive(Debug, Clone, Copy)]
pub enum SparseGraphType {
    Csr,
    Csc,
}

pub trait SparseGraphTypeTrait {
    fn get_type() -> SparseGraphType;
}

pub struct Csr;

impl SparseGraphTypeTrait for Csr {
    fn get_type() -> SparseGraphType {
        SparseGraphType::Csr
    }
}

pub struct Csc;

impl SparseGraphTypeTrait for Csc {
    fn get_type() -> SparseGraphType {
        SparseGraphType::Csc
    }
}

#[derive(Debug)]
pub struct SparseGraph<'a, Ty, Ptr = DefaultPtr, Ix = DefaultIx> {
    pub ptrs: &'a [NodePtr<Ptr>],
    pub indices: &'a [NodeIdx<Ix>],
    pub _phantom: std::marker::PhantomData<Ty>,
}

pub type CsrGraph<'a, Ptr = DefaultPtr, Ix = DefaultIx> = SparseGraph<'a, Csr, Ptr, Ix>;
pub type CscGraph<'a, Ptr = DefaultPtr, Ix = DefaultIx> = SparseGraph<'a, Csc, Ptr, Ix>;

impl<'a, Ty, Ptr: IndexType, Ix: IndexType> SparseGraph<'a, Ty, Ptr, Ix> {
    pub fn new(ptrs: &'a [NodePtr<Ptr>], indices: &'a [NodeIdx<Ix>]) -> Self {
        SparseGraph {
            ptrs,
            indices,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn node_count(&self) -> usize {
        self.ptrs.len() - 1
    }

    pub fn edge_count(&self) -> usize {
        self.indices.len()
    }

    #[inline(always)]
    fn edge_offset(&self, node: NodeIdx<Ix>) -> NodePtr<Ptr> {
        self.ptrs[node.index()]
    }

    pub fn neighbors_range(&self, x: NodeIdx<Ix>) -> Range<usize> {
        let node = x.index();
        let start = self.ptrs[node].index();
        let end = self.ptrs[node + 1].index();
        start..end
    }

    pub fn neighbors_slice(&self, x: NodeIdx<Ix>) -> &[NodeIdx<Ix>] {
        &self.indices[self.neighbors_range(x)]
    }
}

impl<'a, Ptr: IndexType, Ix: IndexType> CscGraph<'a, Ptr, Ix> {
    pub fn in_degree(&self, x: NodeIdx<Ix>) -> usize {
        self.neighbors_range(x).len()
    }
}

impl<'a, Ptr: IndexType, Ix: IndexType> CsrGraph<'a, Ptr, Ix> {
    pub fn out_degree(&self, x: NodeIdx<Ix>) -> usize {
        self.neighbors_range(x).len()
    }
}


