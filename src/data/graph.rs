use std::ops::Range;
use tch::kind::Element;
use tch::Tensor;
use crate::utils::{EdgeIdx, EdgePtr};
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
    fn edge_offset(&self, node: NodeIdx<Ix>) -> EdgePtr<Ptr> {
        self.ptrs[node.index()]
    }

    pub fn neighbors_range(&self, x: NodeIdx<Ix>) -> Range<EdgePtr<usize>> {
        let node = x.index();
        let start = self.ptrs[node].index();
        let end = self.ptrs[node + 1].index();
        start..end
    }

    pub fn neighbors_slice(&self, x: NodeIdx<Ix>) -> &[NodeIdx<Ix>] {
        &self.indices[self.neighbors_range(x)]
    }

    pub fn get_by_ptr(&self, ptr: EdgePtr<usize>) -> NodeIdx<Ix> {
        self.indices[ptr]
    }

    pub fn has_edge(&self, x: NodeIdx<Ix>, y: NodeIdx<Ix>) -> bool {
        let neighbors = self.neighbors_slice(x);
        neighbors.binary_search(&y).is_ok()
    }
    
    pub fn find_edge(&self, x: NodeIdx<Ix>, y: NodeIdx<Ix>) -> Option<EdgePtr<usize>> {
        let neighbors = self.neighbors_slice(x);
        neighbors.binary_search(&y).ok()
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

#[derive(Debug, Clone)]
pub struct EdgeAttr<'a, T> {
    pub data: &'a [T],
}

impl<'a, T> EdgeAttr<'a, T> {
    pub fn new(data: &'a [T]) -> Self {
        EdgeAttr { data }
    }

    pub fn get(&self, edge: EdgePtr<usize>) -> &T {
        &self.data[edge]
    }

    pub fn get_range(&self, edge_range: Range<EdgePtr<usize>>) -> &[T] {
        &self.data[edge_range]
    }
}


pub struct CooGraphBuilder<Ix = DefaultIx, Ptr = DefaultPtr> {
    pub rows: Vec<NodeIdx<Ix>>,
    pub cols: Vec<NodeIdx<Ix>>,
    pub edge_index: Vec<EdgeIdx<Ptr>>,
}

impl<Ix: IndexType + Element, Ptr: IndexType + Element> CooGraphBuilder<Ix, Ptr> {
    pub fn new() -> Self {
        CooGraphBuilder {
            cols: Vec::new(),
            rows: Vec::new(),
            edge_index: Vec::new(),
        }
    }

    pub fn push_edge(&mut self, src: NodeIdx<Ix>, dst: NodeIdx<Ix>, edge_index: NodePtr<Ptr>) {
        self.rows.push(src);
        self.cols.push(dst);
        self.edge_index.push(edge_index);
    }

    pub fn to_tensor(&self) -> (Tensor, Tensor, Tensor) {
        let cols = Tensor::of_slice(&self.cols);
        let rows = Tensor::of_slice(&self.rows);
        let edge_index = Tensor::of_slice(&self.edge_index);
        (cols, rows, edge_index)
    }
    
    pub fn len(&self) -> usize {
        self.cols.len()
    }
}


