use std::fmt;
use std::hash::Hash;

pub type DefaultPtr = i64;
pub type DefaultIx = i64;
pub type NodeIdx<Ix = DefaultIx> = Ix;
pub type NodePtr<Ix = DefaultIx> = Ix;
pub type EdgeIdx<Ix = DefaultIx> = Ix;
pub type EdgePtr<Ix = DefaultIx> = Ix;

pub unsafe trait IndexType: Copy + Default + Hash + Ord + fmt::Debug + 'static {
    fn new(x: usize) -> Self;
    fn index(&self) -> usize;
    fn max() -> Self;
}

unsafe impl IndexType for i64 {
    #[inline(always)]
    fn new(x: usize) -> Self {
        x as i64
    }
    #[inline(always)]
    fn index(&self) -> usize {
        *self as usize
    }
    #[inline(always)]
    fn max() -> Self {
        i64::MAX
    }
}

// pub fn