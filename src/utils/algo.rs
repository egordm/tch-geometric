use std::fmt::Debug;
use std::ops::{Index, IndexMut};
use std::slice::SliceIndex;

#[derive(Debug)]
pub struct OrdHeap<'a, T: Debug> {
    pub data: &'a mut [T],
    pub size: usize,
}

const HEAD: usize = 1;

impl<'a, T: Debug + Copy> OrdHeap<'a, T> {
    pub fn new(data: &'a mut [T]) -> Self {
        let size = data.len() - 1;
        OrdHeap { data, size }
    }

    pub fn head(&self) -> Option<&T> {
        if self.size == 0 {
            None
        } else {
            Some(&self.unchecked_head())
        }
    }

    pub fn unchecked_head(&self) -> &T {
        &self.data[HEAD]
    }

    pub fn unchecked_head_mut(&mut self) -> &mut T {
        &mut self.data[HEAD]
    }

    fn left_child(&self, i: usize) -> usize {
        i << 1
    }

    fn right_child(&self, i: usize) -> usize {
        (i << 1) + 1
    }

    fn parent(&self, i: usize) -> usize {
        i >> 1
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.data[0] = self.data[i];
        self.data[i] = self.data[j];
        self.data[j] = self.data[0];
    }

    pub fn data(&self) -> &[T] {
        &self.data[1..self.size + 1]
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data[1..self.size + 1]
    }

    pub fn pop_by<
        F: Fn(&T, &T) -> bool
    >(&mut self, f: F) -> T {
        let res = self.data[1];
        self.data[1] = self.data[self.size];
        self.size -= 1;
        self.sift_down_by(HEAD, f);
        res
    }

    pub fn pop_by_value<
        W: PartialOrd, F: Fn(&T) -> W
    >(&mut self, f: F) -> T {
        self.pop_by(|a, b| f(a) < f(b))
    }

    pub fn push_by<
        F: Fn(&T, &T) -> bool
    >(&mut self, value: T, f: F) -> Option<()> {
        if self.size + 1 == self.data.len() {
            None
        } else {
            self.size += 1;
            self.data[self.size] = value;
            self.sift_up_by(self.size, f);
            Some(())
        }
    }

    pub fn sift_down_by<
        F: Fn(&T, &T) -> bool
    >(&mut self, mut i: usize, f: F) {
        let mut j = i;
        let (l, r) = (self.left_child(i), self.right_child(i));
        if l <= self.size && f(&self.data[l], &self.data[j]) { j = l }
        if r <= self.size && f(&self.data[r], &self.data[j]) { j = r }
        if j != i {
            self.swap(i, j);
            self.sift_down_by(j, f);
        }
    }

    pub fn sift_down_by_value<
        W: PartialOrd, F: Fn(&T) -> W
    >(&mut self, mut i: usize, f: F) {
        self.sift_down_by(i, |a, b| f(a) < f(b))
    }

    pub fn sift_up_by<
        F: Fn(&T, &T) -> bool
    >(&mut self, mut i: usize, f: F) {
        let mut j = i;
        let mut p = self.parent(j);
        while j > 1 && f(&self.data[j], &self.data[p]) {
            self.swap(j, p);
            j = p;
            p = self.parent(j);
        }
    }

    pub fn sift_up_by_value<
        W: PartialOrd, F: Fn(&T) -> W
    >(&mut self, mut i: usize, f: F) {
        self.sift_up_by(i, |a, b| f(a) < f(b))
    }

    pub fn rebuild_by<
        F: Copy + Fn(&T, &T) -> bool
    >(&mut self, f: F) {
        let n = self.size / 2 + 1;
        for i in (1..=n).rev() {
            self.sift_down_by(i, f);
        }
    }

    pub fn rebuild_by_value<
        W: PartialOrd, F: Fn(&T) -> W
    >(&mut self, f: F) {
        self.rebuild_by(|a, b| f(a) < f(b))
    }
}

impl<'a, T: Debug + Copy + PartialOrd> OrdHeap<'a, T> {
    pub fn pop(&mut self) -> T {
        self.pop_by(|a, b| a < b)
    }

    pub fn push(&mut self, value: T) -> Option<()> {
        self.push_by(value, |a, b| a < b)
    }

    pub fn sift_down(&mut self, i: usize) {
        self.sift_down_by(i, |a, b| a < b);
    }

    pub fn sift_up(&mut self, i: usize) {
        self.sift_up_by(i, |a, b| a < b);
    }

    pub fn rebuild(&mut self) {
        self.rebuild_by(|a, b| a < b);
    }
}


#[cfg(test)]
mod tests {
    use crate::utils::algo::HEAD;
    use crate::utils::OrdHeap;

    #[test]
    fn test_minheap() {
        let mut initial_data = vec![-1, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
        let mut heap = OrdHeap::new(&mut initial_data);
        heap.rebuild();

        assert_eq!(heap.size, 10);
        assert_eq!(*heap.unchecked_head(), 0);

        assert_eq!(heap.pop(), 0);
        assert_eq!(heap.size, 9);

        heap.push(0);
        assert_eq!(heap.size, 10);
        assert_eq!(*heap.unchecked_head(), 0);

        *heap.unchecked_head_mut() = 3;
        heap.sift_down(HEAD);
        assert_eq!(heap.pop(), 1);
        assert_eq!(heap.pop(), 2);
        assert_eq!(heap.pop(), 3);
        assert_eq!(heap.pop(), 3);
    }
}