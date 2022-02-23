use std::ops::{Add, Range, Sub};
use rand::distributions::uniform::{SampleRange, SampleUniform};
use rand::Rng;

pub trait Sampling {
    type Item;

    fn sample(
        &self,
        rng: &mut impl Rng,
        dst: &mut [Self::Item],
    );
}

impl<T: Copy> Sampling for &[T] {
    type Item = T;

    fn sample(&self, rng: &mut impl Rng, dst: &mut [Self::Item]) {
        // Implementation of simple reservoir sampling
        dst.copy_from_slice(&self[0..dst.len()]);

        for i in dst.len()..self.len() {
            let j = rng.gen_range(0..i);
            if j < dst.len() {
                dst[j] = self[i];
            }
        }
    }
}

impl<T: Copy + Add<Output=T> + Sub<Output=T> + From<usize> + Into<usize>> Sampling for Range<T> {
    type Item = T;

    fn sample(&self, rng: &mut impl Rng, dst: &mut [Self::Item]) {
        let src_len = (self.end - self.start).into();

        for i in 0..dst.len() {
            dst[i] = self.start + T::from(i);
        }

        for i in dst.len()..src_len {
            let j = rng.gen_range(0..i);
            if j < dst.len() {
                dst[j] = self.start + T::from(i);
            }
        }
    }
}

pub trait ReplacementSampling {
    type Item;

    fn replacement_sample(
        &self,
        rng: &mut impl Rng,
        dst: &mut [Self::Item],
    );
}

impl<T: Copy> ReplacementSampling for &[T] {
    type Item = T;

    fn replacement_sample(&self, rng: &mut impl Rng, dst: &mut [Self::Item]) {
        for i in 0..dst.len() {
            let j = rng.gen_range(0..self.len());
            dst[i] = self[j];
        }
    }
}

impl<T: SampleUniform + PartialOrd + Copy> ReplacementSampling for Range<T> {
    type Item = T;

    fn replacement_sample(&self, rng: &mut impl Rng, dst: &mut [Self::Item]) {
        for i in 0..dst.len() {
            dst[i] = rng.gen_range(self.clone());
        }
    }
}