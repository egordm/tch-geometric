use std::ops::{Range};
use num_traits::Float;
use rand::distributions::uniform::{SampleUniform};
use rand::Rng;

pub fn reservoir_sampling<T: Copy, I: Iterator<Item=T>>(
    rng: &mut impl Rng,
    mut src: I,
    dst: &mut [I::Item]
) {
    for dst_val in dst.iter_mut() {
        *dst_val = src.next().unwrap();
    }

    for (i, v) in src.enumerate() {
        let j = rng.gen_range(0..i);
        if j < dst.len() {
            dst[j] = v;
        }
    }
}

pub fn reservoir_sampling_weighted<
    T: Copy, W: Float + SampleUniform, I: Iterator<Item=(T, W)>
>(
    rng: &mut impl Rng,
    mut src: I,
    dst: &mut [T]
) {
    let mut w_sum = W::zero();
    for dst_v in dst.iter_mut() {
        let (v, w) = src.next().unwrap();
        *dst_v = v;
        w_sum = w_sum + w;
    }

    for (v, w) in src {
        w_sum = w_sum + w;
        let j = rng.gen_range(W::zero()..w_sum);
        if j < w {
            dst[rng.gen_range(0..dst.len())] = v;
        }
    }
}

pub fn replacement_sampling<T: Copy>(
    rng: &mut impl Rng,
    src: &[T],
    dst: &mut [T]
) {
    for dst_val in dst.iter_mut() {
        let j = rng.gen_range(0..src.len());
        *dst_val = src[j];
    }
}

pub fn replacement_sampling_range<T: Copy + SampleUniform + PartialOrd>(
    rng: &mut impl Rng,
    src: &Range<T>,
    dst: &mut [T]
) {
    for dst_val in dst.iter_mut() {
        *dst_val = rng.gen_range(src.clone());
    }
}
