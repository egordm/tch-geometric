use std::ops::{Range};
use num_traits::Float;
use rand::distributions::uniform::{SampleUniform};
use rand::Rng;

pub fn reservoir_sampling<T: Copy, I: Iterator<Item=T>>(
    rng: &mut impl Rng,
    mut src: I,
    dst: &mut [I::Item]
) -> usize {
    let mut n = 0;
    for (dst_val,  src_val) in dst.iter_mut().zip(src.by_ref()) {
        *dst_val = src_val;
        n += 1;
    }

    let mut i = n;
    for v in src {
        let j = rng.gen_range(0..i);
        if j < dst.len() {
            dst[j] = v;
        }
        i += 1;
    }
    n
}

pub fn reservoir_sampling_weighted<
    T: Copy, W: Float + SampleUniform, I: Iterator<Item=(T, W)>
>(
    rng: &mut impl Rng,
    mut src: I,
    dst: &mut [T]
) -> usize {
    let mut n = 0;
    let mut w_sum = W::zero();
    for dst_v in dst.iter_mut() {
        if let Some((v, w)) = src.next() {
            *dst_v = v;
            w_sum = w_sum + w;
            n += 1;
        } else {
            break;
        }
    }

    for (v, w) in src {
        w_sum = w_sum + w;
        let j = rng.gen_range(W::zero()..w_sum);
        if j < w {
            dst[rng.gen_range(0..dst.len())] = v;
        }
    }
    n
}

pub fn replacement_sampling<T: Copy>(
    rng: &mut impl Rng,
    src: &[T],
    dst: &mut [T]
) -> usize {
    let mut n = 0;
    for dst_val in dst.iter_mut() {
        let j = rng.gen_range(0..src.len());
        *dst_val = src[j];
        n += 1
    }
    n
}

pub fn replacement_sampling_range<T: Copy + SampleUniform + PartialOrd>(
    rng: &mut impl Rng,
    src: &Range<T>,
    dst: &mut [T]
) -> usize {
    let mut n = 0;
    for dst_val in dst.iter_mut() {
        *dst_val = rng.gen_range(src.clone());
        n += 1
    }
    n
}
