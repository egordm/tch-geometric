use std::cell::RefCell;
use std::ops::DerefMut;
use std::sync::Mutex;
use lazy_static::lazy_static;
use rand::rngs::SmallRng;
use rand::{RngCore, SeedableRng};

lazy_static! {
        static ref RNG: Mutex<RefCell<SmallRng>> = {
            Mutex::new(RefCell::new(SmallRng::from_entropy()))
        };
    }

pub fn rng_reseed(seed: [u8; 32]) {
    let rng = RNG.lock().unwrap();
    rng.replace(SmallRng::from_seed(seed));
}

pub fn rng_get() -> SmallRng {
    let guard = RNG.lock().unwrap();
    let mut rng = guard.borrow_mut();
    SmallRng::from_rng(rng.deref_mut()).unwrap()
}