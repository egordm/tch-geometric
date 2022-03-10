use std::collections::HashMap;
use std::hash::Hash;

pub fn hashmap_from<K, V>(keys: impl Iterator<Item=K>, vals: impl Fn(&K) -> V) -> HashMap<String, V>
    where K: Eq + Hash + ToString
{
    keys.map(|k| (k.to_string(), vals(&k))).collect()
}


pub trait IndexOpt<T> {
    type Key;
    type Result;

    fn get(&self, k: &Self::Key) -> Self::Result;
}

impl <'a, T> IndexOpt<T> for Option<&'a [T]> {
    type Key = usize;
    type Result = Option<&'a T>;

    fn get(&self, k: &Self::Key) -> Self::Result {
        self.map(|v| &v[*k])
    }
}

