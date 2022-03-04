use std::collections::HashMap;
use std::hash::Hash;

pub fn hashmap_from<K, V>(keys: impl Iterator<Item=K>, vals: impl Fn(&K) -> V) -> HashMap<String, V>
    where K: Eq + Hash + ToString
{
    keys.map(|k| (k.to_string(), vals(&k))).collect()
}