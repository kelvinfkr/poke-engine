//! A learned leaf for the search, on features the engine already has.
//!
//! `evaluate()` is a hand-weighted sum that cannot see what accumulates (ADR
//! 0014, 0020). The plan's Task 2.2 wants a value head there, but a Python
//! callback per leaf is far too slow at 20k iterations and the Python state
//! encoder reads poke-env objects and beliefs the engine never sees. So the net
//! lives here, on a feature vector computed straight from `State`, and the same
//! `features()` is exported to Python so training rows are exactly what the leaf
//! will see. Inference is a hand-written MLP forward: no ort, no allocation
//! beyond two small buffers.
//!
//! The leaf value is a blend, `(1 - alpha) * static + alpha * NET_SCALE * tanh(net)`,
//! so alpha = 0 is exactly the old search and the A/B has a real control arm.

use crate::state::{PokemonStatus, State};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::RwLock;

pub const FEATURE_COUNT: usize = 2 * SIDE_FEATURES + GLOBAL_FEATURES;
const SLOT_FEATURES: usize = 1 + 1 + 7 + 1; // alive, hp fraction, status one-hot, is active
const SIDE_FEATURES: usize = 6 * SLOT_FEATURES + 7 + 1 + 9 + 2 + 1 + 20;
const GLOBAL_FEATURES: usize = 8 + 1 + 6 + 1 + 1 + 1;
/// The static leaf is on a scale where a full-HP Pokémon with an item is ~140
/// points; the net outputs tanh in [-1, 1] and is mapped onto that scale.
pub const NET_SCALE: f32 = 300.0;

static BLEND_BITS: AtomicU32 = AtomicU32::new(0);
static MODEL: RwLock<Option<ValueNet>> = RwLock::new(None);

#[derive(Clone, Debug)]
pub struct ValueNet {
    /// weights[l] is out_l x in_l, row-major; biases[l] is out_l
    pub weights: Vec<Vec<f32>>,
    pub biases: Vec<Vec<f32>>,
    pub shapes: Vec<(usize, usize)>,
}

impl ValueNet {
    pub fn forward(&self, input: &[f32]) -> f32 {
        let mut x: Vec<f32> = input.to_vec();
        let last = self.shapes.len() - 1;
        for (l, &(out_n, in_n)) in self.shapes.iter().enumerate() {
            let w = &self.weights[l];
            let b = &self.biases[l];
            let mut y = vec![0.0f32; out_n];
            for o in 0..out_n {
                let row = &w[o * in_n..(o + 1) * in_n];
                let mut acc = b[o];
                for i in 0..in_n.min(x.len()) {
                    acc += row[i] * x[i];
                }
                // GELU-free on purpose: ReLU keeps the forward pass trivially portable
                y[o] = if l == last { acc } else { acc.max(0.0) };
            }
            x = y;
        }
        x[0].tanh()
    }
}

pub fn set_model(model: Option<ValueNet>) {
    *MODEL.write().unwrap() = model;
}

pub fn set_blend(alpha: f32) {
    BLEND_BITS.store(alpha.to_bits(), Ordering::Relaxed);
}

pub fn blend() -> f32 {
    f32::from_bits(BLEND_BITS.load(Ordering::Relaxed))
}

/// The leaf the search should use: the static evaluation, pulled toward the
/// net by `blend()` when a model is loaded.
pub fn evaluate_leaf(state: &State, static_value: f32) -> f32 {
    let alpha = blend();
    if alpha <= 0.0 {
        return static_value;
    }
    let guard = MODEL.read().unwrap();
    match guard.as_ref() {
        None => static_value,
        Some(net) => {
            let f = features(state);
            (1.0 - alpha) * static_value + alpha * NET_SCALE * net.forward(&f)
        }
    }
}

fn status_index(status: &PokemonStatus) -> usize {
    match status {
        PokemonStatus::NONE => 0,
        PokemonStatus::BURN => 1,
        PokemonStatus::SLEEP => 2,
        PokemonStatus::FREEZE => 3,
        PokemonStatus::PARALYZE => 4,
        PokemonStatus::POISON => 5,
        PokemonStatus::TOXIC => 6,
    }
}

fn boost(v: i8) -> f32 {
    v as f32 / 6.0
}

fn side_features(out: &mut Vec<f32>, side: &crate::state::Side) {
    let active_index = side.active_index;
    let mut alive = 0.0;
    let mut iter = side.pokemon.into_iter();
    let mut slots = 0;
    while let Some(p) = iter.next() {
        if slots >= 6 {
            break;
        }
        let is_alive = if p.hp > 0 { 1.0 } else { 0.0 };
        alive += is_alive;
        out.push(is_alive);
        out.push(if p.maxhp > 0 {
            p.hp as f32 / p.maxhp as f32
        } else {
            0.0
        });
        let mut status = [0.0f32; 7];
        status[status_index(&p.status)] = 1.0;
        out.extend_from_slice(&status);
        out.push(if iter.pokemon_index == active_index {
            1.0
        } else {
            0.0
        });
        slots += 1;
    }
    while slots < 6 {
        out.extend_from_slice(&[0.0; SLOT_FEATURES]);
        slots += 1;
    }
    out.push(boost(side.attack_boost));
    out.push(boost(side.defense_boost));
    out.push(boost(side.special_attack_boost));
    out.push(boost(side.special_defense_boost));
    out.push(boost(side.speed_boost));
    out.push(boost(side.accuracy_boost));
    out.push(boost(side.evasion_boost));
    let active = side.get_active_immutable();
    out.push(if active.maxhp > 0 {
        side.substitute_health as f32 / active.maxhp as f32
    } else {
        0.0
    });
    let sc = &side.side_conditions;
    out.push((sc.reflect as f32).min(5.0) / 5.0);
    out.push((sc.light_screen as f32).min(5.0) / 5.0);
    out.push((sc.aurora_veil as f32).min(5.0) / 5.0);
    out.push((sc.tailwind as f32).min(4.0) / 4.0);
    out.push(sc.stealth_rock as f32);
    out.push((sc.spikes as f32) / 3.0);
    out.push((sc.toxic_spikes as f32) / 2.0);
    out.push(sc.sticky_web as f32);
    out.push((sc.toxic_count as f32).min(16.0) / 16.0);
    // PP: how much of the active's moveset is left, and its emptiest slot
    let mut total_pp = 0.0;
    let mut min_pp = 1.0f32;
    let mut n_moves = 0.0;
    for m in active.moves.into_iter() {
        let frac = (m.pp.max(0) as f32 / 16.0).min(1.0);
        total_pp += frac;
        min_pp = min_pp.min(frac);
        n_moves += 1.0;
    }
    out.push(if n_moves > 0.0 {
        total_pp / n_moves
    } else {
        0.0
    });
    out.push(if n_moves > 0.0 { min_pp } else { 0.0 });
    out.push(alive / 3.0);
    let mut types = [0.0f32; 20];
    for t in [active.types.0, active.types.1] {
        let i = t as usize;
        if i < 20 {
            types[i] = 1.0;
        }
    }
    out.extend_from_slice(&types);
}

/// The engine-native feature vector, side one first. Length is `FEATURE_COUNT`.
pub fn features(state: &State) -> Vec<f32> {
    let mut out = Vec::with_capacity(FEATURE_COUNT);
    side_features(&mut out, &state.side_one);
    side_features(&mut out, &state.side_two);
    let mut weather = [0.0f32; 8];
    let wi = state.weather.weather_type as usize;
    if wi < 8 {
        weather[wi] = 1.0;
    }
    out.extend_from_slice(&weather);
    out.push((state.weather.turns_remaining.max(0) as f32).min(8.0) / 8.0);
    let mut terrain = [0.0f32; 6];
    let ti = state.terrain.terrain_type as usize;
    if ti < 6 {
        terrain[ti] = 1.0;
    }
    out.extend_from_slice(&terrain);
    out.push((state.terrain.turns_remaining.max(0) as f32).min(8.0) / 8.0);
    out.push(if state.trick_room.active { 1.0 } else { 0.0 });
    out.push((state.trick_room.turns_remaining.max(0) as f32).min(5.0) / 5.0);
    debug_assert_eq!(out.len(), FEATURE_COUNT);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn features_have_the_declared_width_and_no_model_means_the_static_leaf() {
        let state = State::default();
        assert_eq!(features(&state).len(), FEATURE_COUNT);
        set_model(None);
        set_blend(0.7);
        assert_eq!(evaluate_leaf(&state, 12.5), 12.5);
        set_blend(0.0);
    }

    #[test]
    fn a_loaded_net_moves_the_leaf_by_alpha() {
        // one layer, all-zero weights, bias 1 -> tanh(1) after the last layer
        let net = ValueNet {
            weights: vec![vec![0.0; FEATURE_COUNT]],
            biases: vec![vec![1.0]],
            shapes: vec![(1, FEATURE_COUNT)],
        };
        set_model(Some(net));
        set_blend(0.5);
        let v = evaluate_leaf(&State::default(), 10.0);
        let expected = 0.5 * 10.0 + 0.5 * NET_SCALE * 1.0f32.tanh();
        assert!((v - expected).abs() < 1e-3, "{v} vs {expected}");
        set_blend(0.0);
        set_model(None);
    }
}
