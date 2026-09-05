//! A win-condition term for the static evaluation: who is the only answer to what.
//!
//! Everything already in `evaluate` — HP, alive count, status, hazards, screens,
//! boosts — is a property of the position *now*. None of it can say "this
//! Pokemon is the only thing on my side that beats their Garchomp, so trading it
//! ends the game", which is what a 3v3 endgame is mostly made of and what
//! ADR 0038 measured as a third of our losses.
//!
//! The primitive is a 1v1 win matrix
//!
//! ```text
//! W[i][j] = P(my remaining i beats their remaining j, one on one, from the
//!            HP / status / boosts / items they have now)
//! ```
//!
//! estimated from expected turns-to-KO each way plus the speed order, and the
//! term is the *coverage* difference
//!
//! ```text
//! c_j = 1 - prod_i (1 - W[i][j])          how well I answer their j
//! d_i = 1 - prod_j W[i][j]                how well they answer my i
//! coverage = prod_j c_j - prod_i d_i      in [-1, 1]
//! ```
//!
//! Only the coverage half is built. ADR 0038 §3 measured the setup-projection
//! half as flatter across the branches a search compares (0.026–0.036 of its own
//! range against coverage's 0.068), a much weaker predictor of the result
//! (AUC 0.527 against 0.732) and the smaller part of the decision effect — while
//! roughly doubling the damage table. If only one half is built, build coverage.
//!
//! `WINCON_COVERAGE` is zero by default and the first thing `coverage_term` does
//! is check it, so an unweighted build runs not one line of this and is
//! byte-identical to the engine without it.
//!
//! # Why there is a table
//!
//! A leaf term is evaluated once per MCTS iteration — 20,000 to 1,000,000 times
//! a move — against a static leaf that costs about 3 µs. Computing the matrix
//! naively is ~72 `calculate_damage` calls, which lands past the value net's 16×
//! slowdown that was already fatal for self-play generation.
//!
//! What a damage roll depends on is species, stats, types, item, ability,
//! status, moves, screens and the field — and **not HP, and not boosts**, since
//! boosts rescale a roll analytically. HP and boosts are what move inside a
//! tree. So the rolls are computed once, at zero boosts, into a thread-local
//! table indexed by team slot, and a leaf only re-runs turns-to-KO (a ≤8-step
//! loop per pair) and the coverage arithmetic. A forme change (Mega), an item
//! being lost, or a status landing changes a Pokemon's fingerprint and
//! invalidates that Pokemon's row and column — a handful of times a game, not
//! per leaf.
//!
//! See `docs/decisions/0038-win-condition-leaf.md` in the bot repo, and
//! `bot/search/wincon.py`, which is the reference this mirrors.

use super::abilities::{
    ability_modify_attack_against, ability_modify_attack_being_used, Abilities,
};
use super::choice_effects::modify_choice;
use super::damage_calc::{calculate_damage, DamageRolls};
use super::items::{item_modify_attack_against, item_modify_attack_being_used, Items};
use crate::choices::{Choice, MoveCategory};
use crate::state::{
    Pokemon, PokemonIndex, PokemonStatus, Side, SideReference, State, VolatileStatusBitset,
};
use std::cell::RefCell;
use std::sync::atomic::{AtomicU32, Ordering};

/// The weight on the coverage term, in evaluation points. Zero by default:
/// this is a search preference, not a correctness fix, so it has to be measured
/// before it is switched on. `POKEMON_HP` is 100 and `POKEMON_ALIVE` is 30, and
/// ADR 0038 §3 puts "moves the leaf as much as everything else does, across the
/// same actions" at about 205 — so 50–100 is a real but not overwhelming nudge.
pub static WINCON_COVERAGE: AtomicU32 = AtomicU32::new(0);

pub fn wincon_coverage_weight() -> f32 {
    f32::from_bits(WINCON_COVERAGE.load(Ordering::Relaxed))
}

pub fn set_wincon_coverage_weight(points: f32) {
    WINCON_COVERAGE.store(points.to_bits(), Ordering::Relaxed);
}

/// A 1v1 nobody wins inside eight turns is a stall, not a win.
const TURN_CAP: usize = 8;
/// Turns-of-margin to a probability.
const LOGISTIC_K: f32 = 1.4;
/// Speed is worth exactly one turn of margin: at equal turn counts the faster
/// one lands its last hit first and wins outright.
const SPEED_WEIGHT: f32 = 1.0;
/// sqrt(12): a uniform roll on [lo, hi] has sd (hi - lo) / sqrt(12).
const SQRT12: f32 = 3.464_101_6;
const INV_SQRT2: f32 = 0.707_106_77;

const MAX_SLOTS: usize = 6;
const MAX_MOVES: usize = 4;

// ------------------------------------------------------------------- the table

#[derive(Clone, Copy)]
struct MoveRoll {
    /// minimum and maximum damage as a fraction of the defender's max HP, at
    /// zero boosts on both sides
    lo: f32,
    hi: f32,
    accuracy: f32,
    priority: i8,
    physical: bool,
}

impl MoveRoll {
    const NONE: MoveRoll = MoveRoll {
        lo: 0.0,
        hi: 0.0,
        accuracy: 1.0,
        priority: 0,
        physical: false,
    };
}

#[derive(Clone, Copy)]
struct PairHits {
    /// number of damaging moves stored
    n: u8,
    rolls: [MoveRoll; MAX_MOVES],
}

impl PairHits {
    const EMPTY: PairHits = PairHits {
        n: 0,
        rolls: [MoveRoll::NONE; MAX_MOVES],
    };
}

/// How many *variants* of one ordered pair are remembered at once.
///
/// The first version kept one entry per pair and threw it away whenever the
/// pair's fingerprint changed. That is correct, and it was slow: a burn, a
/// consumed berry or a Knock Off changes a fingerprint, and a tree visits burned
/// and unburned branches in whatever order the selection rule likes, so the one
/// entry was rebuilt on a large share of leaves — 60 of the 151 points of
/// overhead the first measurement showed. Eight ways covers the variants a
/// Pokemon has in one game (burned or not, item or not, Mega or not) and turns
/// the rebuild into a once-per-variant cost: measured on ten real ladder
/// positions, going from one entry to four took the pooled cost of the term from
/// +151% of an iteration to +51%, and four to eight took it to +24% — the
/// difference is entirely in the positions that thrashed, whose worst case fell
/// from +104% to +57%.
const WAYS: usize = 8;

#[derive(Clone, Copy)]
struct Slot {
    /// the field, the attacker and the defender, hashed. Zero means empty.
    key: u64,
    hits: PairHits,
}

#[derive(Clone, Copy)]
struct PairCache {
    next: u8,
    slots: [Slot; WAYS],
}

impl PairCache {
    const EMPTY: PairCache = PairCache {
        next: 0,
        slots: [Slot {
            key: 0,
            hits: PairHits::EMPTY,
        }; WAYS],
    };

    fn get(&self, key: u64) -> Option<&PairHits> {
        for slot in self.slots.iter() {
            if slot.key == key {
                return Some(&slot.hits);
            }
        }
        None
    }

    fn put(&mut self, key: u64, hits: PairHits) {
        let index = self.next as usize % WAYS;
        self.next = self.next.wrapping_add(1);
        self.slots[index] = Slot { key, hits };
    }
}

struct Table {
    /// `hits[0][i][j]`: side one's slot i attacking side two's slot j
    hits: [[[PairCache; MAX_SLOTS]; MAX_SLOTS]; 2],
}

impl Table {
    fn new() -> Table {
        Table {
            hits: [[[PairCache::EMPTY; MAX_SLOTS]; MAX_SLOTS]; 2],
        }
    }
}

thread_local! {
    // per thread rather than shared: a search thread builds its own table and
    // then never contends for it. The alternative, a locked global, pays a
    // synchronisation cost on every leaf to save a few dozen damage calls once.
    static TABLE: RefCell<Table> = RefCell::new(Table::new());
}

fn hash_step(accumulator: u64, value: u64) -> u64 {
    // FNV-1a, which is short enough to inline and has no dependencies
    (accumulator ^ value).wrapping_mul(0x100_0000_01b3)
}

/// Everything about a Pokemon that changes a damage roll, except HP and boosts.
///
/// Five fields, because everything else that matters moves with them: the stats,
/// the types and the moves change when the species does (a Mega, a forme change)
/// and the species is in here. This runs for every living Pokemon on every leaf,
/// so its length is a cost.
fn pokemon_fingerprint(pokemon: &Pokemon) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    h = hash_step(h, pokemon.id as i16 as u64);
    h = hash_step(h, pokemon.item as u8 as u64);
    h = hash_step(h, pokemon.ability as i16 as u64);
    h = hash_step(h, pokemon.status as u8 as u64);
    h = hash_step(h, pokemon.terastallized as u64);
    h
}

/// The field and both sides' screens. Not the boosts: those are analytic.
fn global_fingerprint(state: &State) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    h = hash_step(h, state.get_weather() as u8 as u64);
    h = hash_step(h, state.terrain.terrain_type as u8 as u64);
    h = hash_step(h, state.trick_room.active as u64);
    for side in [&state.side_one, &state.side_two] {
        h = hash_step(h, (side.side_conditions.reflect > 0) as u64);
        h = hash_step(h, (side.side_conditions.light_screen > 0) as u64);
        h = hash_step(h, (side.side_conditions.aurora_veil > 0) as u64);
    }
    h
}

/// The HP a Pokemon is given while the table is built: one point below full.
///
/// The premise of the table is that damage does not depend on HP, and that is
/// nearly true. The exceptions are all at thresholds: the engine's own hooks
/// halve the incoming hit for Multiscale and Shadow Shield at *exactly* full HP,
/// halve the outgoing one for Defeatist at half, and raise it for Overgrow /
/// Blaze / Torrent / Swarm at a third, while `modify_choice` scales Eruption and
/// Water Spout by the ratio.
///
/// Building one point below full deals with the first of those, which is the one
/// that would otherwise be *wrong* rather than merely absent: `turns_to_ko`
/// already models Multiscale and Shadow Shield as "the first hit is halved" —
/// which is what they do, since the second hit lands on a target that is no
/// longer full — so letting the engine's hook halve the cached roll as well
/// would count it twice, and would keep counting it for every later hit.
///
/// The rest are *absent*: a pinch ability never fires in the table and a
/// ratio-scaled move is priced near full. That is the same limitation
/// `bot/search/wincon.py` has, and it is deliberate rather than free — putting an
/// HP band in the cache key instead was built and measured, and it triples the
/// variants a pair can have: the term went from +16% of a search iteration to
/// +63%. Threshold abilities that only bite below a third of HP are not worth
/// four times the cost of the whole term.
fn table_hp(maxhp: i16) -> i16 {
    (maxhp - 1).max(1)
}

/// A side as a bench member would meet it: that member active, no boosts, no
/// volatile statuses. Boosts are lost on switch-out, so a bench member does not
/// carry the active's; the active's own boosts are put back analytically at the
/// leaf, which is what makes the table independent of them.
fn scratch_side(side: &Side, slot: PokemonIndex) -> Side {
    let mut out = side.clone();
    out.active_index = slot;
    out.attack_boost = 0;
    out.defense_boost = 0;
    out.special_attack_boost = 0;
    out.special_defense_boost = 0;
    out.speed_boost = 0;
    out.accuracy_boost = 0;
    out.evasion_boost = 0;
    out.volatile_statuses = VolatileStatusBitset(0);
    out.substitute_health = 0;
    for index in 0..MAX_SLOTS {
        let pokemon = &mut out.pokemon[slot_of(index)];
        if pokemon.maxhp > 0 {
            pokemon.hp = table_hp(pokemon.maxhp);
        }
    }
    out
}

/// A copy of the position with both sides stripped of boosts and volatiles, for
/// putting an arbitrary pair on the field and pricing it.
///
/// The table is built by putting the pair on the field of a *copy* of the state
/// and asking the engine what its own move resolution would ask — `modify_choice`
/// and the four ability/item hooks, then `calculate_damage`. Going through the
/// real pipeline rather than the raw `MOVES` entry is what makes the numbers
/// mean the same thing the search's own damage means: Life Orb, Adaptability,
/// Tough Claws, Technician and the rest all live in those hooks and none of them
/// is in a move's base power. Measured against the Python reference on 300 real
/// ladder positions, skipping them cost r = 0.976; going through them costs a
/// clone of the state per leaf that misses the cache.
fn scratch_state(state: &State) -> State {
    let mut out = state.clone();
    out.side_one = scratch_side(&state.side_one, state.side_one.active_index);
    out.side_two = scratch_side(&state.side_two, state.side_two.active_index);
    out
}

fn slot_of(index: usize) -> PokemonIndex {
    match index {
        0 => PokemonIndex::P0,
        1 => PokemonIndex::P1,
        2 => PokemonIndex::P2,
        3 => PokemonIndex::P3,
        4 => PokemonIndex::P4,
        _ => PokemonIndex::P5,
    }
}

/// Every damaging move the attacking side's active has, as a fraction of the
/// defending active's max HP, at zero boosts.
///
/// `state` must already be a `scratch_state` for the pair, and `attacker_side`
/// says which of its two sides is attacking.
fn build_pair(state: &State, attacker_side: SideReference) -> PairHits {
    let mut out = PairHits {
        n: 0,
        rolls: [MoveRoll::NONE; MAX_MOVES],
    };
    let (attacking, defending) = state.get_both_sides_immutable(&attacker_side);
    let attacker = attacking.get_active_immutable();
    let defender = defending.get_active_immutable();
    if defender.maxhp <= 0 {
        return out;
    }
    let moves = [
        &attacker.moves.m0,
        &attacker.moves.m1,
        &attacker.moves.m2,
        &attacker.moves.m3,
    ];
    let scale = 1.0 / defender.maxhp as f32;
    // the defender's choice only reaches `modify_choice` for moves that read it
    // (Sucker Punch and friends); a bench pair has no such thing, so it gets the
    // empty choice, which is how those moves are priced when nothing is known.
    let no_choice = Choice::default();
    for m in moves {
        if m.pp <= 0 || m.disabled {
            continue;
        }
        if m.choice.category == MoveCategory::Status
            || m.choice.category == MoveCategory::Switch
            || m.choice.base_power == 0.0
        {
            continue;
        }
        // exactly the hooks `generate_instructions` runs before it calculates
        // damage, minus the ones that mutate the state
        let mut choice = m.choice.clone();
        modify_choice(state, &mut choice, &no_choice, &attacker_side);
        ability_modify_attack_being_used(state, &mut choice, &no_choice, &attacker_side);
        ability_modify_attack_against(state, &mut choice, &no_choice, &attacker_side);
        item_modify_attack_being_used(state, &mut choice, &attacker_side);
        item_modify_attack_against(state, &mut choice, &attacker_side);
        if choice.category == MoveCategory::Status || choice.base_power == 0.0 {
            continue;
        }
        let hi = match calculate_damage(state, &attacker_side, &choice, DamageRolls::Max) {
            Some((d, _)) if d > 0 => d,
            _ => continue,
        };
        // the engine's own minimum roll: `damage.floor() * 0.85` truncated
        let lo = (hi as f32 * 0.85) as i16;
        let accuracy = if choice.accuracy >= 100.0 {
            1.0
        } else {
            (choice.accuracy / 100.0).max(0.0)
        };
        out.rolls[out.n as usize] = MoveRoll {
            lo: lo as f32 * scale,
            hi: hi as f32 * scale,
            accuracy,
            priority: choice.priority,
            physical: choice.category == MoveCategory::Physical,
        };
        out.n += 1;
        if out.n as usize == MAX_MOVES {
            break;
        }
    }
    out
}

// ------------------------------------------------------------------ the matrix

fn boost_multiplier(stage: i8) -> f32 {
    let stage = stage.clamp(-6, 6) as f32;
    if stage >= 0.0 {
        (2.0 + stage) / 2.0
    } else {
        2.0 / (2.0 - stage)
    }
}

fn phi(z: f32) -> f32 {
    // standard normal CDF via erf, which f32 does not have in std
    0.5 * (1.0 + erf(z * INV_SQRT2))
}

/// `exp` built out of the float's own exponent field (Schraudolph).
///
/// Relative error under 6% and one-sided, and about eight times faster than the
/// libm call.
/// This runs about forty times a leaf and a leaf runs once per MCTS iteration,
/// so the libm version was most of the term's cost: measured at 200,000
/// iterations it took the leaf term from +38% of an iteration to +16%. Three
/// percent on a probability that is already a logistic fitted to a damage roll
/// modelled as a normal is not a number anybody is reading.
#[inline(always)]
fn fast_exp(x: f32) -> f32 {
    if x < -80.0 {
        return 0.0;
    }
    if x > 80.0 {
        return f32::INFINITY;
    }
    // 2^23 / ln 2 for the slope, and 127 << 23 for the offset. The usual
    // Schraudolph constant subtracts 486_411 from that offset to halve the RMS
    // error; this one does not, because keeping the offset exact makes
    // `fast_exp(0.0)` exactly 1.0 — and that is what makes a mirror match read
    // exactly zero rather than 0.005, which is a property worth more here than
    // three percent of accuracy on a heuristic probability.
    let bits = 12_102_203.0 * x + 1_065_353_216.0;
    f32::from_bits(bits as i32 as u32)
}

/// Abramowitz & Stegun 7.1.26 — five significant figures, which is far past what
/// a damage roll modelled as a normal deserves.
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_) * t) + 1.421_413_7) * t - 0.284_496_74) * t
            + 0.254_829_59)
            * t
            * fast_exp(-x * x);
    sign * y
}

/// Expected turns for a stream of `hit` to knock out a defender on `hp`
/// (a fraction of its max), capped at `TURN_CAP`.
///
/// The roll is uniform on [lo, hi]; n hits accumulate to mean n·mu and variance
/// n·sigma², so P(KO within n) is a normal tail and E[hits] is the sum of the
/// survival probabilities. Doing it this way rather than `ceil(hp / mean)` is
/// the point: HP just below the minimum roll answers 1, just above it the answer
/// jumps, and that jump is what an "can I OHKO this" judgement is made of.
fn turns_to_ko(
    lo: f32,
    hi: f32,
    accuracy: f32,
    hp: f32,
    halve_first: bool,
    survive_one: bool,
) -> f32 {
    let mean = 0.5 * (lo + hi);
    if mean <= 0.0 {
        return TURN_CAP as f32;
    }
    let sigma = (hi - lo) / SQRT12;
    let mut expected = 0.0;
    let mut mu_n = 0.0;
    let mut var_n = 0.0;
    let mut survived = 1.0f32;
    for n in 0..TURN_CAP {
        expected += survived;
        // Once it is all but certainly dead the remaining terms are worth at
        // most 8e-3 of a turn between them, and each one costs an exp. This is
        // the difference between eight passes and two or three on the typical
        // pair, and it runs eighteen times a leaf.
        if survived <= 1e-3 {
            break;
        }
        let scale = if halve_first && n == 0 { 0.5 } else { 1.0 };
        mu_n += mean * scale;
        var_n += (sigma * scale) * (sigma * scale);
        if survive_one && n == 0 {
            // Sash / Sturdy leaves it on 1 HP whatever the roll, so the second
            // hit is the earliest KO; the accumulated mean is left alone because
            // from 1 HP anything at all finishes it.
            continue;
        }
        let killed = if var_n <= 1e-12 {
            if mu_n >= hp {
                1.0
            } else {
                0.0
            }
        } else {
            // four standard deviations out the normal CDF is 3e-5 from its
            // limit, which is nothing next to the roll being modelled as normal
            // in the first place — and skipping it skips the exp
            let z = (mu_n - hp) / var_n.sqrt();
            if z > 4.0 {
                1.0
            } else if z < -4.0 {
                0.0
            } else {
                phi(z)
            }
        };
        survived = survived.min(1.0 - killed);
    }
    (expected / accuracy.max(1e-6)).min(TURN_CAP as f32)
}

fn effective_speed(pokemon: &Pokemon, side: &Side, active: bool) -> f32 {
    let mut speed = pokemon.speed as f32;
    if active {
        speed *= boost_multiplier(side.speed_boost);
    }
    if pokemon.status == PokemonStatus::PARALYZE {
        speed *= 0.5;
    }
    if side.side_conditions.tailwind > 0 {
        speed *= 2.0;
    }
    if pokemon.item == Items::CHOICESCARF {
        speed *= 1.5;
    }
    speed
}

/// The attacker's best move into this defender *now*: the cached rolls rescaled
/// by whatever boosts are on the field, then the highest expected fraction per
/// turn. Damage is linear in the attacking stat and inverse-linear in the
/// defending one, so a boost is a multiplier on a cached roll and not a reason
/// to call the damage calculator again.
fn best_hit(
    hits: &PairHits,
    attacker_active: bool,
    attacking_side: &Side,
    defender_active: bool,
    defending_side: &Side,
    intimidated: bool,
) -> (f32, f32, f32, i8) {
    if hits.n == 0 {
        return (0.0, 0.0, 1.0, 0);
    }
    // Two scales, not one per move: the boost stages depend on the move only
    // through whether it is physical, and `boost_multiplier` is a divide. This
    // runs eighteen times a leaf, so hoisting it out of the move loop is worth
    // more than it looks.
    let (attack_physical, attack_special) = if attacker_active {
        (
            attacking_side.attack_boost,
            attacking_side.special_attack_boost,
        )
    } else {
        (0, 0)
    };
    let attack_physical = if intimidated {
        // switching into an Intimidate user costs a stage of Attack, which is a
        // threshold change and not an average
        attack_physical - 1
    } else {
        attack_physical
    };
    let (defend_physical, defend_special) = if defender_active {
        (
            defending_side.defense_boost,
            defending_side.special_defense_boost,
        )
    } else {
        (0, 0)
    };
    let physical_scale = boost_multiplier(attack_physical) / boost_multiplier(defend_physical);
    let special_scale = boost_multiplier(attack_special) / boost_multiplier(defend_special);

    let mut best = (0.0f32, 0.0f32, 1.0f32, 0i8);
    let mut best_rate = 0.0f32;
    for index in 0..hits.n as usize {
        let roll = hits.rolls[index];
        let scale = if roll.physical {
            physical_scale
        } else {
            special_scale
        };
        let (lo, hi) = (roll.lo * scale, roll.hi * scale);
        let rate = 0.5 * (lo + hi) * roll.accuracy;
        if rate > best_rate {
            best_rate = rate;
            best = (lo, hi, roll.accuracy, roll.priority);
        }
    }
    best
}

struct Member {
    slot: usize,
    hp: f32,
    /// effective speed, boosts and paralysis and Tailwind and Scarf included:
    /// a property of the Pokemon, so it is computed once per member rather than
    /// once per pair
    speed: f32,
    active: bool,
    halve_first: bool,
    survive_one: bool,
    ability: Abilities,
}

fn members(side: &Side) -> ([Member; MAX_SLOTS], usize) {
    let mut out = [
        Member {
            slot: 0,
            hp: 0.0,
            speed: 0.0,
            active: false,
            halve_first: false,
            survive_one: false,
            ability: Abilities::NONE,
        },
        Member {
            slot: 0,
            hp: 0.0,
            speed: 0.0,
            active: false,
            halve_first: false,
            survive_one: false,
            ability: Abilities::NONE,
        },
        Member {
            slot: 0,
            hp: 0.0,
            speed: 0.0,
            active: false,
            halve_first: false,
            survive_one: false,
            ability: Abilities::NONE,
        },
        Member {
            slot: 0,
            hp: 0.0,
            speed: 0.0,
            active: false,
            halve_first: false,
            survive_one: false,
            ability: Abilities::NONE,
        },
        Member {
            slot: 0,
            hp: 0.0,
            speed: 0.0,
            active: false,
            halve_first: false,
            survive_one: false,
            ability: Abilities::NONE,
        },
        Member {
            slot: 0,
            hp: 0.0,
            speed: 0.0,
            active: false,
            halve_first: false,
            survive_one: false,
            ability: Abilities::NONE,
        },
    ];
    let mut count = 0;
    for index in 0..MAX_SLOTS {
        let pokemon = &side.pokemon[slot_of(index)];
        if pokemon.hp <= 0 || pokemon.maxhp <= 0 {
            continue;
        }
        let hp = pokemon.hp as f32 / pokemon.maxhp as f32;
        let full = pokemon.hp >= pokemon.maxhp;
        let active = slot_of(index) == side.active_index;
        out[count] = Member {
            slot: index,
            hp,
            speed: effective_speed(pokemon, side, active),
            active,
            halve_first: full
                && (pokemon.ability == Abilities::MULTISCALE
                    || pokemon.ability == Abilities::SHADOWSHIELD),
            survive_one: full
                && (pokemon.ability == Abilities::STURDY || pokemon.item == Items::FOCUSSASH),
            ability: pokemon.ability,
        };
        count += 1;
    }
    (out, count)
}

/// `prod_j c_j - prod_i d_i`, from side one's perspective, in [-1, 1].
///
/// Zero when either side has nothing left — the position is decided and the
/// rest of the evaluation already says so.
pub fn coverage(state: &State) -> f32 {
    let (mine, n_mine) = members(&state.side_one);
    let (theirs, n_theirs) = members(&state.side_two);
    if n_mine == 0 || n_theirs == 0 {
        return 0.0;
    }

    // one hash per living Pokemon, folded with the field, then one lookup per
    // ordered pair: the whole validity check, with no invalidation pass
    let global = global_fingerprint(state);
    let mut my_keys = [0u64; MAX_SLOTS];
    for (index, me) in mine[..n_mine].iter().enumerate() {
        my_keys[index] = pokemon_fingerprint(&state.side_one.pokemon[slot_of(me.slot)]);
    }
    let mut their_keys = [0u64; MAX_SLOTS];
    for (index, them) in theirs[..n_theirs].iter().enumerate() {
        their_keys[index] = pokemon_fingerprint(&state.side_two.pokemon[slot_of(them.slot)]);
    }

    let mut matrix = [[0.0f32; MAX_SLOTS]; MAX_SLOTS];
    let mut scratch: Option<State> = None;
    TABLE.with(|cell| {
        let mut table = cell.borrow_mut();
        for (i, me) in mine[..n_mine].iter().enumerate() {
            for (j, them) in theirs[..n_theirs].iter().enumerate() {
                let forward = hash_step(hash_step(global, my_keys[i]), their_keys[j]) | 1;
                let backward = hash_step(hash_step(global, their_keys[j]), my_keys[i]) | 1;
                if table.hits[0][me.slot][them.slot].get(forward).is_none()
                    || table.hits[1][them.slot][me.slot].get(backward).is_none()
                {
                    // One clone for the whole leaf, not one per pair: a `State`
                    // holds 24 `Move`s and each `Move` holds a `Choice` with a
                    // heap `Vec` of secondaries, so cloning it is the most
                    // expensive thing in this file. Which two are on the field
                    // is then two integer writes.
                    let pair = scratch.get_or_insert_with(|| scratch_state(state));
                    pair.side_one.active_index = slot_of(me.slot);
                    pair.side_two.active_index = slot_of(them.slot);
                    if table.hits[0][me.slot][them.slot].get(forward).is_none() {
                        let hits = build_pair(pair, SideReference::SideOne);
                        table.hits[0][me.slot][them.slot].put(forward, hits);
                    }
                    if table.hits[1][them.slot][me.slot].get(backward).is_none() {
                        let hits = build_pair(pair, SideReference::SideTwo);
                        table.hits[1][them.slot][me.slot].put(backward, hits);
                    }
                }
                matrix[i][j] = pair_win(state, &table, me, them, forward, backward);
            }
        }
    });

    let mut cover_all = 1.0f32;
    for j in 0..n_theirs {
        let mut miss = 1.0f32;
        for row in matrix.iter().take(n_mine) {
            miss *= 1.0 - row[j];
        }
        cover_all *= 1.0 - miss;
    }
    let mut their_cover_all = 1.0f32;
    for row in matrix.iter().take(n_mine) {
        let mut d = 1.0f32;
        for j in 0..n_theirs {
            d *= row[j];
        }
        their_cover_all *= 1.0 - d;
    }
    cover_all - their_cover_all
}

fn pair_win(
    state: &State,
    table: &Table,
    me: &Member,
    them: &Member,
    forward_key: u64,
    backward_key: u64,
) -> f32 {
    // Intimidate applies to whichever one has to switch in; the pair already on
    // the field settled it long ago.
    let my_intimidated = !me.active && them.active && them.ability == Abilities::INTIMIDATE;
    let their_intimidated = !them.active && me.active && me.ability == Abilities::INTIMIDATE;

    let empty = PairHits::EMPTY;
    let mine = best_hit(
        table.hits[0][me.slot][them.slot]
            .get(forward_key)
            .unwrap_or(&empty),
        me.active,
        &state.side_one,
        them.active,
        &state.side_two,
        my_intimidated,
    );
    let theirs = best_hit(
        table.hits[1][them.slot][me.slot]
            .get(backward_key)
            .unwrap_or(&empty),
        them.active,
        &state.side_two,
        me.active,
        &state.side_one,
        their_intimidated,
    );

    let t_mine = turns_to_ko(
        mine.0,
        mine.1,
        mine.2,
        them.hp,
        them.halve_first,
        them.survive_one,
    );
    let t_theirs = turns_to_ko(
        theirs.0,
        theirs.1,
        theirs.2,
        me.hp,
        me.halve_first,
        me.survive_one,
    );
    // neither can break the other: a wall, and a wall is 0.5 whoever is faster
    if t_mine >= TURN_CAP as f32 && t_theirs >= TURN_CAP as f32 {
        return 0.5;
    }

    let edge = if mine.3 != theirs.3 {
        if mine.3 > theirs.3 {
            1.0
        } else {
            -1.0
        }
    } else {
        let (my_speed, their_speed) = (me.speed, them.speed);
        if (my_speed - their_speed).abs() < 1e-6 {
            0.0
        } else if (my_speed > their_speed) != state.trick_room.active {
            1.0
        } else {
            -1.0
        }
    };

    let score = (t_theirs - t_mine) + SPEED_WEIGHT * edge;
    1.0 / (1.0 + fast_exp(-LOGISTIC_K * score))
}

/// The term the evaluation adds: the weight times the coverage difference.
/// Zero weight returns before touching anything, so the default build does not
/// pay for it and does not change by a bit.
pub fn coverage_term(state: &State) -> f32 {
    let weight = wincon_coverage_weight();
    if weight == 0.0 {
        return 0.0;
    }
    weight * coverage(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::choices::{Choice, Choices, MoveCategory};
    use crate::pokemon::PokemonName;
    use crate::state::{Move, PokemonType};

    fn damaging_move(id: Choices, base_power: f32, category: MoveCategory) -> Move {
        let mut choice = Choice::default();
        choice.move_id = id;
        choice.base_power = base_power;
        choice.category = category;
        choice.accuracy = 100.0;
        choice.move_type = PokemonType::NORMAL;
        Move {
            id,
            disabled: false,
            pp: 32,
            choice,
        }
    }

    /// two 100/100 Pokemon a side, everyone knowing one 80 BP physical move
    fn symmetric() -> State {
        let mut state = State::default();
        for side in [&mut state.side_one, &mut state.side_two] {
            for index in 0..2 {
                let pokemon = &mut side.pokemon[slot_of(index)];
                pokemon.id = PokemonName::PIKACHU;
                pokemon.level = 50;
                pokemon.hp = 100;
                pokemon.maxhp = 100;
                pokemon.attack = 100;
                pokemon.defense = 100;
                pokemon.special_attack = 100;
                pokemon.special_defense = 100;
                pokemon.speed = 100;
                pokemon.types = (PokemonType::NORMAL, PokemonType::TYPELESS);
                pokemon.base_types = pokemon.types;
                pokemon.moves.m0 = damaging_move(Choices::TACKLE, 80.0, MoveCategory::Physical);
                pokemon.moves.m1 = Move::default();
                pokemon.moves.m2 = Move::default();
                pokemon.moves.m3 = Move::default();
            }
            for index in 2..MAX_SLOTS {
                side.pokemon[slot_of(index)].hp = 0;
            }
        }
        state
    }

    fn fresh(state: &State) -> f32 {
        // every test starts from a clean table: the thread-local survives
        // between tests in the same thread and the fingerprints are the point
        TABLE.with(|cell| *cell.borrow_mut() = Table::new());
        coverage(state)
    }

    #[test]
    fn a_symmetric_position_is_zero() {
        let value = fresh(&symmetric());
        assert!(
            value.abs() < 1e-5,
            "{}",
            format!("a mirror match must read exactly level, not {value}")
        );
    }

    #[test]
    fn coverage_favours_the_side_that_can_kill() {
        let mut state = symmetric();
        // their two cannot hurt anything
        for index in 0..2 {
            state.side_two.pokemon[slot_of(index)].moves.m0 = Move::default();
        }
        let value = fresh(&state);
        assert!(
            value > 0.9,
            "{}",
            format!(
                "answering everything while nothing answers you should be near +1, not {value}"
            )
        );
    }

    #[test]
    fn a_threat_with_no_answer_reads_negative() {
        let mut state = symmetric();
        for index in 0..2 {
            state.side_one.pokemon[slot_of(index)].moves.m0 = Move::default();
        }
        let value = fresh(&state);
        assert!(
            value < -0.9,
            "{}",
            format!("a threat nothing on my side beats should be near -1, not {value}")
        );
    }

    #[test]
    fn losing_the_only_answer_costs_more_than_losing_a_spare() {
        // their slot 0 is a wall to everything but my slot 0
        let mut state = symmetric();
        state.side_two.pokemon[slot_of(0)].defense = 400;
        state.side_two.pokemon[slot_of(0)].special_defense = 400;
        state.side_one.pokemon[slot_of(0)].attack = 400;
        let base = fresh(&state);
        assert!(base.is_finite(), "{}", format!("{base}"));

        let mut without_answer = state.clone();
        without_answer.side_one.pokemon[slot_of(0)].hp = 0;
        let lost_answer = fresh(&without_answer);

        let mut without_spare = state.clone();
        without_spare.side_one.pokemon[slot_of(1)].hp = 0;
        let lost_spare = fresh(&without_spare);

        assert!(
            lost_answer < lost_spare,
            "{}",
            format!(
                "losing the only answer ({lost_answer}) must cost more than losing the spare \
             ({lost_spare}), from {base}"
            )
        );
    }

    #[test]
    fn speed_breaks_a_tie() {
        let mut state = symmetric();
        state.side_one.pokemon[slot_of(0)].speed = 200;
        state.side_one.pokemon[slot_of(1)].speed = 200;
        let faster = fresh(&state);
        state.side_two.pokemon[slot_of(0)].speed = 200;
        state.side_two.pokemon[slot_of(1)].speed = 200;
        let level = fresh(&state);
        assert!(
            faster > level,
            "{}",
            format!("out-speeding an identical side must read better: {faster} vs {level}")
        );
    }

    #[test]
    fn the_term_is_off_by_default_and_scales_with_the_weight() {
        let mut state = symmetric();
        for index in 0..2 {
            state.side_two.pokemon[slot_of(index)].moves.m0 = Move::default();
        }
        assert_eq!(coverage_term(&state), 0.0, "the default must cost nothing");
        set_wincon_coverage_weight(100.0);
        let scaled = coverage_term(&state);
        let raw = coverage(&state);
        set_wincon_coverage_weight(0.0);
        assert!(
            (scaled - 100.0 * raw).abs() < 1e-3,
            "{}",
            format!("{scaled} vs {raw}")
        );
        assert_eq!(coverage_term(&state), 0.0, "back off again");
    }

    #[test]
    fn a_mega_invalidates_its_row_and_column() {
        let mut state = symmetric();
        let before = fresh(&state);
        // the same table, then a forme change that doubles their attack: if the
        // fingerprint did not catch it the cached rolls would be reused
        state.side_two.pokemon[slot_of(0)].id = PokemonName::CHARIZARDMEGAX;
        state.side_two.pokemon[slot_of(0)].attack = 300;
        let after = coverage(&state);
        assert!(
            after < before - 0.05,
            "{}",
            format!("a Mega that hits far harder must change the matrix: {after} vs {before}")
        );
    }

    #[test]
    fn turns_to_ko_steps_at_the_roll_window() {
        // a roll of 0.50-0.60 of max HP: just under the minimum is one turn,
        // just over the maximum is two, and the middle is not an average
        let under = turns_to_ko(0.5, 0.6, 1.0, 0.49, false, false);
        let over = turns_to_ko(0.5, 0.6, 1.0, 0.61, false, false);
        assert!(
            under < 1.05,
            "{}",
            format!("under the minimum roll is one hit: {under}")
        );
        assert!(
            over > 1.9 && over < 2.05,
            "{}",
            format!("over the maximum roll is two hits: {over}")
        );
    }

    #[test]
    fn a_sash_buys_exactly_one_more_hit() {
        // a roll that always kills from full: one turn plain, two through a Sash
        let plain = turns_to_ko(1.1, 1.2, 1.0, 1.0, false, false);
        let sashed = turns_to_ko(1.1, 1.2, 1.0, 1.0, false, true);
        assert!(
            (plain - 1.0).abs() < 1e-4,
            "{}",
            format!("a guaranteed OHKO is one turn: {plain}")
        );
        assert!(
            (sashed - 2.0).abs() < 1e-4,
            "{}",
            format!("Focus Sash costs exactly one whole turn: {sashed} vs {plain}")
        );
    }

    #[test]
    fn multiscale_is_worth_one_halved_hit_and_not_two() {
        // The engine's own `ability_modify_attack_against` halves the incoming
        // hit while the target is at exactly full HP, and `turns_to_ko` halves
        // the first hit too. Counting both would make a full-HP Dragonite twice
        // the wall it is — and would keep halving every later hit, which is not
        // what Multiscale does. The table is therefore built one HP below full.
        let mut plain = symmetric();
        let mut multiscale = symmetric();
        multiscale.side_two.pokemon[slot_of(0)].ability = Abilities::MULTISCALE;
        multiscale.side_two.pokemon[slot_of(1)].ability = Abilities::MULTISCALE;
        // the comparison that bounds it: halving *every* hit, not just the first
        let mut double_defense = symmetric();
        for index in 0..2 {
            double_defense.side_two.pokemon[slot_of(index)].defense *= 2;
            double_defense.side_two.pokemon[slot_of(index)].special_defense *= 2;
        }

        let base = fresh(&plain);
        let with_ability = fresh(&multiscale);
        let with_bulk = fresh(&double_defense);
        assert!(
            with_ability < base,
            "{}",
            format!("Multiscale must be worth something: {with_ability} vs {base}")
        );
        assert!(
            with_ability > with_bulk,
            "{}",
            format!(
                "one halved hit must be worth less than halving every hit: \
             {with_ability} vs {with_bulk}"
            )
        );
        // and it must go away the moment the target is not full
        plain.side_two.pokemon[slot_of(0)].hp = 99;
        multiscale.side_two.pokemon[slot_of(0)].hp = 99;
        let chipped_plain = fresh(&plain);
        let chipped_ability = fresh(&multiscale);
        assert!(
            (chipped_ability - chipped_plain).abs() < (with_ability - base).abs(),
            "{}",
            format!(
                "a chipped Multiscale is worth less than a full one: \
             {chipped_ability} - {chipped_plain} against {with_ability} - {base}"
            )
        );
    }

    #[test]
    fn an_empty_side_is_zero() {
        let mut state = symmetric();
        for index in 0..MAX_SLOTS {
            state.side_two.pokemon[slot_of(index)].hp = 0;
        }
        assert_eq!(fresh(&state), 0.0);
    }

    #[test]
    fn moves_that_are_unavailable_do_not_count() {
        let mut state = symmetric();
        for index in 0..2 {
            state.side_two.pokemon[slot_of(index)].moves.m0.pp = 0;
        }
        let out_of_pp = fresh(&state);
        assert!(
            out_of_pp > 0.9,
            "{}",
            format!("a move with no PP left cannot be their answer: {out_of_pp}")
        );
    }
}
