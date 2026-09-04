use crate::engine::evaluate::evaluate;

// The leaf the tree scores with. Older generations have no value net module;
// genx blends the static evaluation with a loaded net (see genx::value_net).
#[cfg(any(feature = "gen1", feature = "gen2", feature = "gen3"))]
fn leaf(state: &State) -> f32 {
    evaluate(state)
}
#[cfg(not(any(feature = "gen1", feature = "gen2", feature = "gen3")))]
fn leaf(state: &State) -> f32 {
    crate::engine::value_net::evaluate_leaf(state, evaluate(state))
}
use crate::engine::generate_instructions::generate_instructions_from_move_pair;
use crate::engine::state::MoveChoice;
use crate::instruction::StateInstructions;
use crate::state::State;
use rand::prelude::*;
use rand::rng;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

/// How many turns of random play to run at a leaf before scoring it.
///
/// Zero (the default) keeps the original behaviour: a leaf is scored by the
/// static evaluation the moment it is expanded. A static score a few plies in
/// cannot tell who wins a Protect / Toxic / Recover war, because nothing in the
/// snapshot runs out — a playout lets the residual damage, the PP and the
/// counters actually accrue before the position is judged. Set from Python via
/// `set_playout_turns`.
pub static PLAYOUT_TURNS: AtomicU32 = AtomicU32::new(0);

fn sigmoid(x: f32) -> f32 {
    // Tuned so that ~200 points is very close to 1.0
    1.0 / (1.0 + (-0.0125 * x).exp())
}

#[derive(Debug)]
pub struct Node {
    pub root: bool,
    pub parent: *mut Node,
    pub times_visited: u32,

    // represents the instructions & s1/s2 moves that led to this node from the parent
    pub instructions: StateInstructions,
    pub s1_choice: u8,
    pub s2_choice: u8,

    // represents the total score and number of visits for this node
    // de-coupled for s1 and s2
    pub s1_options: Option<Vec<MoveNode>>,
    pub s2_options: Option<Vec<MoveNode>>,
}

impl Node {
    fn new() -> Node {
        Node {
            root: false,
            parent: std::ptr::null_mut(),
            instructions: StateInstructions::default(),
            times_visited: 0,
            s1_choice: 0,
            s2_choice: 0,
            s1_options: None,
            s2_options: None,
        }
    }
    unsafe fn populate(&mut self, s1_options: Vec<MoveChoice>, s2_options: Vec<MoveChoice>) {
        let s1_options_vec: Vec<MoveNode> = s1_options
            .iter()
            .map(|x| MoveNode {
                move_choice: x.clone(),
                total_score: 0.0,
                visits: 0,
            })
            .collect();
        let s2_options_vec: Vec<MoveNode> = s2_options
            .iter()
            .map(|x| MoveNode {
                move_choice: x.clone(),
                total_score: 0.0,
                visits: 0,
            })
            .collect();

        self.s1_options = Some(s1_options_vec);
        self.s2_options = Some(s2_options_vec);
    }

    pub fn maximize_ucb_for_side(&self, side_map: &[MoveNode]) -> usize {
        let mut choice = 0;
        let mut best_ucb1 = f32::MIN;
        for (index, node) in side_map.iter().enumerate() {
            let this_ucb1 = node.ucb1(self.times_visited);
            if this_ucb1 > best_ucb1 {
                best_ucb1 = this_ucb1;
                choice = index;
            }
        }
        choice
    }

    pub unsafe fn selection(
        &mut self,
        state: &mut State,
        children: &mut HashMap<(usize, usize, usize), Box<[Node]>>,
        rng: &mut impl Rng,
    ) -> (*mut Node, usize, usize) {
        if self.s1_options.is_none() {
            let (s1_options, s2_options) = state.get_all_options();
            self.populate(s1_options, s2_options);
        }

        let s1_mc_index = self.maximize_ucb_for_side(self.s1_options.as_ref().unwrap());
        let s2_mc_index = self.maximize_ucb_for_side(self.s2_options.as_ref().unwrap());
        let key = (self as *mut Node as usize, s1_mc_index, s2_mc_index);
        match children.get_mut(&key) {
            Some(child_vector) => {
                let child_vec_ptr = child_vector as *mut Box<[Node]>;
                let chosen_child = self.sample_node(child_vec_ptr, rng);
                state.apply_instructions(&(*chosen_child).instructions.instruction_list);
                (*chosen_child).selection(state, children, rng)
            }
            None => (self as *mut Node, s1_mc_index, s2_mc_index),
        }
    }

    unsafe fn sample_node(&self, move_vector: *mut Box<[Node]>, rng: &mut impl Rng) -> *mut Node {
        let nodes = &mut **move_vector;

        let total_weight: f32 = nodes
            .iter()
            .map(|n| n.instructions.percentage.max(0.0))
            .sum();

        let mut threshold = rng.random_range(0.0..total_weight);

        for node in nodes.iter_mut() {
            threshold -= node.instructions.percentage.max(0.0);
            if threshold <= 0.0 {
                return node as *mut Node;
            }
        }

        // fallback: return last node (handles float rounding issues that can come up)
        &mut nodes[nodes.len() - 1] as *mut Node
    }

    pub unsafe fn expand(
        &mut self,
        state: &mut State,
        s1_move_index: usize,
        s2_move_index: usize,
        children: &mut HashMap<(usize, usize, usize), Box<[Node]>>,
        rng: &mut impl Rng,
    ) -> *mut Node {
        let s1_move = &self.s1_options.as_ref().unwrap()[s1_move_index].move_choice;
        let s2_move = &self.s2_options.as_ref().unwrap()[s2_move_index].move_choice;
        // if the battle is over or both moves are none there is no need to expand
        if (state.battle_is_over() != 0.0 && !self.root)
            || (s1_move == &MoveChoice::None && s2_move == &MoveChoice::None)
        {
            return self as *mut Node;
        }
        let should_branch_on_damage = self.root || (*self.parent).root;
        let mut new_instructions =
            generate_instructions_from_move_pair(state, s1_move, s2_move, should_branch_on_damage);
        let mut this_pair_vec = Vec::with_capacity(new_instructions.len());
        for state_instructions in new_instructions.drain(..) {
            let mut new_node = Node::new();
            new_node.parent = self;
            new_node.instructions = state_instructions;
            new_node.s1_choice = s1_move_index as u8;
            new_node.s2_choice = s2_move_index as u8;
            this_pair_vec.push(new_node);
        }

        // sample a node from the new instruction list.
        // this is the node that the rollout will be done on.
        // into_boxed_slice drops the Vec's spare capacity and, more importantly,
        // makes it a type that cannot be resized, which ensures the node
        // addresses are stable for the children map keys
        let mut boxed = this_pair_vec.into_boxed_slice();
        let new_node_ptr = self.sample_node(&mut boxed, rng);
        state.apply_instructions(&(*new_node_ptr).instructions.instruction_list);

        let key = (self as *mut Node as usize, s1_move_index, s2_move_index);
        children.insert(key, boxed);
        new_node_ptr
    }

    pub unsafe fn backpropagate(&mut self, score: f32, state: &mut State) {
        self.times_visited += 1;
        if self.root {
            return;
        }

        let parent_s1_movenode =
            &mut (*self.parent).s1_options.as_mut().unwrap()[self.s1_choice as usize];
        parent_s1_movenode.total_score += score;
        parent_s1_movenode.visits += 1;

        let parent_s2_movenode =
            &mut (*self.parent).s2_options.as_mut().unwrap()[self.s2_choice as usize];
        parent_s2_movenode.total_score += 1.0 - score;
        parent_s2_movenode.visits += 1;

        state.reverse_instructions(&self.instructions.instruction_list);
        (*self.parent).backpropagate(score, state);
    }

    pub fn rollout(&mut self, state: &mut State, root_eval: &f32, rng: &mut impl Rng) -> f32 {
        let playout_turns = PLAYOUT_TURNS.load(Ordering::Relaxed);
        let mut battle_is_over = state.battle_is_over();
        // play forward with uniformly random moves, remembering every branch we
        // applied so the state can be restored in reverse order afterwards — the
        // tree above us relies on this node leaving the state exactly as it found it
        let mut applied: Vec<StateInstructions> = Vec::new();
        let mut turns = 0;
        while battle_is_over == 0.0 && turns < playout_turns {
            let (s1_options, s2_options) = state.get_all_options();
            if s1_options.is_empty() || s2_options.is_empty() {
                break;
            }
            let s1_move = s1_options[rng.random_range(0..s1_options.len())];
            let s2_move = s2_options[rng.random_range(0..s2_options.len())];
            let branches = generate_instructions_from_move_pair(state, &s1_move, &s2_move, false);
            if branches.is_empty() {
                break;
            }
            let total: f32 = branches.iter().map(|b| b.percentage.max(0.0)).sum();
            let mut threshold = rng.random_range(0.0..total.max(f32::MIN_POSITIVE));
            let mut chosen = branches.len() - 1;
            for (index, branch) in branches.iter().enumerate() {
                threshold -= branch.percentage.max(0.0);
                if threshold <= 0.0 {
                    chosen = index;
                    break;
                }
            }
            let branch = branches[chosen].clone();
            state.apply_instructions(&branch.instruction_list);
            applied.push(branch);
            battle_is_over = state.battle_is_over();
            turns += 1;
        }
        let score = if battle_is_over == 0.0 {
            sigmoid(leaf(state) - root_eval)
        } else if battle_is_over == -1.0 {
            0.0
        } else {
            battle_is_over
        };
        for branch in applied.iter().rev() {
            state.reverse_instructions(&branch.instruction_list);
        }
        score
    }
}

#[derive(Debug)]
pub struct MoveNode {
    pub move_choice: MoveChoice,
    pub total_score: f32,
    pub visits: u32,
}

impl MoveNode {
    pub fn ucb1(&self, parent_visits: u32) -> f32 {
        if self.visits == 0 {
            return f32::INFINITY;
        }
        let score = (self.total_score / self.visits as f32)
            + (2.0 * (parent_visits as f32).ln() / self.visits as f32).sqrt();
        score
    }
    pub fn average_score(&self) -> f32 {
        let score = self.total_score / self.visits as f32;
        score
    }
}

#[derive(Clone)]
pub struct MctsSideResult {
    pub move_choice: MoveChoice,
    pub total_score: f32,
    pub visits: u32,
}

impl MctsSideResult {
    pub fn average_score(&self) -> f32 {
        if self.visits == 0 {
            return 0.0;
        }
        let score = self.total_score / self.visits as f32;
        score
    }
}

pub struct MctsResult {
    pub s1: Vec<MctsSideResult>,
    pub s2: Vec<MctsSideResult>,
    pub iteration_count: u32,
}

fn mcts_iteration(
    root_node: &mut Node,
    state: &mut State,
    root_eval: &f32,
    children: &mut HashMap<(usize, usize, usize), Box<[Node]>>,
    rng: &mut impl Rng,
) {
    let (mut new_node, s1_move, s2_move) = unsafe { root_node.selection(state, children, rng) };
    new_node = unsafe { (*new_node).expand(state, s1_move, s2_move, children, rng) };
    let rollout_result = unsafe { (*new_node).rollout(state, root_eval, rng) };
    unsafe { (*new_node).backpropagate(rollout_result, state) }
}

enum SearchLimit {
    Time(Duration),
    Iterations(u32),
}

fn run_mcts_loop(
    root_node: &mut Node,
    state: &mut State,
    root_eval: &f32,
    children: &mut HashMap<(usize, usize, usize), Box<[Node]>>,
    limit: SearchLimit,
    seed: u64,
) {
    // A zero seed keeps the original entropy-seeded generator. Anything else
    // makes the whole search reproducible, which is what an offline A/B needs:
    // without it, the same configuration searched twice disagrees with itself by
    // more than the effects being measured, and every "paired" comparison is
    // really two independent draws.
    if seed == 0 {
        run_mcts_loop_with(root_node, state, root_eval, children, limit, &mut rng())
    } else {
        run_mcts_loop_with(
            root_node,
            state,
            root_eval,
            children,
            limit,
            &mut SmallRng::seed_from_u64(seed),
        )
    }
}

fn run_mcts_loop_with(
    root_node: &mut Node,
    state: &mut State,
    root_eval: &f32,
    children: &mut HashMap<(usize, usize, usize), Box<[Node]>>,
    limit: SearchLimit,
    rng: &mut impl Rng,
) {
    let start_time = std::time::Instant::now();
    loop {
        for _ in 0..1000 {
            mcts_iteration(root_node, state, root_eval, children, rng);
        }
        if root_node.times_visited >= 10_000_000 {
            break;
        }
        match limit {
            SearchLimit::Time(max_time) => {
                if start_time.elapsed() >= max_time {
                    break;
                }
            }
            SearchLimit::Iterations(n) => {
                if root_node.times_visited >= n {
                    break;
                }
            }
        }
    }
}

pub fn perform_mcts(
    state: &mut State,
    side_one_options: Vec<MoveChoice>,
    side_two_options: Vec<MoveChoice>,
    max_time: Duration,
    max_iterations: u32,
) -> MctsResult {
    perform_mcts_seeded(
        state,
        side_one_options,
        side_two_options,
        max_time,
        max_iterations,
        0,
    )
}

/// `perform_mcts` with a reproducible generator; a seed of 0 means entropy.
pub fn perform_mcts_seeded(
    state: &mut State,
    side_one_options: Vec<MoveChoice>,
    side_two_options: Vec<MoveChoice>,
    max_time: Duration,
    max_iterations: u32,
    seed: u64,
) -> MctsResult {
    let mut root_node = Node::new();
    unsafe {
        root_node.populate(side_one_options, side_two_options);
    }
    root_node.root = true;
    let mut children: HashMap<(usize, usize, usize), Box<[Node]>> = HashMap::new();

    let root_eval = leaf(state);
    let search_limit = if max_iterations > 0 {
        SearchLimit::Iterations(max_iterations)
    } else {
        SearchLimit::Time(max_time)
    };
    run_mcts_loop(
        &mut root_node,
        state,
        &root_eval,
        &mut children,
        search_limit,
        seed,
    );

    let result = MctsResult {
        s1: root_node
            .s1_options
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| MctsSideResult {
                move_choice: v.move_choice.clone(),
                total_score: v.total_score,
                visits: v.visits,
            })
            .collect(),
        s2: root_node
            .s2_options
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| MctsSideResult {
                move_choice: v.move_choice.clone(),
                total_score: v.total_score,
                visits: v.visits,
            })
            .collect(),
        iteration_count: root_node.times_visited,
    };

    result
}
