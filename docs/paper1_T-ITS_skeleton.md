# PAPER 1 SKELETON — IEEE T-ITS / TRC (Q1) — COMPLETE BUILD
<!-- Fill-in-the-blank writing guide. Conventions:
     [INSERT: ...]            = prose you write (often needs campaign results)
     [EQ-n: ...]              = numbered equation to typeset
     [TABLE-n: ...]           = table to build from campaign data
     [FIG-n: ...]             = figure (PDF, IEEE column width, font >= 8 pt)
     [NEW]                    = section/idea added in this complete build
     [REVISED]                = materially changed from the first draft
     [REVIEWER ATTACK RISK]   = a claim a referee will probe; obey the guard
     ⚠ REVIEWER TRAP          = phrasing/claim that gets papers rejected
     Target: 12–14 pages two-column incl. references.

     BUILD NOTE (2026-06-13): hardened after a full T-ITS/TRC publication audit.
     Central theme UNCHANGED — deployability-constrained RL (camera-observable
     state + reward). NOT a MARL-architecture paper, NOT a computer-vision paper.
     The audit's three load-bearing fixes are baked in:
       (1) one minimal feature-estimator validation (III-E) so "camera-observable"
           is demonstrated, not asserted — and it supplies IV-C's noise numbers;
       (2) precise claim language (no "sim-to-real" as a verb, no "first",
           "information-restriction cost" not "cost of deployability");
       (3) calibration provenance for the Vietnamese mix + saturation flow,
           with count-vs-PCU stated and the moto-share sweep as the defense. -->

---

## TITLE

**INSTRUCTION:** ≤ 15 words. Must contain the constraint (camera-observable), the method class (multi-agent RL), and the setting (mixed traffic). No "novel"/"deep"/"sim-to-real" in the title.

> **SELECTED:** *Camera-Observable Multi-Agent Reinforcement Learning for Traffic Signal Control in Mixed Motorcycle-Dominant Traffic*

[REVIEWER ATTACK RISK] Do NOT add "sim-to-real" to the title. The paper *reduces the gap by construction and measures its cost in simulation*; it does not cross the gap with field results. The title must not promise more than VI delivers.

---

## ABSTRACT (150–250 words, single paragraph, no citations, no undefined acronyms) [REVISED]

**INSTRUCTION:** Exactly 6 sentences, in this order. Write each, then merge.

1. **Problem:** RL-based TSC trains on simulator-privileged state (exact queues, per-vehicle waits, network arrivals) that no roadside camera provides, creating a deployment gap. [INSERT]
2. **Approach:** We constrain BOTH state and reward to quantities a commodity traffic camera can estimate (YOLOv11 + ByteTrack ROIs): a 26-dim per-agent observation with per-approach aggregation and signed group-mean pressure, and an Extended-PRESSLIGHT reward with a mean-of-squares anti-starvation queue penalty, a signed pressure term, and a local ROI-exit throughput term. [INSERT]
3. **Method:** Parameter-shared MAPPO with a centralized per-agent-value-head critic and per-agent GAE, trained in SUMO's sublane model under a motorcycle-dominant vehicle mix. [INSERT]
4. **Headline result:** [INSERT: "On a 5-junction corridor and a 16-junction grid under time-varying peak demand, the camera-constrained policy reduces mean travel time by X% vs Webster and Y% vs max-pressure, retaining Z% of a privileged-state policy's performance."] ← from TABLE-4/5 + TABLE-6.
5. **Robustness/cost:** [INSERT: under a sensing-noise model whose magnitudes are measured from the detector stack, performance degrades gracefully out to 2× the calibrated envelope, while the information-restriction cost vs a privileged policy is W%.] ← from FIG-6 + VI-C.
6. **Significance:** [REVISED] quantifying the cost of restricting RL-TSC to camera-computable signals shows such control is practical without simulator-privileged sensing; closed-loop field validation is reported in a companion study. [INSERT]

**Index Terms:** traffic signal control, multi-agent reinforcement learning, deployability, mixed traffic, sublane model, SUMO. [REVISED — "sim-to-real" removed]

---

## I. INTRODUCTION (~1.25 pages)

### I-A. Motivation
**INSTRUCTION:** 4–6 sentences. Urban congestion cost → adaptive TSC promise → RL beats fixed-time *in simulation*. Hook: almost none of it is deployable on existing roadside sensing. [INSERT]

### I-B. The observation sim-to-real gap (THE gap paragraph) [REVISED]
**INSTRUCTION:** 6–8 sentences. Define precisely what prior RL-TSC consumes that a camera cannot give: `getLastStepHaltingNumber`, per-vehicle accumulated waiting, network-wide arrivals. Distinguish this *observation/reward* gap from the *dynamics* gap studied in robotics transfer. One sentence on why detector-based (loop) TSC only partially solves it (point sensors, no composition/speed profile). [REVISED] State the scope honestly in one sentence: *this paper closes the gap by design and quantifies its cost in simulation; closed-loop field operation is the companion study.* [INSERT]
[REVIEWER ATTACK RISK] ⚠ Do not claim "first". Claim the pairing of a camera-constrained **state AND reward** is under-addressed; cite the closest 3 works honestly (II-B).

### I-C. Challenges
**INSTRUCTION:** The 3 technical challenges the design answers: (i) camera ROIs measure approach-level aggregates, not per-lane truth; (ii) reward terms like per-vehicle waiting are unmeasurable at deployment; (iii) motorcycle-dominant non-lane-based flow breaks car-centric, lane-indexed state designs. [INSERT]

### I-D. Contributions (4 items, each verifiable in the paper) [REVISED]
- **C1 (State):** A 26-dim vision-proxy observation — per-approach aggregation of 5 bounded-[0,1] features (queue, occupancy, speed, motorcycle share, heavy-vehicle share) + signed group-mean pressure `0.5(1+q̄_A−q̄_B)`; **every feature has a documented camera estimator (Table 1) whose recovery is validated against ground truth (III-E).** [INSERT]
- **C2 (Reward):** An Extended-PRESSLIGHT reward computable from the same proxies — mean-of-squares queue penalty (anti-starvation, Jensen), signed pressure in [−1,1], and a local ROI-exit throughput term whose SUMO computation is *semantically identical to the deployable ByteTrack measurement*. [INSERT]
- **C3 (Cost of restriction):** [REVISED] [REVIEWER ATTACK RISK] A quantification of the **information-restriction cost** — privileged-state vs camera-proxy performance — and of degradation under a **measured** sensing-noise model. Do NOT call this "the cost of deployability" unqualified: it is the cost of restricting information plus modeled sensing noise; real-detector domain shift is bounded by III-E and discussed in VII, not claimed as measured here.
- **C4 (Mixed-traffic evaluation):** Multi-seed evaluation on a 5-junction corridor and a 4×4 grid under SUMO's sublane model with a motorcycle-dominant vehicle mix, time-varying demand, against Webster, SUMO-actuated, max-pressure, and IPPO, with sublane-vs-lane cross-evaluation showing the modeling choice changes outcomes. [INSERT]

### I-E. Paper organization
**INSTRUCTION:** 2 mechanical sentences. [INSERT]

---

## II. RELATED WORK (~1 page, 3 subsections + positioning table)

### II-A. RL for traffic signal control
**INSTRUCTION:** 1 chronological paragraph: DQN-era single junction → pressure-based (PressLight, MPLight, CoLight) → MARL/CTDE (MAPPO, IPPO). End: all consume privileged simulator state. Cite ≥ 8. [INSERT]

### II-B. Deployability- and sensing-constrained TSC (direct competitors — be exhaustive) [REVISED]
**INSTRUCTION:** 8–10 sentences over three threads: (i) loop/detector-state TSC; (ii) camera/CV-state RL-TSC, 2022–2026 (search "vision-based traffic signal control reinforcement learning"); (iii) sim-to-real TSC transfer. For each closest work: one sentence on what they did + one on what they lack (usually: state-only constraint with an unconstrained privileged reward, no noise-robustness, single intersection, no mixed/sublane traffic). [INSERT]
[REVIEWER ATTACK RISK] ⚠ A referee will likely be an author of one of these. The honest differentiator is the **paired state+reward** constraint and the measured restriction cost — lead with that, not with primacy. Cite generously.

### II-C. Mixed and motorcycle-dominant traffic modeling [REVISED]
**INSTRUCTION:** SUMO sublane model; PCU concepts; field studies of Vietnamese/Indian/Indonesian heterogeneous traffic; the near-absence of RL-TSC under sublane dynamics. [REVISED] State the vehicle-composition citation here (the source for the 73/17/7/3 mix used in V-A) and note explicitly that motorcycle share is reported **by vehicle count**, not PCU. [INSERT]

**[TABLE-1: Positioning matrix.** [REVISED] Rows = 6–8 closest works + ours. Columns: State source (privileged / loop / camera-feasible) | **Reward source (privileged / camera-feasible)** ← the discriminating column | Multi-junction | Mixed traffic / sublane | Noise robustness | Restriction-cost measured. Ours = ✓ where true; leave honest gaps (no field results; ≤16 junctions).]

---

## III. PROBLEM FORMULATION (~0.75–1 page)

### III-A. Dec-POMDP definition
**INSTRUCTION:** Formal tuple; agents = signalized junctions; binary action (phase group A/B); 5 s decision interval, 3 s environment-inserted yellow; min-green 15 s, max-green 60 s (both selected empirically — see V-A "Signal-timing bounds"); episode = 1080 steps (5400 s).

**[EQ-1: Dec-POMDP tuple ⟨N, S, {A_i}, {O_i}, T, R, γ⟩, γ = 0.99]**

### III-B. The observation projection (formal core of the vision-proxy idea) [REVISED]
**INSTRUCTION:** Define privileged state `s_t` vs deployed observation `o_{i,t} = φ_i(s_t) + ε_{i,t}`, with φ the camera-computable projection and ε the sensing-noise process (ε = 0 in training, ε ≠ 0 at deployment). State the design rule: the policy executes strictly through φ, and **every reward term admits a φ-computable estimator**.
[REVIEWER ATTACK RISK] ⚠ Do NOT write "R factors through φ exclusively." The training-time waiting penalty reads privileged `getWaitingTime` for gradient stability and falls back to the φ-computable halted-queue proxy; the fully-proxy reward is evaluated as an ablation (VI-F). State this in one sentence so the released code does not contradict the formalization.

**[EQ-2: o_{i,t} = φ_i(s_t) + ε_{i,t}; execution a_{i,t} ~ π_θ(·|o_{i,t}); each reward term r^(k) = g_k(φ(s),a) up to the documented waiting-term exception.]**

### III-C. Observation space (26-dim, per agent)
**INSTRUCTION:** Present TABLE-2, then 1 paragraph on per-approach aggregation (lanes of an incoming edge pooled — length-ratio features by unweighted mean, per-vehicle statistics by vehicle-count-weighted mean), which removes approach blind spots AND matches the one-ROI-per-approach camera geometry; signed group-mean pressure is group-size invariant.

**[TABLE-2: 26-dim layout.** Indices 0–19: 4 approaches × {effective_queue_norm, occupancy_norm, avg_speed_norm, motorbike_share, heavy_vehicle_share}; 20–23 phase one-hot; 24 green-timer norm; 25 signed pressure. Columns: index | feature | SUMO source | camera estimator (YOLOv11+ByteTrack) | alignment quality (H/M/L) | validated in III-E?]

**[EQ-3: pressure_norm = clip(½(1 + q̄_A − q̄_B), 0, 1), q̄_G = (1/|G|)Σ_{l∈G} q_l]**

[REVIEWER ATTACK RISK] ⚠ Do not write "perfectly maps to camera limits". Write "each feature admits a camera estimator with residual error measured in III-E and propagated in VI-E"; alignment is Medium for most features and you will be asked for magnitudes.

### III-D. Action space & signal constraints
**INSTRUCTION:** 4 sentences. Binary phase-group action; environment-enforced yellow and min-green; justify the two-phase simplification by the legacy hardware of the targeted controllers in developing-city deployments; defer multi-phase/protected-turn extensions to future physical integration. [INSERT]

### III-E. [NEW] Feature-estimator validation (bounded, in-scope)
**INSTRUCTION:** This is the single highest-value addition from the audit; it converts "camera-observable" from asserted to demonstrated and supplies IV-C's noise magnitudes. Keep it small and scoped — NOT a deployment pipeline, NOT a CV contribution.
- Run the same YOLOv11 + ByteTrack stack on a bounded set of frames **with ground truth** — either (a) one annotated mixed-traffic clip, or (b) SUMO-rendered frames where the true feature is known by construction.
- Report estimated-vs-ground-truth for the 3 most error-prone features (effective queue, occupancy, average speed) and the two class shares.
- The per-feature error statistics measured here are exactly the parameters of the IV-C noise model — cite III-E as their source.
**[FIG-1a: estimated vs ground-truth scatter + per-feature error (bias, σ) for the 5 lane features.]**
[REVIEWER ATTACK RISK] ⚠ Without this, the title's "camera-observable" is unvalidated and C1/C2 are assertions. Keep claims bounded: "feature recovery on N frames in setting X", deployment generality deferred to the companion study (VII-2).

---

## IV. METHODOLOGY (~2 pages)

### IV-A. Extended-PRESSLIGHT vision-proxy reward (revision 1.2.0)
**INSTRUCTION:** Open with the design rule: every term factors through φ (III-B), with the documented waiting-term exception. One sub-paragraph + equation per term.

**[EQ-4: full reward]**
R_i(t) = w_q·(1/n)Σ_l q_l² + w_p·p_i(t) + w_T·T_i(t) + w_s·𝟙[switch] + w_v·max(τ−v̄,0) + w_w·w̄(t),
clipped to [−2.0, +1.5]; weights (−1.0, −0.5, +1.0, −0.1, −0.2, −0.3); raw range [−1.94, +1.50].

1. **Anti-starvation queue penalty.** Defect of mean-then-square: a saturated lane among empty ones vanishes into the average. **[EQ-5: Jensen (1/n)Σq_l² ≥ ((1/n)Σq_l)², equality iff uniform]** + 2 sentences: the gap equals the queue variance, so the penalty prices imbalance; worked micro-example q=(1,0,0,0) scores 0.25 vs 0.0625 (4× on concentration). [INSERT]
2. **Signed pressure.** **[EQ-6: p_i = (Σ_{l∈red} q_l − Σ_{l∈green} q_l)/n_i ∈ [−1,1]]** + 3 sentences: one-sided max(·,0) gives zero gradient when allocation is correct; the signed form also rewards serving the congested side. [REVIEWER ATTACK RISK] ⚠ Do NOT claim max-pressure stability theory transfers — normalization + clipping break its assumptions; write "pressure-inspired", claim no throughput-optimality guarantee. [INSERT]
3. **Local ROI-exit throughput.** **[EQ-7: T_i = clip(|prev_ids(L_i) \ cur_ids(L_i)|/κ, 0, 1), κ = 20]** + 3 sentences: per-agent credit (no cross-agent split of a network-wide count); the SUMO computation (per-lane vehicle-ID set difference) is *semantically identical* to ByteTrack track-IDs exiting the ROI — the reward itself is the deployable measurement. This equivalence is C2's strongest point; state it explicitly. [INSERT]
4. **Remaining terms** (switch; low-speed; waiting with queue-proxy fallback): 1 sentence each. Restate the deployment caveat: **reward is computed only in training**; the waiting term's privileged read is the one φ-exception (III-B), ablated in VI-F. [INSERT]

### IV-B. MAPPO architecture (v3) — standard method, stated honestly
**INSTRUCTION:** [REVIEWER ATTACK RISK] Frame as a *standard* CTDE method, NOT a contribution — one paragraph, no novelty claims. Parameter-shared actor π_θ(a|o_i) over 26-dim local obs; centralized critic V_ψ with **one value head per agent** over the concatenated proxy state; per-agent GAE(λ=0.95) on per-agent rewards; advantages standardized jointly across agents; PPO clip 0.2, target-KL 0.015 early stop, value-loss clipping 0.2, entropy 0.01→0.001; Welford obs/state normalization serialized in checkpoints.

**[EQ-8: per-agent GAE]  [EQ-9: clipped surrogate + per-head clipped value loss]**
**[FIG-1: architecture — camera ROIs → φ → shared actor (deployment path, solid) | concat → multi-head critic (training only, dashed). The figure must show the train/deploy asymmetry: only the actor + normalization stats cross the deployment boundary.]**
**[ALGORITHM-1: rollout (5 s decision step, yellow insertion, min/max-green), per-agent GAE, joint normalization, PPO update with KL early stop, checkpoint.]**

### IV-C. Sensing-noise model (for VI-E) [REVISED]
**INSTRUCTION:** Define ε per feature; **parameters are the error statistics measured in III-E** (or, where III-E cannot cover a parameter, cited detector/tracker benchmarks — state provenance per parameter). Queue: multiplicative N(0, σ_q²) (px-per-meter calibration). Speed: additive N(0, σ_v²) concentrated below 3 m/s (ByteTrack displacement noise). Class shares: confusion flip rate ρ_c (YOLO). Occupancy: perspective-residual bias b_occ. Structural modes: single-camera dropout (zero one approach), one-step pipeline delay. Sweep the whole envelope over {0, 0.5×, 1×, 2×}.
**[TABLE-3a: noise parameters with per-parameter provenance (III-E measured / cited).]**
[REVIEWER ATTACK RISK] ⚠ Do not present any noise magnitude as "calibrated" unless its source (III-E or a citation) is named. The sweep is the real defense: conclusions are about the *shape* across scales, so the exact 1× anchor is not load-bearing — say so.

---

## V. EXPERIMENTAL SETUP (~1.5 pages)

### V-A. Simulation environment & mixed-traffic calibration [REVISED]
**INSTRUCTION:** SUMO [INSERT pinned version], sublane model (lateral resolution 0.4 m). Vehicle mix and its provenance:
- **Composition:** moto/car/truck/bus = 73/17/7/3 **by vehicle count** (`vTypeDistribution`), cited to [INSERT: Vietnamese urban traffic-composition field study]; note `docs/state.md` records the published band as 60–80% motorcycle, within which 73% sits.
- [NEW] [REVIEWER ATTACK RISK] **Count vs PCU:** state explicitly that 73% is a *count* share; at motorcycle PCU ≈ 0.25–0.30 this is ≈ 38–45% of traffic *load* in PCU. Use **count share** when arguing sublane/filtering behavior (the right metric there) and **PCU** for any capacity/saturation statement — do not conflate them.
- **Sublane parameters:** `latAlignment=arbitrary`, `minGapLat=0.12 m` for motorcycles; give the lane-sharing arithmetic (lane width / moto width → k motorcycles abreast) and cite the field source for these vType parameters, not only the share.
- **Saturation flow:** report simulated per-lane saturation flow [INSERT measured PCU/h] vs field range [INSERT cite], and confirm it is consistent with the 73%-by-count mix.
[REVIEWER ATTACK RISK] ⚠ Without the citation + saturation number + count/PCU clarity, a TRC referee treats "Vietnamese mixed traffic" as decoration. The **moto-share sweep in VI-D {50/73/90%} is the formal defense of the central 73% value** — reference it here so 73% reads as a sweep center, not a load-bearing point estimate.

#### Signal-timing bounds (min-green, max-green) — design-selection pilot
**INSTRUCTION:** 3–4 sentences. Min-green raised 10→15 s (= 3 decision intervals) so a served phase clears a meaningful platoon before the policy may switch (10 s let phases flip before motorcycles cleared the stop line). Max-green fixed at 60 s by a single-seed selection pilot on the larger network (N3 grid, seed 42, 100 k steps, min-green 15, identical reward 1.2.0 and demand), greedy policy on held-out demand. The 60 s bound dominates 90 s on every primary metric, largest on the starvation-sensitive P95 tail — consistent with the anti-starvation reward (longer max-green starves the cross-direction).
[REVIEWER ATTACK RISK] ⚠ State plainly this is a **single-seed design-selection pilot, not a multi-seed ablation**; the 15–25% gaps + the principled P95 mechanism justify it as a configuration choice; offer to promote to 5 seeds if requested. Do NOT present as a contribution.

**[TABLE-V-A1: max-green selection pilot (N3 grid, seed 42, 100 k steps, held-out eval).** Lower is better; bold the winner.]

| max-green | Mean travel time (s) | Mean waiting (s) | P95 waiting (s) |
|---|---|---|---|
| **60 s (selected)** | **311.1** | **135.8** | **504** |
| 90 s | 367.0 | 173.6 | 669 |
| Δ (60 vs 90) | −15.2 % | −21.8 % | −24.7 % |

Note: max-green also sets the `green_timer_norm` scale (`timer / max_green`), so it is a coupled obs/MDP parameter fixed before training and held identical across all arms and baselines (SUMO-actuated `maxDur` matched to 60 s for fairness).

### V-B. Networks and demand
**INSTRUCTION:** N2 = 1×5 signalized corridor (5 junctions), N3 = 4×4 grid (16 junctions); 4-way, 2 lanes/approach, `netgenerate` with program-derived phase groups (released). Demand: within-episode trapezoid, inhomogeneous Poisson **[EQ-10: piecewise-linear λ(t), knots 0/900/1500/3000/3600/4800/5400 s at 0.4/1.67/0.83 veh/s, scaled by n_entries/6]**, held-out evaluation seeds.
**[FIG-2: demand profile λ(t) + N2/N3 network diagrams side by side.]**

### V-C. Baselines
**INSTRUCTION:** One paragraph each, with implementation honesty:
- **Webster fixed-time** — measured-flow, PCU-weighted (0.3/1.0/2.0/2.5), **[EQ-11: C = (1.5L+5)/(1−Y)]**, cycle clamped [40,120] s, proportional splits — NOT an equal-split strawman.
- **SUMO actuated** — gap-based, minDur/maxDur = 15/60 matching the RL constraint (V-A).
- **Max-pressure** — same phase groups and observation pipeline as MAPPO (fair).
- **IPPO** — identical actor + budget; independent per-agent critics on local 26-dim obs only (isolates centralized-training value).
- **MAPPO-privileged** — upper bound: identical algorithm, observation augmented with exact-SUMO per-approach features (halting count, accumulated waiting, vehicle count); reward identical to the proxy arm (only the observation differs). [INSERT 1 sentence on the 38-dim privileged state.]

### V-D. Metrics & statistical protocol
**INSTRUCTION:** Primary: mean travel time, mean waiting (tripinfo, write-unfinished). Secondary: P95 waiting (starvation), throughput-served ratio, mean speed, switches/episode. Post-hoc only (never in reward): CO₂, fuel (HBEFA). Protocol: **5 training seeds × 10 evaluation route-seeds per scenario**; mean ± 95% CI over training seeds; Mann-Whitney U vs strongest baseline; Holm-Bonferroni across scenarios; Cohen's d. Seeds: 42/123/456/789/1337. [INSERT]

### V-E. Training configuration & reproducibility
**[TABLE-3: hyperparameters]** (γ 0.99, λ 0.95, clip 0.2, target-KL 0.015, lr 3e-4 linear decay, horizon 128, minibatch 256, 10 epochs, [INSERT total steps fixed by the convergence pilot], hardware, [INSERT measured steps/s]).
**INSTRUCTION:** Release sentence: code, networks, demand generators, lane-group derivations, all seeds and run configs at [INSERT repo URL], frozen at a tagged commit; obs/reward schema version stamped in every run config.

---

## VI. RESULTS (~3 pages — build this before writing anything else)

### VI-A. Main comparison
**[TABLE-4: N2 corridor — methods × {travel, waiting, P95 wait, throughput ratio, switches}, mean ± CI, bold best, † significant vs best baseline]**
**[TABLE-5: N3 grid — same columns]**
**INSTRUCTION:** 2 paragraphs; where MAPPO wins, by how much, where it does not (report honestly — actuated is often competitive off-peak). Every claim cites a cell. [INSERT]

### VI-B. Convergence
**[FIG-3: episode reward (mean ± band, 5 seeds) vs steps, N2 & N3]  [FIG-4: entropy + approx-KL curves]**
**INSTRUCTION:** 1 paragraph incl. training-cost honesty (steps/s, wall-clock). [INSERT]

### VI-C. The cost of restricting to camera-computable signals (headline for C3) [REVISED]
**[TABLE-6 / FIG-5: MAPPO-privileged vs MAPPO-proxy on all primary metrics; retained-performance % with CI]**
**INSTRUCTION:** 2 paragraphs. (1) The information-restriction cost is [INSERT]% on travel time — interpret. (2) Why the proxy holds up (which features carry the signal — link to VI-F).
[REVIEWER ATTACK RISK] ⚠ Add one sentence: both arms use clean SUMO features, so this isolates the cost of *information restriction*; the cost of *real detector error* is modeled separately in VI-E and bounded by III-E; real-world domain shift is discussed in VII, not claimed measured here. This precision is what separates an A from a downgraded C3.

### VI-D. Generalization (out-of-distribution demand + composition) [REVISED]
**INSTRUCTION:** Train on trapezoid; evaluate on held-out uniform low/medium/high/asymmetric profiles + **motorcycle-share sweep {50%, 73%, 90%}** — explicitly the formal sensitivity defense of the V-A central mix.
**[TABLE-7: OOD demand results]  [FIG-6a: performance vs moto-share]** [INSERT prose tying the flat-ish moto-share curve back to robustness of the 73% choice.]

### VI-E. Robustness to sensing noise (headline figure for C3)
**[FIG-6: degradation curves — x = noise scale {0, 0.5×, 1×, 2× measured ε}, y = mean travel time; one line per method incl. baselines (heuristics degrade too) + camera-dropout bar (one approach zeroed) + one-step-delay condition.]**
**INSTRUCTION:** 2 paragraphs. Claim: the camera-constrained policy degrades gracefully within the measured envelope while [INSERT comparative finding vs baselines]. Anchor the 1× envelope to III-E magnitudes. [INSERT]

### VI-F. Ablations (all RETRAINED, ≥ 3 seeds — never eval-time reweighting)
**[TABLE-8: reward ablations — full / no-pressure / no-throughput / queue-only / unsigned-pressure / mean-then-square-queue / fully-proxy-waiting (the III-B φ-exception removed)]**
**[TABLE-9: state & architecture ablations — minus class-shares / minus pressure-feature / per-lane-truncated (pre-fix blind-spot variant) / IPPO vs MAPPO / sublane-vs-lane cross-evaluation (train lane-based → test sublane)]**
**INSTRUCTION:** 1 short paragraph per table; which components matter most, each linked to its design argument in IV. [REVISED] The **sublane-vs-lane cross-table is the evidence that the mixed-traffic modeling changes outcomes** — without a degradation there, the mixed-traffic contribution (C4) is unsupported. The fully-proxy-waiting row closes the III-B honesty loop. [INSERT]

### VI-G. Environmental & secondary metrics
**[TABLE-10: CO₂, fuel per episode — post-hoc only]**
**INSTRUCTION:** 1 paragraph; restate these never entered training (reinforces the deployability story). [INSERT]

---

## VII. DISCUSSION & LIMITATIONS (~0.5 page — write it before Reviewer 2 does)

**INSTRUCTION:** One honest paragraph each; consistency-check against the abstract:
1. Two-phase control without protected turns; extension path. [INSERT]
2. [REVISED] Sim-only scope: noise magnitudes are measured (III-E) but there is no closed-loop field result here; explicitly scoped to the companion deployment study. Ensure the abstract/title never contradict this. [INSERT]
3. Calibration depth of the Vietnamese mix: which parameters are field-cited vs assumed; count-vs-PCU caveat restated. [INSERT]
4. Scale: ≤ 16 junctions ≠ city scale; what breaks first (shared-actor credit assignment, pressure locality). [INSERT]

---

## VIII. CONCLUSION (~0.3 page)

**INSTRUCTION:** 4 sentences, no new claims: (1) restate the paired state+reward camera-constraint formulation; (2) the 2–3 strongest numbers (restriction cost %, improvement vs Webster, noise-robustness finding); (3) one practitioner implication; (4) future work = closed-loop field deployment (companion), multi-phase, city-scale. [INSERT]

---

## ACKNOWLEDGMENT / REFERENCES / SUPPLEMENT

- **References:** 35–45; ≥ 10 from 2023–2026; T-ITS self-citations 3–5; cite every baseline and every tool (SUMO, YOLOv11, ByteTrack) and the Vietnamese composition source(s).
- **Supplementary / repo:** demand `.rou.xml` of the exact published runs (⚠ commit at camera-ready — duarouter versions are not byte-stable), best-seed checkpoint, evaluation scripts, III-E validation data + script, CHECK-suite outputs.

---

## PRE-SUBMISSION GATE (do not submit unless all ✓)

- [ ] Every number in the Abstract traceable to a table
- [ ] 5 seeds × 10 eval seeds everywhere; CI + significance marks in every results table
- [ ] Ablations retrained (no eval-time reward reweighting anywhere)
- [ ] No "perfect/first/novel" claims that II-B contradicts; no "sim-to-real" as a verb
- [ ] III-E feature-estimator validation present; its error stats == IV-C noise parameters
- [ ] V-A: saturation-flow number + Vietnamese-mix citation + count-vs-PCU sentence all present
- [ ] VI-C labeled "information-restriction cost" with the clean-features caveat sentence
- [ ] VI-C (restriction cost) and VI-E (noise robustness) BOTH present — they are the paper
- [ ] VI-F includes sublane-vs-lane cross-eval and the fully-proxy-waiting ablation
- [ ] Repo frozen at a tagged commit; schema/version stamps verified in all run configs
- [ ] Discussion VII consistent with Abstract/Title (sim-only scope stated, not buried)
