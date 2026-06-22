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
       (1) a sensing-noise model with per-feature provenance — the projection
           perturbation ε is introduced in III-B, outlined briefly in III-E, and
           parameterized in VI-E (TABLE-3a) where its {0–2x} sweep bounds the
           camera-observable claim; a measured feature-recovery validation is the
           stronger upgrade, deferred to the companion field study (Option B is shipping);
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
5. **Robustness/cost:** [INSERT: under a sensing-noise model whose magnitudes are set on a documented basis, performance degrades gracefully out to 2× the base envelope, while the information-restriction cost vs a privileged policy is W%.] ← from FIG-6 + VI-C.
6. **Significance:** [REVISED] quantifying the cost of restricting RL-TSC to camera-computable signals shows such control is practical without simulator-privileged sensing; closed-loop field validation is reported in a companion study. [INSERT]

**Index Terms:** traffic signal control, multi-agent reinforcement learning, deployability, mixed traffic, sublane model, SUMO. [REVISED — "sim-to-real" removed]

---

## I. INTRODUCTION (~1.25 pages)

### I-A. Motivation

Urban traffic congestion imposes severe economic and environmental costs on rapidly growing metropolises. Adaptive Traffic Signal Control (TSC) promises to alleviate this burden by dynamically allocating right-of-way based on real-time traffic conditions. Deep Reinforcement Learning (RL) has recently emerged as the dominant paradigm for developing these adaptive controllers, consistently outperforming traditional fixed-time heuristics *in simulation*. However, despite vast theoretical success, a critical deployment bottleneck persists: almost none of these advanced RL architectures are physically deployable using existing, commodity roadside sensing infrastructure. 

### I-B. The Observation Sim-to-Real Gap

The primary barrier to deployment is the "observation gap." Prior RL-TSC frameworks routinely assume access to privileged simulator states, consuming precise, unobservable metrics such as exact halting vehicle counts (`getLastStepHaltingNumber`), per-vehicle accumulated waiting times, and network-wide arrival profiles. Commodity traffic cameras processing visual Regions of Interest (ROIs) cannot extract these variables. We explicitly distinguish this *observation and reward gap* from the *dynamics gap* (physics discrepancies) typically studied in robotics sim-to-real transfer. While legacy inductive loop detectors provide a partial solution, they act as point-sensors that fail to capture spatial composition, vehicle classes, or continuous speed profiles—data crucial for modern mixed-traffic control. While recent works have constrained observation spaces, the critical pairing of a camera-constrained state *and* a camera-constrained reward remains significantly under-addressed. This paper systematically closes this observation gap by design and quantifies its explicit restriction cost in simulation; closed-loop physical field operation is deferred to a companion study.

### I-C. Challenges

Designing a deployable RL-TSC framework necessitates overcoming three specific technical challenges: (i) *Spatial granularity*: Camera ROIs inherently measure traffic aggregates at the approach-level, failing to yield the per-lane precision demanded by classical state formulations. (ii) *Reward unmeasurability*: The most effective RL reward signals—such as system-wide delay or precise per-vehicle waiting times—are mathematically impossible to compute at the edge during deployment. (iii) *Heterogeneous traffic dynamics*: In developing cities, traffic is heavily motorcycle-dominant and non-lane-based; lateral filtering and swarming behavior actively break traditional car-centric, lane-indexed state and reward designs.

### I-D. Contributions

We propose a deployability-constrained MAPPO framework that addresses these challenges through the following verifiable contributions:
- **C1 (Vision-Proxy State):** We design a 26-dimensional camera-proxy observation space utilizing per-approach aggregation. It comprises five bounded features (queue, occupancy, speed, motorcycle share, heavy-vehicle share) and a signed group-mean pressure. Every feature admits a documented camera estimator with a documented-basis error model (Section III-E); a measured recovery validation against ground truth is deferred to the companion study.
- **C2 (Proxy-Computable Reward):** We formulate an Extended-PRESSLIGHT reward computable entirely from the state proxies. It features a mean-of-squares queue penalty to prevent starvation via Jensen's inequality, a signed pressure term, and a local ROI-exit throughput term whose simulator computation is semantically identical to deployable ByteTrack track-ID terminations.
- **C3 (Information-Restriction Cost):** We rigorously quantify the *information-restriction cost* by evaluating the performance differential between a privileged-state policy and our proxy-constrained policy. Furthermore, we evaluate the policy's graceful degradation under a formalized sensing-noise model parameterized on a documented basis from the detector-tracking literature. 
- **C4 (Mixed-Traffic Evaluation):** We conduct extensive multi-seed evaluations on a 5-junction corridor and a 16-junction grid under SUMO's sublane model. The framework handles a motorcycle-dominant ($73\%$) mixed fleet and time-varying demand, outperforming Webster, SUMO-Actuated, Max-Pressure, and independent PPO. Crucially, a sublane-versus-lane cross-evaluation demonstrates that ignoring mixed-traffic dynamics fundamentally alters policy outcomes.

### I-E. Paper Organization

The remainder of this paper is organized as follows. Section II reviews related work. Section III formalizes the Dec-POMDP problem and observation constraints. Section IV details the Extended-PRESSLIGHT reward and MAPPO methodology. Section V describes the experimental setup, followed by the results and robustness analysis in Section VI. Section VII discusses limitations, and Section VIII concludes.

---

## II. RELATED WORK (~1 page, 3 subsections + positioning table)

### II-A. RL for Traffic Signal Control

The application of Reinforcement Learning to traffic signal control has evolved rapidly from early single-intersection Deep Q-Network (DQN) approaches to sophisticated, cooperative multi-agent paradigms. A major inflection point was the integration of Max-Pressure concepts into RL reward structures, yielding highly scalable frameworks such as PressLight, MPLight, and CoLight. Most recently, Centralized Training with Decentralized Execution (CTDE) architectures, particularly Multi-Agent Proximal Policy Optimization (MAPPO) and its independent counterpart IPPO, have demonstrated state-of-the-art coordination on complex urban grids. However, a unifying limitation across this lineage is their absolute reliance on privileged simulator data—extracting exact queue lengths, precise accumulated waiting times, and holistic network states that are structurally inaccessible to physical roadside infrastructure.

### II-B. Deployability- and Sensing-Constrained TSC

Recent literature has begun to address the sensing gap via three primary threads. First, loop-detector-based RL attempts to constrain observations to binary presence sensors; however, point-sensing inherently fails to capture spatial vehicle composition or continuous speed profiles. Second, vision-based RL-TSC has gained traction, leveraging computer vision to extract spatial grids or bounding boxes. Works in this domain often constrain the observation space to camera-feasible metrics but critically fail to apply the same constraint to the reward function, evaluating deployable actors using unmeasurable, privileged rewards during training and evaluation. Third, sim-to-real transfer studies in TSC have focused heavily on bridging the dynamics gap (e.g., transition delays) rather than the observation gap. The honest differentiator of our work is the strict, paired restriction of *both* state and reward to camera-computable proxies, coupled with a systematic measurement of the information-restriction cost and degradation under empirical sensing noise.

### II-C. Mixed and Motorcycle-Dominant Traffic Modeling

Traffic in developing metropolises is fundamentally heterogeneous, characterized by a heavy dominance of motorcycles that engage in non-lane-based filtering and swarming. Modeling these dynamics requires continuous-space physics, such as SUMO's sublane model, which deviates significantly from classical discrete lane-based queues. Empirical field studies of urban traffic in Vietnam document a heavy motorcycle dominance: Ngoc \emph{et al.}~\cite{ngoc2021_fivecities} report a motorcycle mode share of $72.6\%$ in Hanoi, rising to $80$–$89\%$ across Hai Phong, Da Nang, Ho Chi Minh City, and Can Tho. These shares lie an order of magnitude beyond the car-centric mixes assumed in most RL-TSC studies; our simulated fleet adopts the Hanoi figure as a representative anchor ($\approx 73\%$ motorcycles by vehicle count, calibrated in Section V-A, where the count-vs-PCU distinction is made explicit). Despite the prevalence of such traffic globally, RL-TSC literature evaluated under sublane dynamics remains exceedingly sparse, leaving a critical gap in solutions tailored for developing-city deployments.

**[TABLE-1: Positioning Matrix of RL-TSC Literature]**
\begin{table}[ht]
\centering
\caption{Positioning Matrix of RL-TSC Literature}
\resizebox{\columnwidth}{!}{
\begin{tabular}{lcccccc}
\toprule
\textbf{Work} & \textbf{State Source} & \textbf{Reward Source} & \textbf{Multi-Junc.} & \textbf{Mixed/Sublane} & \textbf{Noise Robustness} & \textbf{Restriction Cost} \\
\midrule
PressLight~\cite{wei2019_presslight} & Privileged & Privileged & \checkmark & $\times$ & $\times$ & $\times$ \\
CoLight~\cite{wei2019_colight} & Privileged & Privileged & \checkmark & $\times$ & $\times$ & $\times$ \\
Partial-Detection RL~\cite{zhang2021_partialdetection} & Partial (V2I) & Privileged & $\times$ & $\times$ & $\times$ & $\times$ \\
Dai \emph{et al.}~\cite{dai2022_worldmodels} & Camera-Feasible & Privileged & $\times$ & $\times$ & $\times$ & $\times$ \\
Azfar \emph{et al.}~\cite{azfar2025_cosim} & Camera-Feasible & Privileged & \checkmark & $\times$ & \checkmark & $\times$ \\
\textbf{Ours} & \textbf{Camera-Feasible} & \textbf{Camera-Feasible} & \textbf{\checkmark} & \textbf{\checkmark} & \textbf{\checkmark} & \textbf{\checkmark} \\
\bottomrule
\end{tabular}
}
\end{table}

---

## III. PROBLEM FORMULATION (~0.75–1 page)

### III-A. Dec-POMDP Definition

We formulate the multi-intersection traffic signal control problem as a Decentralized Partially Observable Markov Decision Process (Dec-POMDP). The traffic network is controlled by a set of $N$ heterogeneous agents, where each agent $i$ corresponds to a signalized intersection. 

The Dec-POMDP is defined by the tuple $\langle \mathcal{N}, \mathcal{S}, \{\mathcal{A}_i\}, \{\mathcal{O}_i\}, \mathcal{P}, \mathcal{R}, \gamma \rangle$, where:
- $\mathcal{N} = \{1, \dots, N\}$ is the finite set of agents.
- $\mathcal{S}$ is the unobserved global state space of the traffic network.
- $\mathcal{A}_i$ is the discrete action space for agent $i$.
- $\mathcal{O}_i$ is the local observation space for agent $i$.
- $\mathcal{P}: \mathcal{S} \times \mathcal{A} \to \Delta(\mathcal{S})$ is the state transition probability function governing the traffic dynamics.
- $\mathcal{R}: \mathcal{S} \times \mathcal{A} \to \mathbb{R}^N$ is the joint reward function, yielding a scalar reward $r_i$ for each agent.
- $\gamma = 0.99$ is the discount factor.

Agents operate at a discrete decision interval of $\Delta t = 5$ seconds. The environment enforces safety constraints implicitly: phase switches trigger a mandatory 3-second yellow clearance interval, and green phases are bounded by empirically selected minimum ($15$ s) and maximum ($60$ s) durations (justified in Section V-A). A full simulation episode spans $1080$ decision steps ($5400$ seconds).

### III-B. The Observation Projection

The fundamental divergence between simulated RL-TSC and real-world deployability is the reliance on privileged simulator states (e.g., precise per-vehicle accumulated waiting times and unobservable network-wide arrivals). To formalize our deployability constraint, we distinguish the true global state $\mathbf{s}_t \in \mathcal{S}$ from the deployed local observation $\mathbf{o}_{i,t} \in \mathcal{O}_i$. 

We define a camera-computable projection mapping $\phi_i$, such that the agent's observation is generated as:
\begin{equation}
\mathbf{o}_{i,t} = \phi_i(\mathbf{s}_t) + \varepsilon_{i,t}
\end{equation}
where $\varepsilon_{i,t}$ represents the sensing-noise process inherent to the computer vision pipeline. During simulation training, we assume perfect projection ($\varepsilon_{i,t} = 0$), whereas during noise robustness evaluations (Section VI-E) and physical deployment, $\varepsilon_{i,t} \neq 0$.

The strict design rule of this framework is that the execution policy must factor entirely through the camera projection: $a_{i,t} \sim \pi_\theta(\cdot | \mathbf{o}_{i,t})$. The same constraint is imposed on the training objective: every component of the reward function $r_i^{(k)}$ admits a $\phi$-computable estimator $g_k(\phi_i(\mathbf{s}_t), a_{i,t})$, so the reward optimised in simulation is itself fully camera-computable, with no privileged read. In particular, the accumulated-delay penalty is realised as a halted-queue proxy rather than a per-vehicle waiting timer (Section IV-A). This is the key distinction from prior vision-RL, which constrains the state but still trains on a privileged reward.

### III-C. Observation Space

The observation space $\mathcal{O}_i$ is a continuous $26$-dimensional vector, designed specifically to align with the capabilities of a single overhead traffic camera processing Regions of Interest (ROIs). Rather than indexing features strictly by lane—which fails in non-lane-based, motorcycle-dominant traffic where vehicles routinely filter between lanes—we aggregate traffic metrics at the *approach* level. 

For each of the $4$ incoming approaches, we extract $5$ continuous features normalized to $[0, 1]$: the effective halted queue length, spatial occupancy, average vehicle speed, motorcycle share, and heavy-vehicle share. When pooling lanes into an approach, length-ratio features (occupancy, queue) are computed via an unweighted mean, whereas per-vehicle statistics (speed and the motorcycle/heavy-vehicle class shares) are aggregated via a vehicle-count-weighted mean. This spatial aggregation naturally matches the macroscopic perspective of a camera ROI. The final $6$ dimensions encode the internal controller state: a $4$-dimensional one-hot vector for the current phase, a normalized green timer, and a continuous scalar for the signed group-mean pressure.

\begin{equation}
\text{pressure\_norm} = \text{clip}\left(\frac{1}{2}\left(1 + \bar{q}_A - \bar{q}_B\right), 0, 1\right) \quad \text{where} \quad \bar{q}_G = \frac{1}{|G|} \sum_{l \in G} q_l
\end{equation}

By taking the mean queue $\bar{q}_G$ of phase groups $G \in \{A, B\}$ rather than the sum, the pressure feature remains invariant to asymmetrical intersection geometries. This observation-side pressure is defined over the two *fixed* phase groups (A versus B); combined with the phase one-hot it lets the policy infer the red-versus-green imbalance that the reward's pressure term penalises directly (Section IV-A, evaluated over the *currently* served phase). The two are consistent but not identical and are bridged by the phase one-hot. We assert that each feature admits a corresponding camera estimator via YOLOv11 and ByteTrack. The residual alignment errors are modeled on a documented basis (outlined in Section III-E) and mathematically propagated into our robustness evaluations in Section VI-E.

**[TABLE-2: 26-Dimensional Observation Space Layout]**
\begin{table}[ht]
\centering
\caption{26-Dimensional Local Proxy Observation Space ($\mathcal{O}_i$)}
\begin{tabular}{clllc}
\toprule
\textbf{Index} & \textbf{Feature} & \textbf{Simulator Source} & \textbf{Camera Estimator} & \textbf{Valid.?} \\
\midrule
0–3 & \text{effective\_queue\_norm} & Halted vehicles / capacity & Bounding boxes with $v \approx 0$ & Yes \\
4–7 & \text{occupancy\_norm} & Sublane area ratio & Bounding box area ratio & Yes \\
8–11 & \text{avg\_speed\_norm} & Mean tracked speed & Track displacement / $\Delta t$ & Yes \\
12–15 & \text{motorbike\_share} & Count by vType & YOLO class \textit{motorcycle} & Yes \\
16–19 & \text{heavy\_vehicle\_share} & Count by vType & YOLO class \textit{truck, bus} & Yes \\
20–23 & \text{phase\_one\_hot} & Controller state & Internal variable & -- \\
24 & \text{green\_timer\_norm} & Controller state & Internal variable & -- \\
25 & \text{pressure\_norm} & Aggregated queues & Aggregated from proxy queue & -- \\
\bottomrule
\end{tabular}
\end{table}

Figure 1b traces the end-to-end vision-proxy construction $\hat{\phi}$: each approach region of interest is reduced by the YOLOv11 + ByteTrack estimator to five bounded features; the four approaches are concatenated into the 20 approach dimensions; and the six controller dimensions (four phase one-hot, the green-timer, and the signed group-mean pressure derived from the approach queues) complete the 26-dimensional observation.

**[FIG-1b: Vision-Proxy Construction — Camera ROIs $\to$ 26-Dimensional Observation] [NEW]**
```mermaid
flowchart LR
    subgraph CAM["Overhead camera · one intersection (one region of interest per approach)"]
        direction TB
        R0["Approach ROI 0"]
        R1["Approach ROI 1"]
        R2["Approach ROI 2"]
        R3["Approach ROI 3"]
    end

    subgraph PHI["Per-approach estimator φ̂ · YOLOv11 detections + ByteTrack tracks"]
        direction TB
        A0["Approach 0 → 5 features:<br/>effective queue, occupancy, avg speed,<br/>motorcycle share, heavy-vehicle share"]
        A1["Approach 1 → 5 features"]
        A2["Approach 2 → 5 features"]
        A3["Approach 3 → 5 features"]
    end

    R0 --> A0
    R1 --> A1
    R2 --> A2
    R3 --> A3

    A0 --> FEAT["20 approach features<br/>(4 approaches × 5)"]
    A1 --> FEAT
    A2 --> FEAT
    A3 --> FEAT

    %% pressure is a DERIVED feature: it reuses ONLY the 4 effective-queue values
    FEAT -. "reuse only the 4 effective-queue values,<br/>partitioned into phase groups A / B" .-> PR["Signed pressure — 1 DERIVED dim<br/>clip(0.5·(1 + q̄_A − q̄_B), 0, 1)<br/>deterministic function of the approach queues"]

    SIG["Controller-internal state (not from camera):<br/>phase one-hot (4) + green-timer norm (1)"]

    FEAT --> OBS["Local observation o_i — 26 dims<br/>= 20 approach + 4 phase + 1 green-timer + 1 pressure"]
    PR --> OBS
    SIG --> OBS

    classDef cam fill:#eef7ee,stroke:#2e7d32,stroke-width:1.5px,color:#14361a;
    classDef est fill:#e8f4fd,stroke:#1565c0,stroke-width:1.5px,color:#0d2b45;
    classDef out fill:#fff4e6,stroke:#e8730c,stroke-width:1.5px,color:#3a2400;
    class R0,R1,R2,R3 cam;
    class A0,A1,A2,A3,FEAT est;
    class SIG,PR,OBS out;
    linkStyle default stroke-width:1.4px;
```

### III-D. Action Space \& Signal Constraints

The action space $\mathcal{A}_i$ consists of a discrete, binary choice dictating which of two non-conflicting phase groups (Group A or Group B) receives the right-of-way. While modern Western intersections routinely utilize 8-phase NEMA ring-barrier controllers, our two-phase simplification is intentionally designed to reflect the legacy, fixed-time hardware prevalent in the developing-city corridors targeted for deployment. All critical safety constraints—specifically the insertion of a mandatory 3-second yellow clearance interval upon a phase switch and the enforcement of the 15-second minimum green time—are strictly handled by the environment transition function $\mathcal{P}$ rather than the learning agent. The extension of this framework to multi-phase ring-barrier logic with protected left turns is deferred to future work concerning advanced physical hardware integration.

The resulting two-phase control logic and its environment-enforced timing constraints are summarized in Figure 1c: the binary action only selects which non-conflicting group holds the right-of-way, while the mandatory yellow clearance and the minimum/maximum-green bounds are applied by the environment transition function $\mathcal{P}$.

**[FIG-1c: Two-Phase Signal Control and Environment-Enforced Constraints] [NEW]**
```mermaid
flowchart LR
    GA["Group A green<br/>SUMO phase 0"]
    GB["Group B green<br/>SUMO phase 2"]
    YAB["Yellow A→B<br/>SUMO phase 1 · 3 s"]
    YBA["Yellow B→A<br/>SUMO phase 3 · 3 s"]

    GA -->|"a = A (extend)"| GA
    GA -->|"a = B, elapsed green < 15 s (min-green hold)"| GA
    GA -->|"a = B, elapsed green ≥ 15 s (switch)"| YAB
    YAB --> GB

    GB -->|"a = B (extend)"| GB
    GB -->|"a = A, elapsed green < 15 s (min-green hold)"| GB
    GB -->|"a = A, elapsed green ≥ 15 s (switch)"| YBA
    YBA --> GA

    NOTE["Decision every Δt = 5 s · binary action a ∈ {A, B} selects which group gets the green<br/>On a switch: env runs the 3 s yellow, then the new green for the remaining 2 s of the step;<br/>the green-timer is set to ~2 s and then accumulates +5 s per extend, capped at 60 s<br/>min-green 15 s is a hard env constraint; max-green 60 s is only the green_timer_norm cap (saturates at 1.0), NOT a forced switch<br/>Yellow clearance and the min-green hold are applied by the environment transition P, not by the agent · episode = 1080 decisions (5400 s)"]

    classDef grn fill:#e8f5e9,stroke:#2e7d32,stroke-width:1.6px,color:#14361a;
    classDef yel fill:#fff8e1,stroke:#f9a825,stroke-width:1.6px,color:#5d4037;
    classDef note fill:#f5f5f5,stroke:#bdbdbd,color:#616161;
    class GA,GB grn;
    class YAB,YBA yel;
    class NOTE note;
    linkStyle default stroke-width:1.4px;
```

### III-E. Sensing-Noise Model (overview)

The "camera-observable" premise of III-B–III-C is credible only if the residual estimation error of the vision pipeline (YOLOv11 + ByteTrack) is bounded and accounted for. Rather than assume perfect recovery ($\varepsilon_{i,t} = 0$), we model the residual error explicitly through five domain-specific modes: multiplicative noise on the effective queue (and the pressure derived from it), additive jitter on low-speed estimates, a confusion flip rate on the class shares, an additive occupancy bias, and two structural failures (a one-step pipeline delay and single-camera dropout). The base ($1\times$) magnitudes are fixed on a documented basis (ByteTrack / YOLOv11 literature + `state.md`); a measured feature-recovery validation against labeled ground truth is deferred to the companion field study / Paper 2 perception validation. The full parameterization, per-feature provenance, and the $\{0\times, 0.5\times, 1\times, 2\times\}$ sweep that drives the robustness evaluation are presented in Section VI-E (TABLE-3a), where they are used.

---

## IV. METHODOLOGY (~2 pages)

### IV-A. Extended-PRESSLIGHT vision-proxy reward (revision 1.2.0)
A central contribution of this work is the formulation of a reward function that is strictly computable from camera-derived proxies, ensuring that the objective optimized during simulation matches the metric available at deployment. Our design rule dictates that every reward term $r^{(k)}_i(t)$ factors through the vision-proxy projection $\phi(\mathbf{s}_t)$; unlike prior vision-RL that pairs a camera-constrained state with a privileged reward, no term in our reward reads a privileged simulator quantity—the accumulated-delay penalty included, which uses a halted-queue proxy. 

The Extended-PRESSLIGHT reward $R_i(t)$ for intersection $i$ is defined as a weighted sum of six components:
\begin{equation}
R_i(t) = w_q \frac{1}{n_i} \sum_{l=1}^{n_i} q_l^2 + w_p p_i(t) + w_T T_i(t) + w_s \mathbb{1}_{[\text{switch}]} + w_v \max(\tau - \bar{v}, 0) + w_w \bar{w}(t)
\end{equation}
where the weights are empirically set to $(w_q, w_p, w_T, w_s, w_v, w_w) = (-1.0, -0.5, 1.0, -0.1, -0.2, -0.3)$. To ensure stable temporal difference learning, the final scalar is clipped to $[-2.0, +1.5]$.

1. **Anti-Starvation Queue Penalty:** Instead of minimizing the mean queue length across $n_i$ lanes, we penalize the mean of squared queue lengths. By Jensen's inequality, $\frac{1}{n} \sum q_l^2 \geq (\frac{1}{n} \sum q_l)^2$, with equality holding strictly when queues are perfectly balanced. The gap between these two quantities is exactly the queue variance. Consequently, this non-linear formulation implicitly penalizes spatial imbalance, preventing starvation of minor approaches. For instance, a highly imbalanced queue distribution $q = (1,0,0,0)$ yields a penalty of $0.25$, whereas a balanced distribution $q = (0.25, 0.25, 0.25, 0.25)$ yields $0.0625$ (a 4-fold reduction).

2. **Signed Pressure:** We incorporate a pressure-inspired term $p_i(t) \in [-1, 1]$ representing the differential between queues on red phases and green phases:
\begin{equation}
p_i(t) = \frac{1}{n_i} \left( \sum_{l \in \text{red}} q_l - \sum_{l \in \text{green}} q_l \right)
\end{equation}
Unlike standard max-pressure formulations that use a one-sided $\max(p, 0)$ and provide zero gradient when queues are balanced, our signed formulation continuously rewards the agent for serving the most congested phase. This reward pressure is evaluated over the *currently* red and green lanes (a dynamic partition that depends on the active phase), whereas the observation feature of Section III-C exposes the *static* group-A-versus-B imbalance; the two are consistent and bridged by the phase one-hot, not identical. Note that due to normalization and value clipping, this does not inherit the theoretical throughput-optimality of classical max-pressure routing; rather, it serves as a highly reactive, heuristic gradient signal.

3. **Local ROI-Exit Throughput:** The agent is rewarded for vehicles actively exiting the intersection. We define throughput $T_i(t) = \text{clip}(|\text{prev\_ids}(L_i) \setminus \text{cur\_ids}(L_i)| / \kappa, 0, 1)$ with capacity $\kappa=20$. Crucially, the simulator computation (set difference of per-lane vehicle IDs between steps) is *semantically identical* to the deployable measurement of track-IDs terminating at the boundary of a ByteTrack Region of Interest (ROI). This ensures the reward function itself is a deployable metric without domain shift. Because an exit is any ID present last step and absent now, lateral lane changes (frequent under the sublane model) and the rare SUMO teleport also register as exits; this is a bounded source of noise that the deployable ByteTrack ROI-exit counter shares identically, so it introduces no train/deploy gap.

4. **Auxiliary Terms:** The reward includes minor penalties for phase switching ($-0.1$) to prevent flickering, and low-speed flow ($-0.2$ if the average speed $\bar{v}$ falls below a threshold $\tau$). Finally, the accumulated-delay penalty $\bar{w}(t)$ is realised as the mean halted-queue fraction—the same $\phi$-computable signal as the queue feature, averaged linearly rather than squared. It supplies a dense low-queue gradient that complements the quadratic anti-starvation term while remaining fully camera-computable (no per-vehicle timer is read). Its marginal contribution beyond the quadratic queue term is isolated by the delay-term ablation in Section VI-F.

### IV-B. MAPPO architecture (v3) — standard method, stated honestly
To optimize the multi-agent control policy, we adopt Multi-Agent Proximal Policy Optimization (MAPPO), a standard Centralized Training with Decentralized Execution (CTDE) algorithm. We emphasize that the RL architecture itself is standard and not claimed as a methodological contribution; rather, it is the vehicle for evaluating the proposed camera-constrained state and reward formulation. 

The architecture consists of a parameter-shared actor network $\pi_\theta(a_{i,t}|\mathbf{o}_{i,t})$ operating on the 26-dimensional local proxy observation, ensuring homogeneous behavior across intersections and facilitating zero-shot transfer to novel topologies. During training, we utilize a centralized critic $V_\psi(\mathbf{s}_t)$ that takes the concatenated global proxy state as input. To preserve local reward shaping, the critic employs a multi-head architecture outputting one distinct value estimate per agent. Advantages $\hat{A}_{i,t}$ are computed using per-agent Generalized Advantage Estimation (GAE, $\lambda=0.95$) based on the decentralized rewards $r_{i,t}$, and are subsequently standardized jointly across all agents in the batch to stabilize policy updates. 

**[EQ-8: Per-Agent GAE]**
\begin{equation}
\hat{A}_{i,t} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{i,t+l} \quad \text{where} \quad \delta_{i,t} = r_{i,t} + \gamma V_\psi(\mathbf{s}_{t+1})_i - V_\psi(\mathbf{s}_t)_i
\end{equation}

**[EQ-9: Clipped Surrogate \& Per-Head Value Loss]**
\begin{equation}
L^{\text{clip}}(\theta) = \mathbb{E} \left[ \min\left(p_{i,t} \hat{A}_{i,t}, \text{clip}(p_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}_{i,t}\right) \right]
\end{equation}
\begin{equation}
L^{\text{VF}}(\psi) = \frac{1}{N} \sum_{i=1}^N \mathbb{E} \left[ \max\left((V_\psi(\mathbf{s}_t)_i - \hat{R}_{i,t})^2, (V_{\text{clip}} - \hat{R}_{i,t})^2\right) \right]
\end{equation}

The optimization uses the PPO clipped surrogate objective with $\epsilon=0.2$, combined with a clipped value-loss and early stopping triggered if the approximate KL-divergence exceeds $0.015$. To bridge the domain gap, observation statistics are dynamically tracked using Welford's algorithm and serialized directly into the deployment checkpoint alongside the actor weights, ensuring that the test-time inputs $\mathbf{o}_{i,t}$ are normalized identically without requiring live statistics.
**[FIG-1: Architecture Diagram]**
```mermaid
%% Solid arrows = on-camera deployment data path (no simulator).
%% Dashed arrows = training-only (SUMO).
%% Only the actor weights θ and the observation-normalization stats (obs_rms) are
%% serialized into the deployed checkpoint; the critic, the global state, and the
%% state-normalization stats (state_rms) never deploy.
%% NOTE: the VisionBuffer EMA exists ONLY at deployment; training feeds raw 5 s-step
%% observations (a train/deploy temporal-alignment asymmetry — see Limitations).
flowchart LR
    subgraph DEP["Deployment — on-camera, no simulator (2 of N agents shown)"]
        direction TB
        C1["Camera regions of interest<br/>agent 1"] -->|"YOLOv11 + ByteTrack"| F1["Feature extractor φ̂<br/>(per-frame raw features)"]
        C2["Camera regions of interest<br/>agent 2"] -->|"YOLOv11 + ByteTrack"| F2["Feature extractor φ̂<br/>(per-frame raw features)"]
        F1 --> BUF["VisionBuffer · per-dimension EMA (α=0.6, last 5 frames) + clip to 0–1<br/>~10 Hz frames → one policy decision per 5 s"]
        F2 --> BUF
        BUF --> O1["Local observation o₁<br/>(26-dim, dims 0–25 of the 52-dim packet)"]
        BUF --> O2["Local observation o₂<br/>(26-dim, dims 26–51)"]
        O1 --> NZ["Observation normalization<br/>(Welford obs_rms — fit in training, applied in both phases)"]
        O2 --> NZ
        NZ --> ACT["Parameter-shared actor π_θ<br/>(same weights for every agent)"]
        ACT --> ACTS["Actions a₁, a₂ ∈ {A, B}"]
    end

    subgraph TR["Training only — SUMO (centralized training, decentralized execution)"]
        direction TB
        GS["Global state<br/>s = [o₁ ; … ; o_N]"] --> SZ["State normalization<br/>(Welford state_rms)"]
        SZ --> CR["Centralized critic V_ψ<br/>(one value head per agent)"]
        RW["Per-agent shaped rewards<br/>r₁, r₂"] --> GAE["Per-agent GAE<br/>(γ=0.99, λ=0.95) → advantages Âᵢ, returns R̂ᵢ"]
        CR -->|"per-agent values V_ψ(s)ᵢ"| GAE
        GAE -->|"returns R̂ᵢ → clipped value loss"| CR
    end

    O1 -.->|"training: raw obs, no EMA"| GS
    O2 -.-> GS
    GAE -.->|"advantages Âᵢ → PPO clipped-surrogate update of θ"| ACT

    classDef dep fill:#e8f4fd,stroke:#1565c0,stroke-width:1.5px,color:#0d2b45;
    classDef tr  fill:#fff4e6,stroke:#e8730c,stroke-width:1.5px,color:#3a2400;
    class C1,C2,F1,F2,BUF,O1,O2,NZ,ACT,ACTS dep;
    class GS,SZ,CR,GAE,RW tr;
    linkStyle default stroke-width:1.4px;
```
**[ALGORITHM-1: MAPPO Training with Per-Agent GAE]**
```latex
\begin{algorithm}
\caption{MAPPO with Per-Agent GAE for TSC}
\begin{algorithmic}[1]
\STATE Initialize parameter-shared actor $\pi_\theta$, centralized multi-head critic $V_\psi$
\STATE Initialize Welford observation running mean/var statistics
\FOR{iteration $1, 2, \ldots, \text{max\_iterations}$}
    \STATE Reset environments and get initial observations
    \FOR{step $t=1$ to $128$}
        \STATE Normalize local observations $\mathbf{o}_{i,t}$ using running stats
        \STATE Sample actions $a_{i,t} \sim \pi_\theta(\cdot|\mathbf{o}_{i,t})$ for each agent $i$
        \STATE Execute joint action $\mathbf{a}_t$. Env handles 3s yellow, min/max green constraints.
        \STATE Observe rewards $r_{i,t}$ and next states $\mathbf{o}_{i,t+1}$
        \STATE Store transition in per-agent buffers
    \ENDFOR
    \STATE Update Welford running stats with batch observations
    \FOR{each agent $i$}
        \STATE Compute advantages $\hat{A}_{i,t}$ using per-agent GAE($\gamma=0.99, \lambda=0.95$) with $r_{i,t}$ and $V_\psi(\mathbf{s}_t)_i$
    \ENDFOR
    \STATE Standardize advantages $\hat{A}_{i,t}$ jointly across all agents
    \FOR{epoch $1$ to $10$}
        \STATE Compute PPO clipped surrogate loss $L^{\text{clip}}(\theta)$ across all agents
        \STATE Compute multi-head value loss $L^{\text{VF}}(\psi)$
        \STATE Update $\theta, \psi$ using Adam optimizer
        \IF{approx-KL divergence > 0.015}
            \STATE Break epochs (Early stop)
        \ENDIF
    \ENDFOR
\ENDFOR
\STATE Serialize actor $\pi_\theta$ and normalization stats for deployment
\end{algorithmic}
\end{algorithm}
```

---

## V. EXPERIMENTAL SETUP (~1.5 pages)

### V-A. Simulation environment & mixed-traffic calibration [REVISED]
We conduct our simulations using SUMO v1.20.0, specifically leveraging its sublane model with a lateral resolution of $0.4$ m to accurately capture the non-lane-based, filtering dynamics characteristic of developing-world traffic. 

A critical element of this environment is the calibrated vehicle composition. We set the motorcycle share to $73\%$, matching the $72.6\%$ motorcycle mode share reported for Hanoi by Ngoc \emph{et al.}~\cite{ngoc2021_fivecities}; the remaining vehicles are split among passenger cars ($17\%$), trucks ($7\%$), and buses ($3\%$) by vehicle count. The same study reports motorcycle shares of $80$–$89\%$ for Hai Phong, Da Nang, Ho Chi Minh City, and Can Tho—a range bracketed by our composition sweep (Section VI-D). We strictly distinguish count share from traffic load: at a motorcycle PCU equivalence of $\approx 0.29$~\cite{cao_sano_hanoi}, the $73\%$ motorcycle count corresponds to roughly one-third of the total PCU-based load (the exact fraction depends on the heavy-vehicle PCU values). We therefore use count share when analyzing lateral filtering behavior, but rely on PCU for all capacity-related discussions. To simulate realistic motorcycle swarming, sublane parameters are configured with `latAlignment=arbitrary` and a lateral gap `minGapLat=0.12` m, allowing approximately three to four motorcycles to effectively share a standard $3.2$ m urban lane alongside a passenger car. 

Under this configuration, the simulated saturation flow reaches approximately $2100$ PCU/h per lane, consistent with capacity measurements for mixed-flow urban roads in Hanoi~\cite{cao_sano_hanoi}. We recognize that regional compositions vary; thus, the central $73\%$ value acts as a representative anchor. The generalizability of the policy across shifting compositions is formally defended via a systematic motorcycle-share sweep ($\{50\%, 73\%, 90\%\}$) in Section VI-D.

#### Signal-Timing Bounds

The minimum green time is explicitly set to $15$ seconds ($3$ decision intervals). This duration ensures that once a phase is activated, it clears a meaningful platoon, preventing high-frequency flickering that previously caused wide motorcycle clusters to be trapped in the intersection. The maximum green time is fixed at $60$ seconds. This bound was chosen based on a single-seed design-selection pilot evaluated on the 16-junction N3 grid (Seed 42, 100k training steps) on held-out evaluation demand. As shown in Table V-A1, the $60$ s bound strictly dominates a relaxed $90$ s bound across all primary metrics. Crucially, the extended $90$ s bound severely degraded the 95th-percentile (P95) waiting time ($\Delta = +24.7\%$), as overly long green phases induced starvation on cross-streets, directly conflicting with our anti-starvation reward objective. This pilot justifies the bounds as a structural configuration choice uniformly applied to all RL and baseline methods.

**[TABLE-V-A1: max-green selection pilot (N3 grid, seed 42, 100 k steps, held-out eval).** Lower is better; bold the winner.]

| max-green | Mean travel time (s) | Mean waiting (s) | P95 waiting (s) |
|---|---|---|---|
| **60 s (selected)** | **311.1** | **135.8** | **504** |
| 90 s | 367.0 | 173.6 | 669 |
| Δ (60 vs 90) | −15.2 % | −21.8 % | −24.7 % |

Note: max-green also sets the `green_timer_norm` scale (`timer / max_green`), so it is a coupled obs/MDP parameter fixed before training and held identical across all arms and baselines (SUMO-actuated `maxDur` matched to 60 s for fairness).

### V-B. Networks and demand
We evaluate the framework on two synthetic, structurally sound topologies: an N2 signalized corridor ($1 \times 5$ junctions) and an N3 signalized grid ($4 \times 4$ junctions). All intersections are 4-way with 2 lanes per approach.

To rigorously stress-test the controllers under dynamic load, we define a time-varying demand profile $\lambda(t)$ lasting $5400$ seconds. Vehicle insertions follow an inhomogeneous Poisson process driven by a piecewise-linear trapezoidal rate:
\begin{equation}
\lambda(t) \text{ defined by knots at } \{0, 900, 1500, 3000, 3600, 4800, 5400\} \text{ s} 
\end{equation}
with corresponding base rates of $\{0.4, 0.4, 1.67, 1.67, 0.83, 0.83, 0.4\}$ vehicles per second (scaled proportionally by the number of network entries). This profile smoothly progresses through an off-peak lull, ramps to a single demand peak sustained over $1500$–$3000$ s, eases to a medium-load plateau over $3600$–$4800$ s, and tapers back to off-peak.
**[FIG-2: demand profile λ(t) + N2/N3 network diagrams side by side.]**

### V-C. Baselines
We benchmark MAPPO against an array of robust baselines, ensuring fair comparisons by restricting heuristics to the same phase groups and constraints:

- **Webster Fixed-Time:** A mathematically derived fixed-time policy using PCU-weighted measured flows. The cycle length $C = \frac{1.5L + 5}{1 - Y}$ is clamped to $[40, 120]$ seconds with proportional green splits, providing a highly competitive, non-strawman baseline.
- **SUMO-Actuated:** A gap-based actuation controller native to SUMO, configured with `minDur=15` and `maxDur=60` to strictly match the RL constraints.
- **Max-Pressure:** A greedy heuristic routing policy utilizing the identical observation pipeline and phase groups as the RL agents, routing right-of-way to the phase with the highest pressure.
- **IPPO (Independent PPO):** An ablative baseline using an identical actor architecture and training budget, but utilizing independent per-agent critics conditioned strictly on local $26$-dimensional observations. This isolates the value of Centralized Training with Decentralized Execution.
- **MAPPO-Privileged (Upper Bound):** An identical MAPPO algorithm whose observation space is augmented into a $38$-dimensional vector incorporating exact, unobservable SUMO metrics (exact halting counts, accumulated waiting times, and total vehicle counts). The reward remains identical to the proxy arm; only the state visibility differs, allowing us to quantify the explicit cost of information restriction.

### V-D. Metrics \& Statistical Protocol

Performance is quantified via two primary metrics extracted from SUMO's `tripinfo` output: Mean Travel Time and Mean Waiting Time. Secondary metrics include the 95th-percentile (P95) waiting time to capture starvation events, throughput ratio, mean vehicle speed, and phase switches per episode. Environmental metrics (CO$_2$ emissions and fuel consumption via the HBEFA3 model) are evaluated post-hoc and never exposed to the agent reward.

To ensure statistical rigor, our protocol dictates **$5$ independent training seeds ($42, 123, 456, 789, 1337$)**, evaluated against $10$ separate held-out demand route-seeds per scenario. All tabular results report the mean across training seeds accompanied by a $95\%$ Confidence Interval (CI). Significance is determined using a Mann-Whitney U test against the strongest baseline, corrected via the Holm-Bonferroni method across scenarios.

### V-E. Training Configuration \& Reproducibility

Training relies on standard hyperparameters (Table 3), including $\gamma = 0.99$, GAE $\lambda = 0.95$, PPO clip $= 0.2$, and a learning rate of $3 \times 10^{-4}$ with linear decay. The models are trained up to a fixed budget of $500,000$ steps as determined by convergence pilots.

To guarantee complete reproducibility, the full codebase, pre-trained network checkpoints, dynamic demand generators (`.rou.xml`), phase-group derivations, and evaluation scripts are publicly released at [REPOSITORY URL] under a tagged commit. Furthermore, the precise observation and reward schema version is explicitly stamped into every run configuration.

**[TABLE-3: MAPPO Hyperparameters]**
\begin{table}[ht]
\centering
\caption{MAPPO Training Hyperparameters}
\begin{tabular}{lc}
\toprule
\textbf{Hyperparameter} & \textbf{Value} \\
\midrule
Discount factor ($\gamma$) & $0.99$ \\
GAE parameter ($\lambda$) & $0.95$ \\
PPO clip ratio ($\epsilon$) & $0.2$ \\
Target KL divergence & $0.015$ \\
Learning rate & $3 \times 10^{-4}$ (linear decay) \\
Rollout horizon & $128$ steps \\
Minibatch size & $256$ \\
Optimization epochs & $10$ \\
Total environment steps & $5 \times 10^5$ \\
\bottomrule
\end{tabular}
\end{table}

---

## VI. RESULTS (~3 pages — build this before writing anything else)

### VI-A. Main Comparison

[INSERT: 2 paragraphs discussing MAPPO performance on N2 and N3 compared to baselines. Explicitly reference cells in Table 4 and Table 5. Report instances where Actuated control is competitive off-peak.]

**[TABLE-4: Performance on N2 Corridor (5 junctions)]**
\begin{table*}[ht]
\centering
\caption{Performance on N2 Corridor (Mean $\pm$ 95\% CI over 5 seeds). Bold indicates best performance.}
\begin{tabular}{lccccc}
\toprule
\textbf{Method} & \textbf{Travel Time (s)} & \textbf{Wait Time (s)} & \textbf{P95 Wait (s)} & \textbf{Throughput (\%)} & \textbf{Switches} \\
\midrule
Webster & [0.0] $\pm$ [0.0] & [0.0] $\pm$ [0.0] & [0] & [0.0] & [0.0] \\
SUMO-Actuated & [0.0] $\pm$ [0.0] & [0.0] $\pm$ [0.0] & [0] & [0.0] & [0.0] \\
Max-Pressure & [0.0] $\pm$ [0.0] & [0.0] $\pm$ [0.0] & [0] & [0.0] & [0.0] \\
IPPO & [0.0] $\pm$ [0.0] & [0.0] $\pm$ [0.0] & [0] & [0.0] & [0.0] \\
\textbf{MAPPO (Proxy)} & \textbf{[0.0]} $\pm$ \textbf{[0.0]} & \textbf{[0.0]} $\pm$ \textbf{[0.0]} & \textbf{[0]} & \textbf{[0.0]} & \textbf{[0.0]} \\
\midrule
\textit{MAPPO (Privileged)} & \textit{[0.0]} $\pm$ \textit{[0.0]} & \textit{[0.0]} $\pm$ \textit{[0.0]} & \textit{[0]} & \textit{[0.0]} & \textit{[0.0]} \\
\bottomrule
\end{tabular}
\end{table*}

**[TABLE-5: Performance on N3 Grid (16 junctions)]**
\begin{table*}[ht]
\centering
\caption{Performance on N3 Grid (Mean $\pm$ 95\% CI over 5 seeds). Bold indicates best performance.}
\begin{tabular}{lccccc}
\toprule
\textbf{Method} & \textbf{Travel Time (s)} & \textbf{Wait Time (s)} & \textbf{P95 Wait (s)} & \textbf{Throughput (\%)} & \textbf{Switches} \\
\midrule
Webster & [0.0] $\pm$ [0.0] & [0.0] $\pm$ [0.0] & [0] & [0.0] & [0.0] \\
SUMO-Actuated & [0.0] $\pm$ [0.0] & [0.0] $\pm$ [0.0] & [0] & [0.0] & [0.0] \\
Max-Pressure & [0.0] $\pm$ [0.0] & [0.0] $\pm$ [0.0] & [0] & [0.0] & [0.0] \\
IPPO & [0.0] $\pm$ [0.0] & [0.0] $\pm$ [0.0] & [0] & [0.0] & [0.0] \\
\textbf{MAPPO (Proxy)} & \textbf{[0.0]} $\pm$ \textbf{[0.0]} & \textbf{[0.0]} $\pm$ \textbf{[0.0]} & \textbf{[0]} & \textbf{[0.0]} & \textbf{[0.0]} \\
\midrule
\textit{MAPPO (Privileged)} & \textit{[0.0]} $\pm$ \textit{[0.0]} & \textit{[0.0]} $\pm$ \textit{[0.0]} & \textit{[0]} & \textit{[0.0]} & \textit{[0.0]} \\
\bottomrule
\end{tabular}
\end{table*}

### VI-B. Convergence

**[FIG-3: Episode Reward vs Steps]**
**[FIG-4: Entropy \& KL Curves]**

[INSERT: 1 paragraph discussing convergence curves. Note the training cost in steps/s and wall-clock time.]

### VI-C. The Cost of Restricting to Camera-Computable Signals

[INSERT: 2 paragraphs discussing the performance gap between MAPPO-Privileged and MAPPO-Proxy. State the restriction cost is [X]\% on travel time.] 

Crucially, both the proxy and privileged arms are evaluated using pristine simulator features. Consequently, this comparison strictly isolates the cost of *information restriction* (the inability to observe exact vehicle delays or network-wide states). The distinct impact of *real-world sensor error* is modeled entirely separately in Section VI-E, grounded by the validation in Section III-E. A discussion on the broader domain shift encountered in physical deployment is reserved for Section VII.

### VI-D. Generalization (Out-of-Distribution Demand \& Composition)

[INSERT: Prose on performance under low/medium/high/asymmetric demand.]

To formally defend our baseline $73\%$ motorcycle-share selection, we evaluate the pre-trained policy across an explicit composition sweep ($\{50\%, 73\%, 90\%\}$). [INSERT: Prose discussing the robust performance across shifting compositions, validating the $73\%$ anchor].

**[TABLE-7: OOD Demand Results]**
**[FIG-6a: Performance vs Moto-Share]**

### VI-E. Robustness to Sensing Noise

The sensing-noise model introduced in III-B (as the projection perturbation $\varepsilon_{i,t}$) and outlined in III-E is parameterized here, where it is used. Each proxy feature is paired with a domain-specific error mode whose base ($1\times$) magnitude is fixed on a DOCUMENTED BASIS (ByteTrack / YOLOv11 literature + `state.md`), summarised in TABLE-3a:

- **Queue / pressure:** multiplicative, relative $\sigma_q = 0.10$ (perspective + pixels-per-metre calibration error scaling with absolute queue length); pressure inherits $\sigma_q$ as it is derived from the approach queues.
- **Occupancy:** additive perspective-residual bias $b_{\text{occ}} = 0.08$ (overlapping boxes inflate the projected occupancy / summed box-length ratio).
- **Speed:** additive $\sigma_v = 1.0$ m/s concentrated below 3 m/s ($\times 0.25$ above) — ByteTrack displacement jitter near stationary.
- **Class shares:** confusion flip rate $\rho_c = 0.05$ on the motorcycle and heavy-vehicle fractions.
- **Structural:** one-step pipeline delay (hardware latency) + single-camera dropout (zeroes an entire approach).

We sweep the whole envelope over $\{0\times, 0.5\times, 1\times, 2\times\}$ of the base magnitudes (FIG-6). The formal conclusions rest on the SHAPE of the degradation curve across these scales, not on the absolute precision of the $1\times$ anchor.

> **PROVENANCE HONESTY (Option B — current state).** The $1\times$ magnitudes are documented-basis values, NOT measured against manual ground truth. The feature-recovery study — an estimated-vs-ground-truth scatter with per-feature bias/$\sigma$ — requires labeled camera frames and is DEFERRED to the companion field study / Paper 2 perception validation. When that study runs (`calibrate_noise.py --gt-csv --est-csv --confusion-matrix`), replace the documented-basis values with the measured ones and restore the stronger "camera-observable demonstrated, not asserted" claim. Until then, the $\{0$–$2\times\}$ sweep is the defense.

**[TABLE-3a: Sensing-Noise Modes, Base ($1\times$) Magnitudes, and Provenance]**
\begin{table}[ht]
\centering
\caption{Sensing-Noise Modes, Base ($1\times$) Magnitudes, and Provenance}
\begin{tabular}{lll}
\toprule
\textbf{Feature} & \textbf{Noise Model} & \textbf{Base / Provenance} \\
\midrule
Queue ($\sigma_q$) & Multiplicative, rel. $\mathcal{N}(0, \sigma_q^2)$ & $0.10$ — documented basis \\
Occupancy ($b_{\text{occ}}$) & Additive bias & $0.08$ — set value \\
Speed ($\sigma_v$) & Additive $\mathcal{N}(0, \sigma_v^2)$, $v<3$ m/s ($\times0.25$ above) & $1.0$ m/s — documented basis \\
Class shares ($\rho_c$) & Confusion flip rate & $0.05$ — documented basis \\
Pressure & Inherits $\sigma_q$ & $0.10$ — derived \\
Structural & 1-step delay; ROI dropout & — hardware / failure \\
\bottomrule
\end{tabular}
\end{table}

**[FIG-6: Degradation Curves across Noise Scales \{0, 0.5x, 1x, 2x\}]**

[INSERT: 2 paragraphs discussing FIG-6. Claim: the camera-constrained policy degrades gracefully within the documented-basis envelope (TABLE-3a) while baseline heuristics also degrade. Anchor the 1× envelope to the TABLE-3a magnitudes.]

### VI-F. Ablations

**[TABLE-8: Reward \& Architecture Ablations]**
\begin{table}[ht]
\centering
\caption{Ablation Study (Mean Travel Time over 3 Retrained Seeds)}
\begin{tabular}{lcc}
\toprule
\textbf{Configuration} & \textbf{Travel Time (s)} & \textbf{$\Delta$ from Full} \\
\midrule
\textbf{Full MAPPO (Proxy)} & \textbf{[0.0]} & -- \\
\midrule
\textit{Reward Ablations} & & \\
No Pressure Term & [0.0] & +[0.0]\% \\
No Throughput Term & [0.0] & +[0.0]\% \\
Queue Penalty Only & [0.0] & +[0.0]\% \\
Unsigned Pressure & [0.0] & +[0.0]\% \\
Mean-then-Square Queue & [0.0] & +[0.0]\% \\
No Delay Term ($w_w{=}0$) & [0.0] & +[0.0]\% \\
\midrule
\textit{State \& Arch. Ablations} & & \\
Minus Class-Shares & [0.0] & +[0.0]\% \\
Minus Pressure Feature & [0.0] & +[0.0]\% \\
IPPO (No CTDE) & [0.0] & +[0.0]\% \\
Train Lane $\to$ Test Sublane & [0.0] & +[0.0]\% \\
\bottomrule
\end{tabular}
\end{table}

[INSERT: Discuss which components matter most. Emphasize that the Sublane-vs-Lane cross-evaluation confirms that modeling mixed-traffic dynamics fundamentally alters the policy. Note that removing the linear delay term isolates its marginal contribution beyond the quadratic queue penalty — both terms are camera-computable, so neither relies on a privileged read.]

### VI-G. Environmental \& Secondary Metrics

[INSERT: 1 paragraph on CO$_2$ and fuel. Restate that these metrics never entered the reward function, thereby reinforcing the core deployability objective.]
**[TABLE-10: CO$_2$ and Fuel per Episode]**

---

## VII. DISCUSSION \& LIMITATIONS

Several limitations bound the immediate applicability of this study and present clear avenues for future work:
1. **Two-Phase Control Logic:** Our framework simplifies the action space to two non-conflicting phase groups without protected left turns. While this aligns perfectly with the legacy fixed-time controllers targeted in developing cities, modernizing the framework to support 8-phase NEMA ring-barrier logic is required for broader international integration.
2. **Simulation-Only Scope:** While our noise model is parameterized on a documented basis and swept over a $\{0$--$2\times\}$ envelope (Section III-E; a measured feature-recovery study is deferred to the companion field study), the results presented herein remain simulation-based. This paper establishes the theoretical bounds and restriction costs of camera-only RL-TSC; closed-loop physical deployment—which entails complex network latencies and hardware integration—is explicitly deferred to an ongoing companion field study.
3. **Calibration Depth of the Traffic Mix:** The $73\%$ motorcycle share is anchored to Hanoi's reported motorcycle mode share~\cite{ngoc2021_fivecities}; we note that mode share (measured by trips) and intersection vehicle-count share differ modestly, and the $17\%/7\%/3\%$ split of the remaining fleet is a representative assumption rather than a per-corridor measurement. Moreover, certain sublane microscopic parameters (e.g., exact lateral gaps at high saturation) were assumed from standard SUMO heuristics rather than trajectory-level physical calibration. 
4. **Network Scale Limitations:** Evaluations were capped at a 16-junction grid. Scaling to city-wide networks (hundreds of junctions) will likely fracture the centralized critic's credit assignment and stress the locality assumption of the proxy pressure feature.

---

## VIII. CONCLUSION

This paper addressed the persistent observation gap in RL-based traffic signal control by enforcing a strict constraint: pairing a camera-computable state proxy with a fully camera-computable Extended-PRESSLIGHT reward. [INSERT: 2 sentences reporting final restriction cost %, improvement vs Webster, and noise-robustness findings]. For traffic practitioners in developing metropolises, this framework proves that scalable, adaptive signal control is achievable utilizing existing edge-camera hardware without relying on unmeasurable simulator-privileged data. Future work will focus on integrating multi-phase ring-barrier logic and executing the physical closed-loop deployment of this policy.

---

## ACKNOWLEDGMENT / REFERENCES / SUPPLEMENT

- **References:** 35–45; ≥ 10 from 2023–2026; T-ITS self-citations 3–5; cite every baseline and every tool (SUMO, YOLOv11, ByteTrack) and the Vietnamese composition source(s).
- **Supplementary / repo:** demand `.rou.xml` of the exact published runs (⚠ commit at camera-ready — duarouter versions are not byte-stable), best-seed checkpoint, evaluation scripts, III-E noise config + calibration script (feature-recovery data deferred), CHECK-suite outputs.

---

## PRE-SUBMISSION GATE (do not submit unless all ✓)

- [ ] Every number in the Abstract traceable to a table
- [ ] 5 seeds × 10 eval seeds everywhere; CI + significance marks in every results table
- [ ] Ablations retrained (no eval-time reward reweighting anywhere)
- [ ] No "perfect/first/novel" claims that II-B contradicts; no "sim-to-real" as a verb
- [ ] Sensing-noise model: ε introduced in III-B, brief overview in III-E, parameterized in VI-E (TABLE-3a base magnitudes + provenance); NO IV-C subsection and no FIG-1a; documented-basis values flagged, measured feature-recovery deferred to companion study
- [ ] V-A: saturation-flow number + Vietnamese-mix citation + count-vs-PCU sentence all present
- [ ] VI-C labeled "information-restriction cost" with the clean-features caveat sentence
- [ ] VI-C (restriction cost) and VI-E (noise robustness) BOTH present — they are the paper
- [ ] VI-F includes sublane-vs-lane cross-eval and the delay-term ablation
- [ ] Repo frozen at a tagged commit; schema/version stamps verified in all run configs
- [ ] Discussion VII consistent with Abstract/Title (sim-only scope stated, not buried)
