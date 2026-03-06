# CardioSim Researcher Outreach - 2026-03-05

> 7 personalized emails for researchers cited in or relevant to the CardioSim paper.
> Review each email, then forward from your inbox if appropriate.

---

## 1. John Jiang - UT Austin (RL Pacemaker Synthesis)

**To:** jjiang@utexas.edu
**Subject:** CardioSim: open-source Gymnasium environments for cardiac RL

Dear John,

I read your 2024 paper on learning reward machines from demonstrations for pacemaker synthesis with great interest. The idea of using reward machines to encode temporal specifications for cardiac pacing is compelling, and it motivated part of our recent work.

We just released CardioSim, an open-source Python package providing three Gymnasium-compatible RL environments for cardiac electrophysiology: pacemaker rate optimization, antiarrhythmic drug dosing, and defibrillation timing. The pacing environment wraps a FitzHugh-Nagumo model with a cardiac conduction system simulator, producing observations (heart rate, HRV, membrane potential) that could serve as a standardized benchmark for the reward machine approach you developed.

The package is on PyPI (`pip install cardiosim`) and the paper and code are at https://github.com/HassDhia/cardiosim. We include PPO baselines but would be curious to see how reward machine policies compare on our environments.

Would be happy to discuss if this is useful for your work.

Best regards,
Hass Dhia
Smart Technology Investments Research Institute

---

## 2. Ufuk Topcu - UT Austin (Formal Methods + RL)

**To:** utopcu@utexas.edu
**Subject:** Gymnasium environments for cardiac RL benchmarking

Dear Professor Topcu,

Your group's work on formal methods for RL in safety-critical systems, including the recent pacemaker synthesis paper with reward machines, addresses an important gap in cardiac device verification. We cited this work in our new paper on CardioSim.

CardioSim provides three Gymnasium-compatible environments for cardiac electrophysiology (pacing control, drug dosing, defibrillation timing) with configurable difficulty tiers. We think these could be useful as standardized benchmarks for testing formally verified RL policies, particularly because the environments include safety-relevant dynamics like therapeutic windows and refractory periods.

The package is available at https://github.com/HassDhia/cardiosim and on PyPI. A preprint describing the environments and baseline results is included in the repository.

Best regards,
Hass Dhia
Smart Technology Investments Research Institute

---

## 3. Elisa M. Tosca - University of Padova (RL Drug Dosing)

**To:** elisa.tosca@unipd.it
**Subject:** CardioSim: RL benchmarks for drug dosing with PK/PD models

Dear Elisa,

Your 2024 paper on model-informed reinforcement learning for precision dosing was a key reference for our antiarrhythmic dosing environment. The integration of PK/PD models with RL that you demonstrated is exactly the approach we wanted to make accessible as a benchmark.

We released CardioSim, which includes an AntiarrhythmicDosing-v0 environment that couples the Aliev-Panfilov cardiac model with a single-compartment PK model and sigmoid Emax pharmacodynamics. PPO agents achieve 5.1x improvement over random baselines on this environment. The narrow therapeutic window and delayed drug effects create a challenging optimization landscape that we think captures some of the core difficulties you identified in your work.

The code is at https://github.com/HassDhia/cardiosim (MIT licensed, `pip install cardiosim`). We would welcome feedback on whether the PK/PD dynamics are representative enough to be useful for benchmarking more sophisticated dosing algorithms.

Best regards,
Hass Dhia
Smart Technology Investments Research Institute

---

## 4. Antonin Raffin - DLR (Stable-Baselines3)

**To:** antonin.raffin@dlr.de
**Subject:** New Gymnasium environments for cardiac RL (built on SB3)

Dear Antonin,

We built CardioSim using Stable-Baselines3 for all our PPO baselines, and wanted to share the project with you. CardioSim provides three Gymnasium-compatible environments for cardiac electrophysiology: pacemaker rate control, antiarrhythmic drug dosing, and defibrillation timing.

One finding that might interest you from an RL perspective: the defibrillation timing environment has stochastic success dynamics that make random policies surprisingly competitive with PPO (ratio 0.95). The environment rewards successful defibrillation but penalizes unnecessary shocks, creating a setting where PPO's learned caution about shocking is sometimes counterproductive compared to the random agent's high-frequency shocking. We found that high entropy coefficients (0.1) and stochastic evaluation were necessary to prevent policy collapse.

The package is at https://github.com/HassDhia/cardiosim and on PyPI. All three environments follow the Gymnasium API and work directly with SB3.

Best regards,
Hass Dhia
Smart Technology Investments Research Institute

---

## 5. Sara Dutta - FDA/CDER (Antiarrhythmic Simulation)

**To:** sara.dutta@fda.hhs.gov
**Subject:** Open-source RL environments for antiarrhythmic drug simulation

Dear Sara,

Your 2024 simulation study on antiarrhythmic mechanisms in myocardial ischemia provided valuable context for our work on computational approaches to cardiac pharmacology. We cited your paper in CardioSim, an open-source package that provides Gymnasium-compatible RL environments for cardiac electrophysiology.

Our AntiarrhythmicDosing-v0 environment models drug concentration dynamics with a single-compartment PK model coupled to the Aliev-Panfilov cardiac action potential model. While simplified compared to the ionic models used in your work, it captures the core challenge of maintaining therapeutic drug concentration while suppressing arrhythmia and avoiding toxicity.

We think standardized simulation benchmarks could complement the high-fidelity modeling work at FDA/CDER. The package is at https://github.com/HassDhia/cardiosim (MIT license, `pip install cardiosim`).

Best regards,
Hass Dhia
Smart Technology Investments Research Institute

---

## 6. Yawei Liu - Zhejiang University (Deep RL Drug Dosing)

**To:** liuyawei@zju.edu.cn
**Subject:** CardioSim: Gymnasium environments for cardiac drug dosing RL

Dear Yawei,

Your work on optimizing warfarin dosing using deep reinforcement learning demonstrated the potential of RL for drug dosing optimization. We cited your paper in CardioSim, where we extended the idea to antiarrhythmic drug dosing with explicit PK/PD dynamics.

CardioSim provides three Gymnasium-compatible environments including an antiarrhythmic dosing environment where PPO agents learn to maintain drug concentrations within a therapeutic window while responding to arrhythmia. The delayed drug effects and narrow therapeutic range create a challenging optimization problem similar to what you encountered with warfarin.

The full package (code, paper, baselines) is at https://github.com/HassDhia/cardiosim. Would be interested to know if the environment dynamics are relevant to your dosing optimization research.

Best regards,
Hass Dhia
Smart Technology Investments Research Institute

---

## 7. Iraia Isasi - University of the Basque Country (ML Defibrillation)

**To:** iraia.isasi@ehu.eus
**Subject:** RL environments for defibrillation timing optimization

Dear Iraia,

Your systematic review on machine learning algorithms for predicting defibrillation success was an important reference for our work. We released CardioSim, which includes a DefibrillationTiming-v0 environment that frames the shock timing decision as an RL problem.

The environment simulates ventricular fibrillation using a chaotic FitzHugh-Nagumo regime where shock success probability depends on energy delivered, timing, and tissue fatigue. One interesting result: the stochastic success dynamics make random policies competitive with PPO, suggesting that the optimal policy in this domain may need to balance exploration (frequent shocking) against efficiency (minimizing energy). This connects to your review's discussion of predicting optimal shock windows.

The package is at https://github.com/HassDhia/cardiosim (MIT licensed, `pip install cardiosim`). We think RL-based defibrillation timing could complement the prediction approaches surveyed in your review.

Best regards,
Hass Dhia
Smart Technology Investments Research Institute
