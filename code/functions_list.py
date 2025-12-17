import numpy as np
from typing import Any, Tuple
import matplotlib.pyplot as plt
WEEKS_PER_YEAR = 52.14

def _get(p: Any, key_or_idx: Any):
    """
    Helper to read from either:
      - numpy array/list (0-based index), or
      - dict-like (by key), or
      - object with attributes (by name).
    """
    if isinstance(p, (list, tuple, np.ndarray)):
        return p[key_or_idx]
    # dict-like
    if isinstance(p, dict):
        return p[key_or_idx]
    # object with attributes
    return getattr(p, key_or_idx)

def parameters(param: Any) -> Tuple[
    int, int, int, float, int, int,
    float, float, float, float,
    float, float, float, int,
    float, float, float,
    np.ndarray, int, float
]:
    """
    Python port of MATLAB `parameters.m`.

    Input
    -----
    `param` in the same order as MATLAB:
      [ DurationSimulation(years), Nstrains, Dimmunity(weeks),
        sigma, omega, x,
        Cperweek, Nagents, alpha,
        AgeDeath(years), BasicReproductionNumber ]

    Or a dict/dataclass with the same names.

    Returns (same order as MATLAB):
      Nagents, Nstrains, Nst, AgeDeath, NI0perstrain, NR0perstrain,
      Cpertimestep, MRpertimestep, Precovery, Pimmunityloss,
      Ptransmission, x, StrengthImmunity, Immunity,
      StrengthCrossImmunity, prevalence_in_migrants, CCC,
      time, Ntimesteps, dt_years
    """
    # ---- Read raw inputs (supports vector, dict, or object) ----
    DurationSimulation      = float(_get(param, 0) if not isinstance(param, dict) and not hasattr(param, 'DurationSimulation')
                                    else _get(param, 'DurationSimulation'))
    Nstrains_in             = int  (_get(param, 1) if not isinstance(param, dict) and not hasattr(param, 'Nstrains')
                                    else _get(param, 'Nstrains'))
    Dimmunity               = float(_get(param, 2) if not isinstance(param, dict) and not hasattr(param, 'Dimmunity')
                                    else _get(param, 'Dimmunity'))
    StrengthImmunity_raw    = float(_get(param, 3) if not isinstance(param, dict) and not hasattr(param, 'sigma')
                                    else _get(param, 'sigma'))                   # 'sigma' in your top script
    StrengthCross_raw       = float(_get(param, 4) if not isinstance(param, dict) and not hasattr(param, 'omega')
                                    else _get(param, 'omega'))                   # 'omega' in your top script
    x                       = float(_get(param, 5) if not isinstance(param, dict) and not hasattr(param, 'x')
                                    else _get(param, 'x'))
    Cperweek                = float(_get(param, 6) if not isinstance(param, dict) and not hasattr(param, 'Cperweek')
                                    else _get(param, 'Cperweek'))
    Nagents                 = int  (_get(param, 7) if not isinstance(param, dict) and not hasattr(param, 'Nagents')
                                    else _get(param, 'Nagents'))
    MR_per_week             = float(_get(param, 8) if not isinstance(param, dict) and not hasattr(param, 'alpha')
                                    else _get(param, 'alpha'))                   # 'alpha' used as migration rate per week
    AgeDeath                = float(_get(param, 9) if not isinstance(param, dict) and not hasattr(param, 'AgeDeath')
                                    else _get(param, 'AgeDeath'))
    BasicReproductionNumber = float(_get(param,10) if not isinstance(param, dict) and not hasattr(param, 'BasicReproductionNumber')
                                    else _get(param, 'BasicReproductionNumber'))

    # ---- Multiplier (kept as in MATLAB; currently = 1) ----
    multiplier = 1  # MATLAB had: (1+1)*(param(9)>0) + 1*(param(9)==0) (commented out)

    # ---- Time grid ----
    Endtime_weeks = DurationSimulation * WEEKS_PER_YEAR
    dt_weeks = 1.0 / 7.0
    dt_years = dt_weeks / WEEKS_PER_YEAR
    time = np.arange(0.0, Endtime_weeks + 1e-12, dt_weeks)  # include endpoint like 0:dt:Endtime
    Ntimesteps = time.size

    # ---- Initial seeding ----
    NI0perstrain = 10
    NR0perstrain = 10

    # ---- Epidemiological params ----
    Nstrains = int(Nstrains_in * multiplier)  # number of initial strains (after multiplier)
    Nst = int(Nstrains_in)                    # number of strains modeled
    # Durations (weeks)
    Dinfection = 2.0
    # Co-infection carrying capacity
    CCC = float(Nstrains_in)
    # Migration controls
    MR = MR_per_week                           # number of migrations per week
    prevalence_in_migrants = 0.1

    # ---- Immunity strengths, clamped to [0,1] ----
    StrengthImmunity = float(np.clip(StrengthImmunity_raw, 0.0, 1.0))
    Immunity = int(StrengthImmunity > 0)       # 0 = none, 1 = waning immunity present
    StrengthCrossImmunity = float(np.clip(StrengthCross_raw, 0.0, 1.0))

    # ---- Rates per week ----
    Rrecovery = 1.0 / Dinfection
    Rimmunityloss = 1.0 / Dimmunity
    Rdeath = 1.0 / AgeDeath / WEEKS_PER_YEAR   # per week

    # ---- Probabilities per timestep (dt_weeks) ----
    Precovery = 1.0 - np.exp(-dt_weeks * Rrecovery)
    Pimmunityloss = 1.0 - np.exp(-dt_weeks * Rimmunityloss)

    # ---- Migration per timestep ----
    MRpertimestep = MR * dt_weeks

    # ---- Contacts per timestep ----
    Cpertimestep = Cperweek * dt_weeks

    # ---- Base probability of transmission per contact ----
    # Ptransmission = (Rdeath + Rrecovery + MR / Nagents) * R0 / Cperweek
    Ptransmission = (Rdeath + Rrecovery + MR / Nagents) * BasicReproductionNumber / Cperweek

    # ---- Return tuple (order exactly matches MATLAB) ----
    return (
        Nagents, Nstrains, Nst, AgeDeath, NI0perstrain, NR0perstrain,
        Cpertimestep, MRpertimestep, Precovery, Pimmunityloss,
        Ptransmission, x, StrengthImmunity, Immunity,
        StrengthCrossImmunity, prevalence_in_migrants, CCC,
        time, Ntimesteps, dt_years
    )


def initialise_agents(params):
    """
    Python port of MATLAB:
      [AgentCharacteristics, ImmuneStatus, time] = initialise_agents(params)

    Returns
    -------
    AgentCharacteristics : (Nagents, Nstrains+1) float
        cols 0..Nstrains-1: infection copies for each strain (0/1/2/…)
        last col: age (years)
    ImmuneStatus : (Nagents, Nstrains) int {0,1}
        strain-specific immunity flags
    time : np.ndarray
        time grid (weeks), as returned by parameters(params)
    """
    (Nagents, Nstrains, Nst, AgeDeath, NI0perstrain, NR0perstrain,
     _Cpertimestep, _MRpertimestep, _Precovery, _Pimmunityloss,
     _Ptransmission, _x, _StrengthImmunity, _Immunity,
     _StrengthCrossImmunity, _prevalence_in_migrants, _CCC,
     time, _Ntimesteps, _dt_years) = parameters(params)

    Nagents   = int(Nagents)
    Nstrains  = int(Nstrains)
    Nst       = int(Nst)
    AgeDeath  = float(AgeDeath)
    NI0       = int(NI0perstrain)
    NR0       = int(NR0perstrain)

    # Allocate
    AgentCharacteristics = np.zeros((Nagents, Nstrains + 1), dtype=float)
    ImmuneStatus         = np.zeros((Nagents, Nstrains), dtype=int)

    # Ages ~ Uniform(0, AgeDeath)
    AgentCharacteristics[:, -1] = np.random.rand(Nagents) * AgeDeath

    # Fallback like MATLAB if too few agents to seed NI0 per strain
    if Nagents < NI0 * Nst:
        NI0 = 4
        NR0 = 4

    # Pools for sampling without replacement across strains
    pool_inf = np.arange(Nagents)  # for infections seeding
    pool_imm = np.arange(Nagents)  # for immunity seeding

    for i in range(Nst):  # 0..Nst-1 strains to seed
        if pool_inf.size < NI0 or pool_imm.size < NR0:
            break

        # choose positions within the remaining pools
        pos_inf = np.random.choice(pool_inf.size, size=NI0, replace=False)
        pos_imm = np.random.choice(pool_imm.size, size=NR0, replace=False)

        infected_agents = pool_inf[pos_inf]
        immune_agents   = pool_imm[pos_imm]

        # set one copy of strain i
        AgentCharacteristics[infected_agents, i] = 1.0
        ImmuneStatus[immune_agents, i] = 1

        # remove those agents from further seeding (across strains)
        pool_inf = np.delete(pool_inf, pos_inf)
        pool_imm = np.delete(pool_imm, pos_imm)

    return AgentCharacteristics, ImmuneStatus, time


def simulator(AgentCharacteristics, ImmuneStatus, params,
              specifyPtransmission: int = 0,
              cross_immunity_effect_on_coinfections: int = 1):
    """
    Python port of MATLAB simulator.m

    Inputs
    ------
    AgentCharacteristics : (Nagents, Nstrains+1) float
        cols 0..Nstrains-1: infection copies per strain
        last col: agent age (years)
    ImmuneStatus : (Nagents, Nstrains) int {0,1}
    params : same container you pass to parameters(params)
    specify Ptransmission : 1 to force Ptransmission=0.0301, else 0
    cross_immunity_effect_on_coinfections : 1 on, 0 off

    Returns
    -------
    SSPrev : (Nstrains, Ntimesteps)
    AgentsInfectedByKStrains : (Nstrains, Ntimesteps)
    """

    (Nagents, Nstrains, Nst, AgeDeath, _NI0, _NR0,
     Cpertimestep, MRpertimestep, Precovery, Pimmunityloss,
     Ptransmission, x, StrengthImmunity, Immunity,
     StrengthCrossImmunity, prevalence_in_migrants, CCC,
     time, Ntimesteps, dt_years) = parameters(params)

    Nagents   = int(Nagents)
    Nstrains  = int(Nstrains)
    Nst       = int(Nst)
    AgeDeath  = float(AgeDeath)
    CCC       = float(CCC)

    # Optionally override Ptransmission
    if specifyPtransmission == 1:
        Ptransmission = 0.0301

    # ---- Cross-immunity-accelerated recovery probability per step ----
    dt_weeks = 1.0/7.0  # from parameters.m
    Rrecovery = -np.log(1.0 - Precovery) / dt_weeks
    if StrengthCrossImmunity != 1:
        Rrecovery_cici = 1.0 / ((1.0 / Rrecovery) * (1.0 - StrengthCrossImmunity))
        Precovery_cici = 1.0 - np.exp(-dt_weeks * Rrecovery_cici)
    else:
        Precovery_cici = 1.0

    # ---- Pre-generated random streams (mirrors pregenerate_random_numbers.m) ----
    ContactRand = np.random.poisson(Cpertimestep, size=(1_000_000, 1)).astype(int)
    MRRand      = np.random.poisson(MRpertimestep, size=(1_000_000, 1)).astype(int)
    SamplingU   = np.random.rand(1_000_000, 1)
    countCR = 0  # contacts
    countMR = 0  # migrants
    countU  = 0  # generic uniforms

    def _takeU(n):
        nonlocal SamplingU, countU
        end = countU + n
        if end > len(SamplingU):
            SamplingU = np.random.rand(1_000_000, 1)
            countU = 0
            end = n
        out = SamplingU[countU:end, 0]
        countU = end
        return out

    def _takeCR():
        nonlocal ContactRand, countCR
        x = ContactRand[countCR, 0]
        countCR += 1
        if countCR >= len(ContactRand):
            ContactRand = np.random.poisson(Cpertimestep, size=(1_000_000, 1)).astype(int)
            countCR = 0
        return x

    def _takeMR():
        nonlocal MRRand, countMR
        x = MRRand[countMR, 0]
        countMR += 1
        if countMR >= len(MRRand):
            MRRand = np.random.poisson(MRpertimestep, size=(1_000_000, 1)).astype(int)
            countMR = 0
        return x

    # ---- Outputs ----
    SSPrev = np.zeros((Nstrains, Ntimesteps), dtype=float)
    AgentsInfectedByKStrains = np.zeros((Nstrains, Ntimesteps), dtype=float)

    # t = 0
    BB = AgentCharacteristics[:, :Nstrains]
    SSPrev[:, 0] = BB.sum(axis=0)

    tot0 = BB.sum()
    if tot0 > 1:
        kvec = BB.sum(axis=1).astype(int)
        kvec = kvec[kvec != 0]
        if kvec.size:
            K, counts = np.unique(kvec, return_counts=True)
            AgentsInfectedByKStrains[K - 1, 0] = counts
    elif tot0 == 1:
        AgentsInfectedByKStrains[0, 0] = 1

    # Tracks “fast recovery” flags (CICI) for each (agent, strain)
    CICI = np.zeros_like(BB)

    # ---- Main time loop ----
    for t in range(Ntimesteps - 1):
        CurrentAC  = AgentCharacteristics.copy()
        CurrentImm = ImmuneStatus.copy()
        DD = CurrentAC[:, :Nst]  # infections per strain at start of step

        # ===== RECOVERY =====
        inf_norm = (DD > 0) & (CICI == 0)
        inf_cici = (DD > 0) & (CICI > 0)

        r_n_rows, r_n_cols = np.where(inf_norm)
        if r_n_rows.size:
            rec = (np.random.rand(r_n_rows.size) < Precovery)
            AgentCharacteristics[r_n_rows[rec], r_n_cols[rec]] = 0
            # only normal recoveries gain ss-immunity
            ImmuneStatus[r_n_rows[rec], r_n_cols[rec]] = 1 * Immunity

        r_c_rows, r_c_cols = np.where(inf_cici)
        if r_c_rows.size:
            rec = (np.random.rand(r_c_rows.size) < Precovery_cici)
            AgentCharacteristics[r_c_rows[rec], r_c_cols[rec]] = 0
            CICI[r_c_rows[rec], r_c_cols[rec]] = 0  # no immunity granted here

        # ===== WANING IMMUNITY =====
        w_rows, w_cols = np.where(CurrentImm == 1)
        if w_rows.size:
            lose = (np.random.rand(w_rows.size) < Pimmunityloss)
            ImmuneStatus[w_rows[lose], w_cols[lose]] = 0

        # ===== TRANSMISSION =====
        G = DD.sum(axis=1)
        infected_agents = np.where(G > 0)[0]
        if infected_agents.size:
            # base per-contact susceptibility, with co-infection resistance
            TotalInf = DD.sum(axis=1)
            P1 = Ptransmission * np.power((1.0 - TotalInf / CCC), x)
            P1 = np.clip(P1, 0.0, 1.0)
            P1 = np.repeat(P1[:, None], Nstrains, axis=1)
            InfectionProb = P1.copy()

            # strain-specific immunity
            if StrengthImmunity > 0:
                mask_ss = (CurrentImm == 1)
                InfectionProb[mask_ss] = P1[mask_ss] * (1.0 - StrengthImmunity)

            # cross-strain immunity (any immunity to any strain)
            if StrengthCrossImmunity > 0:
                any_imm = (CurrentImm == 1).any(axis=1)[:, None]
                mask_cs = (CurrentImm == 0) & np.repeat(any_imm, Nstrains, axis=1)
                InfectionProb[mask_cs] = P1[mask_cs] * (1.0 - StrengthCrossImmunity)

            for a in infected_agents:
                infecting_strains = np.where(DD[a, :] > 0)[0]
                X = _takeCR()  # contacts for this source agent
                if X <= 0:
                    continue

                # sample contacts (with replacement), avoid self
                U = _takeU(X)
                contacts = np.ceil(Nagents * U).astype(int) - 1
                selfmask = (contacts == a)
                if np.any(selfmask):
                    others = np.arange(Nagents)
                    others = others[others != a]
                    contacts[selfmask] = np.random.choice(others, size=selfmask.sum(), replace=True)

                # choose one transmitting strain per contact among agent's current strains
                U2 = _takeU(X)
                if infecting_strains.size == 1:
                    chosen = np.full(X, infecting_strains[0], dtype=int)
                else:
                    idx = np.ceil(infecting_strains.size * U2).astype(int) - 1
                    chosen = infecting_strains[idx]

                # success Bernoulli
                susc = InfectionProb[contacts, chosen]
                U3 = _takeU(X)
                success = (U3 < susc)

                if np.any(success):
                    contacts = contacts[success]
                    chosen   = chosen[success]
                    # dedupe same contact (keep first)
                    order = np.argsort(contacts)
                    contacts = contacts[order]
                    chosen   = chosen[order]
                    keep = np.concatenate([[True], np.diff(contacts) > 0])
                    contacts = contacts[keep]
                    chosen   = chosen[keep]

                    # increment copies from the snapshot state
                    AgentCharacteristics[contacts, chosen] = CurrentAC[contacts, chosen] + 1

                    if cross_immunity_effect_on_coinfections == 1:
                        temp = AgentCharacteristics[:, :Nstrains].copy()
                        temp[contacts, chosen] = 0               # remove the newly acquired strains
                        temp = temp[contacts, :]
                        temp[temp > 0] = 1                       # mark other extant strains
                        add = np.zeros_like(CICI)
                        add[contacts, :] = temp
                        CICI = np.clip(CICI + add, 0, 1)

                    # those contacts cannot be infected again in this pass
                    InfectionProb[contacts, :] = 0.0

        # ===== AGE, DEATH, BIRTH =====
        AgentCharacteristics[:, Nstrains] = dt_years + CurrentAC[:, Nstrains]
        dead = np.where(AgentCharacteristics[:, Nstrains] > AgeDeath)[0]
        if dead.size:
            AgentCharacteristics[dead, :Nstrains] = 0
            ImmuneStatus[dead, :] = 0
            AgentCharacteristics[dead, Nstrains] = 0.001
            CICI[dead, :] = 0

        # ===== MIGRATION =====
        NumMig = _takeMR()
        if NumMig > 0:
            if NumMig >= Nagents:
                migrants = np.random.permutation(Nagents)
            else:
                migrants = np.random.choice(Nagents, size=NumMig, replace=False)

            infected_mig = (np.random.rand(NumMig) < prevalence_in_migrants)
            n_im = int(infected_mig.sum())
            if n_im > 0:
                mig_strains = np.random.randint(0, Nst, size=n_im)  # 0..Nst-1

            cm = ci = 0
            for m in range(NumMig):
                idx = migrants[m]
                ImmuneStatus[idx, :] = 0
                CICI[idx, :] = 0
                AgentCharacteristics[idx, Nstrains] = np.random.rand() * AgeDeath
                AgentCharacteristics[idx, :Nstrains] = 0
                if infected_mig[cm]:
                    AgentCharacteristics[idx, mig_strains[ci]] = 1
                    ci += 1
                cm += 1

        # ===== RECORDING =====
        BB = AgentCharacteristics[:, :Nstrains]
        SSPrev[:, t + 1] = BB.sum(axis=0)

        tot = BB.sum()
        if tot > 1:
            kvec = BB.sum(axis=1).astype(int)
            kvec = kvec[kvec != 0]
            if kvec.size:
                K, counts = np.unique(kvec, return_counts=True)
                AgentsInfectedByKStrains[K - 1, t + 1] = counts
        elif tot == 1:
            AgentsInfectedByKStrains[0, t + 1] = 1

    return SSPrev, AgentsInfectedByKStrains


def simulator_v2(AgentCharacteristics, ImmuneStatus, params,
              specifyPtransmission: int = 0,
              cross_immunity_effect_on_coinfections: int = 1):
    """
    Python port of MATLAB simulator.m (optimised based on profiling)

    Inputs
    ------
    AgentCharacteristics : (Nagents, Nstrains+1) float
        cols 0..Nstrains-1: infection copies per strain
        last col: agent age (years)
    ImmuneStatus : (Nagents, Nstrains) int {0,1}
    params : same container you pass to parameters(params)
    specifyPtransmission : 1 to force Ptransmission=0.0301, else 0
    cross_immunity_effect_on_coinfections : 1 on, 0 off

    Returns
    -------
    SSPrev : (Nstrains, Ntimesteps)
    AgentsInfectedByKStrains : (Nstrains, Ntimesteps)
    """

    (Nagents, Nstrains, Nst, AgeDeath, _NI0, _NR0,
     Cpertimestep, MRpertimestep, Precovery, Pimmunityloss,
     Ptransmission, x, StrengthImmunity, Immunity,
     StrengthCrossImmunity, prevalence_in_migrants, CCC,
     time, Ntimesteps, dt_years) = parameters(params)

    Nagents   = int(Nagents)
    Nstrains  = int(Nstrains)
    Nst       = int(Nst)
    AgeDeath  = float(AgeDeath)
    CCC       = float(CCC)

    # Optionally override Ptransmission
    if specifyPtransmission == 1:
        Ptransmission = 0.0301

    # ---- Cross-immunity-accelerated recovery probability per step ----
    dt_weeks = 1.0/7.0  # from parameters.m
    Rrecovery = -np.log(1.0 - Precovery) / dt_weeks
    if StrengthCrossImmunity != 1:
        Rrecovery_cici = 1.0 / ((1.0 / Rrecovery) * (1.0 - StrengthCrossImmunity))
        Precovery_cici = 1.0 - np.exp(-dt_weeks * Rrecovery_cici)
    else:
        Precovery_cici = 1.0

    # ---- Pre-generated random streams (mirrors pregenerate_random_numbers.m) ----
    ContactRand = np.random.poisson(Cpertimestep, size=(1_000_000, 1)).astype(int)
    MRRand      = np.random.poisson(MRpertimestep, size=(1_000_000, 1)).astype(int)
    SamplingU   = np.random.rand(1_000_000, 1)
    countCR = 0  # contacts
    countMR = 0  # migrants
    countU  = 0  # generic uniforms

    def _takeU(n):
        nonlocal SamplingU, countU
        end = countU + n
        if end > len(SamplingU):
            SamplingU = np.random.rand(1_000_000, 1)
            countU = 0
            end = n
        out = SamplingU[countU:end, 0]
        countU = end
        return out

    def _takeCR():
        nonlocal ContactRand, countCR
        x = ContactRand[countCR, 0]
        countCR += 1
        if countCR >= len(ContactRand):
            ContactRand = np.random.poisson(Cpertimestep, size=(1_000_000, 1)).astype(int)
            countCR = 0
        return x

    def _takeMR():
        nonlocal MRRand, countMR
        x = MRRand[countMR, 0]
        countMR += 1
        if countMR >= len(MRRand):
            MRRand = np.random.poisson(MRpertimestep, size=(1_000_000, 1)).astype(int)
            countMR = 0
        return x

    # ---- Outputs ----
    SSPrev = np.zeros((Nstrains, Ntimesteps), dtype=float)
    AgentsInfectedByKStrains = np.zeros((Nstrains, Ntimesteps), dtype=float)

    # t = 0
    BB = AgentCharacteristics[:, :Nstrains]
    SSPrev[:, 0] = BB.sum(axis=0)

    tot0 = BB.sum()
    if tot0 > 1:
        kvec = BB.sum(axis=1).astype(int)
        kvec = kvec[kvec != 0]
        if kvec.size:
            K, counts = np.unique(kvec, return_counts=True)
            AgentsInfectedByKStrains[K - 1, 0] = counts
    elif tot0 == 1:
        AgentsInfectedByKStrains[0, 0] = 1

    # Tracks “fast recovery” flags (CICI) for each (agent, strain)
    CICI = np.zeros_like(BB)

    # ---- Main time loop ----
    for t in range(Ntimesteps - 1):
        CurrentAC  = AgentCharacteristics.copy()
        CurrentImm = ImmuneStatus.copy()
        DD = CurrentAC[:, :Nst]  # infections per strain at start of step

        # ===== RECOVERY =====
        inf_norm = (DD > 0) & (CICI == 0)
        inf_cici = (DD > 0) & (CICI > 0)

        r_n_rows, r_n_cols = np.where(inf_norm)
        if r_n_rows.size:
            rec = (np.random.rand(r_n_rows.size) < Precovery)
            AgentCharacteristics[r_n_rows[rec], r_n_cols[rec]] = 0
            # only normal recoveries gain ss-immunity
            ImmuneStatus[r_n_rows[rec], r_n_cols[rec]] = 1 * Immunity

        r_c_rows, r_c_cols = np.where(inf_cici)
        if r_c_rows.size:
            rec = (np.random.rand(r_c_rows.size) < Precovery_cici)
            AgentCharacteristics[r_c_rows[rec], r_c_cols[rec]] = 0
            CICI[r_c_rows[rec], r_c_cols[rec]] = 0  # no immunity granted here

        # ===== WANING IMMUNITY =====
        w_rows, w_cols = np.where(CurrentImm == 1)
        if w_rows.size:
            lose = (np.random.rand(w_rows.size) < Pimmunityloss)
            ImmuneStatus[w_rows[lose], w_cols[lose]] = 0

        # ===== TRANSMISSION =====
        # Reuse TotalInf for both G and susceptibility calculation
        TotalInf = DD.sum(axis=1)
        infected_agents = np.where(TotalInf > 0)[0]

        if infected_agents.size:
            # base per-contact susceptibility, with co-infection resistance
            P1 = Ptransmission * np.power((1.0 - TotalInf / CCC), x)
            P1 = np.clip(P1, 0.0, 1.0)

            # expand to (Nagents, Nstrains)
            InfectionProb = np.repeat(P1[:, None], Nstrains, axis=1)

            # strain-specific immunity
            if StrengthImmunity > 0:
                mask_ss = (CurrentImm == 1)
                InfectionProb[mask_ss] *= (1.0 - StrengthImmunity)

            # cross-strain immunity (any immunity to any strain)
            if StrengthCrossImmunity > 0:
                any_imm = (CurrentImm == 1).any(axis=1)[:, None]
                mask_cs = (CurrentImm == 0) & np.repeat(any_imm, Nstrains, axis=1)
                InfectionProb[mask_cs] *= (1.0 - StrengthCrossImmunity)

            for a in infected_agents:
                infecting_strains = np.where(DD[a, :] > 0)[0]
                if infecting_strains.size == 0:
                    continue

                X = _takeCR()  # contacts for this source agent
                if X <= 0:
                    continue

                # sample contacts (with replacement), avoid self efficiently
                U = _takeU(X)
                # map U into 0..Nagents-2 and then "skip" a
                contacts = (U * (Nagents - 1)).astype(int)
                contacts[contacts >= a] += 1  # now in 0..Nagents-1, excluding a

                # choose one transmitting strain per contact among agent's strains
                U2 = _takeU(X)
                if infecting_strains.size == 1:
                    chosen = np.empty(X, dtype=int)
                    chosen.fill(infecting_strains[0])
                else:
                    idx = (U2 * infecting_strains.size).astype(int)
                    idx[idx == infecting_strains.size] = infecting_strains.size - 1
                    chosen = infecting_strains[idx]

                # success Bernoulli
                susc = InfectionProb[contacts, chosen]
                U3 = _takeU(X)
                success = (U3 < susc)

                if np.any(success):
                    contacts = contacts[success]
                    chosen   = chosen[success]
                    # dedupe same contact (keep first)
                    order = np.argsort(contacts)
                    contacts = contacts[order]
                    chosen   = chosen[order]
                    keep = np.concatenate([[True], np.diff(contacts) > 0])
                    contacts = contacts[keep]
                    chosen   = chosen[keep]

                    # increment copies from the snapshot state
                    AgentCharacteristics[contacts, chosen] = CurrentAC[contacts, chosen] + 1

                    if cross_immunity_effect_on_coinfections == 1:
                        # --- Optimised cross-immunity update: only touched rows ---
                        temp = AgentCharacteristics[contacts, :Nstrains].copy()
                        temp[np.arange(contacts.size), chosen] = 0   # remove newly acquired strain
                        temp[temp > 0] = 1                           # mark other extant strains
                        CICI[contacts, :] = np.clip(
                            CICI[contacts, :] + temp,
                            0,
                            1,
                        )

                    # those contacts cannot be infected again in this pass
                    InfectionProb[contacts, :] = 0.0

        # ===== AGE, DEATH, BIRTH =====
        AgentCharacteristics[:, Nstrains] = dt_years + CurrentAC[:, Nstrains]
        dead = np.where(AgentCharacteristics[:, Nstrains] > AgeDeath)[0]
        if dead.size:
            AgentCharacteristics[dead, :Nstrains] = 0
            ImmuneStatus[dead, :] = 0
            AgentCharacteristics[dead, Nstrains] = 0.001
            CICI[dead, :] = 0

        # ===== MIGRATION =====
        NumMig = _takeMR()
        if NumMig > 0:
            if NumMig >= Nagents:
                migrants = np.random.permutation(Nagents)
            else:
                migrants = np.random.choice(Nagents, size=NumMig, replace=False)

            infected_mig = (np.random.rand(NumMig) < prevalence_in_migrants)
            n_im = int(infected_mig.sum())
            if n_im > 0:
                mig_strains = np.random.randint(0, Nst, size=n_im)  # 0..Nst-1

            cm = ci = 0
            for m in range(NumMig):
                idx = migrants[m]
                ImmuneStatus[idx, :] = 0
                CICI[idx, :] = 0
                AgentCharacteristics[idx, Nstrains] = np.random.rand() * AgeDeath
                AgentCharacteristics[idx, :Nstrains] = 0
                if infected_mig[cm]:
                    AgentCharacteristics[idx, mig_strains[ci]] = 1
                    ci += 1
                cm += 1

        # ===== RECORDING =====
        BB = AgentCharacteristics[:, :Nstrains]
        SSPrev[:, t + 1] = BB.sum(axis=0)

        tot = BB.sum()
        if tot > 1:
            kvec = BB.sum(axis=1).astype(int)
            kvec = kvec[kvec != 0]
            if kvec.size:
                K, counts = np.unique(kvec, return_counts=True)
                AgentsInfectedByKStrains[K - 1, t + 1] = counts
        elif tot == 1:
            AgentsInfectedByKStrains[0, t + 1] = 1

    return SSPrev, AgentsInfectedByKStrains

def div(SSP: np.ndarray) -> np.ndarray:
    """
    Reciprocal Simpson diversity over time.
    SSP: (Nstrains, T) counts per strain per timestep
    returns: (T,)
    """
    SSP1 = SSP - 1
    N = SSP.sum(axis=0)                 # (T,)
    D = N * (N - 1)
    sumSSP = (SSP * SSP1).sum(axis=0)   # (T,)
    with np.errstate(divide='ignore', invalid='ignore'):
        D = D / sumSSP
        inf_mask = ~np.isfinite(D) & (sumSSP == 0)
        D[inf_mask] = N[inf_mask]
        D[np.isnan(D)] = 0.0
    return D

def plotheatmap(x, y, z, vmax=90, xylim=(-0.03, 10.005, 0.4, 42.6)):
    z = np.array(z, dtype=float).copy()
    z[z == 0] = np.nan  # mask zeros like MATLAB

    # pixel-centered edges
    dx = (x[1] - x[0]) / 2.0 if len(x) > 1 else 0.5
    dy = (y[1] - y[0]) / 2.0 if len(y) > 1 else 0.5
    x_edges = np.concatenate([x - dx, [x[-1] + dx]])     # length M+1
    y_edges = np.concatenate([y - dy, [y[-1] + dy]])     # length N+1

    # reversed grayscale like your MATLAB map
    cmap = plt.cm.get_cmap('gray_r')

    # NOTE: C must be (M,N) where M=len(x_edges)-1 and N=len(y_edges)-1
    # We swapped axes so: x-axis=time (y_edges), y-axis=strain (x_edges)
    mesh = plt.pcolormesh(y_edges, x_edges, z, shading='flat',
                          vmin=0, vmax=vmax, cmap=cmap)
    plt.colorbar(mesh)
    if xylim:
        plt.axis(xylim)


def rmsd(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    return np.sqrt(np.mean(diff**2))

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return 100 * np.mean(np.abs((y_pred - y_true) / y_true))