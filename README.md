# Evolution Simulator

A pure-Python (standard library + Pygame) simulation in which organisms forage,
compete, reproduce and **evolve on their own**. There is no fitness function
the engine optimises — organisms simply survive, starve, get hunted, die of old
age and sometimes reproduce. Evolution *emerges* from thousands of stochastic
combinations of those events.

## What This Simulation Is NOT

- **Not a neural-network / GA playground.** Mutation is not biased towards
  improvement, there is no fitness-directed optimization loop, and no gene is
  rewarded *a priori*. Selection acts only through survival and non-random
  reproduction.
- **Not a red queen arms-race script.** No event "injects" specific traits.
  Predators and prey *can* co-evolve (faster prey runs away better; better
  attack/speed predators hunt better) purely through differential survival.
- **Not a physical or ecological model.** Movement, energy and reproduction are
  deliberately simple, tuned for qualitative emergent behaviour and speed, not
  biological fidelity.
- **Not exempt from extinction.** A population can collapse to zero (food
  shortage, a cold snap, a predator outbreak). `--seed` changes everything;
  some seeds go extinct. That is the point.
- **Not infinitely scalable.** It is pure Python; large worlds and long runs
  are slow. See [Performance](#performance).

## Features

- Emergent evolution: heritable traits (speed, vision, size, metabolism,
  fertility, lifespan, efficiency, aggression, temperature tolerance, and a
  mutation-rate gene) change by mutation and recombination only.
- Procedural terrain (biomes: water / grassland / forest / barren) with an
  influence on fertility and movement cost.
- Finite, regenerating food patches — competition for food is real pressure.
- Seasonal temperature and environmental events (droughts, cold snaps, heat
  waves, resource booms/collapses) that stress populations and ecosystems.
- Optional predation: a second, naturally replenished population with
  attack/speed/size-based hunts, cooldowns and a hunger gate.
- Species detection by genetic similarity, diversity tracking, and a per-tick
  population/energy model that stays finite and bounded.
- Headless CLI (no graphics), parameter sweeps, JSON/CSV statistics, and a
  fully deterministic byte-for-byte save/resume format.
- Pygame visualisation (organisms coloured by trait or species, terrain map,
  fitness/diversity/population/speed charts, selection + ancestry panels).
- Lifecycle statistics per generation (births, deaths, mutations,
  reproductions, food consumed, achieved lifespan, deaths by cause), a
  run-level summary, and a capped, deterministic live event log.
- 87 unit/integration tests, strict `mypy`, and `ruff`-clean.

## Requirements & Install

Python 3.10+, `pygame >= 2.5`. No NumPy.

```sh
pip install -e .          # installs evolution-simulator + pygame
pip install -e ".[dev]"   # adds pytest, mypy, ruff

python -m evolution_sim                       # open the Pygame visualisation
python -m evolution_sim --help
python -m evolution_sim --headless --generations 100
```

## Quick Start

```sh
# 200 generations, no window, prints a progress line every 25 generations
python -m evolution_sim --headless --generations 200

# deterministic: same seed => identical state
python -m evolution_sim --headless --generations 10 --seed 7 --json-stats out.json

# over-ride any config field on the command line
python -m evolution_sim --headless --generations 50 --initial-population 300 --mutation-rate 0.1

# sweep a config field across values and seeds (headless, CSV output)
python -m evolution_sim --experiment mutation_rate \
    --experiment-values 0.02,0.05,0.10 --experiment-seeds 1,2,3 \
    --generations 80 --experiment-output results/

# resume a saved population exactly where it stopped
python -m evolution_sim --load saves/save.json --generations 50
```

The GUI controls are shown in the window's hint bar: `SPACE` pause, `N` single
step, `G` advance one generation, `T` switch organism colouring, `+`/`-`
simulation speed, `S` save, `L` load, click to select an organism, drag to pan,
mouse wheel to zoom, `ESC` to quit.

## CLI Reference

| Flag | Meaning |
| --- | --- |
| `--headless` | run without the Pygame window |
| `--generations N` | generations to run (default 50) |
| `--seed N` | random seed (default 42) |
| `--config PATH` | load a JSON config file (unknown keys ignored) |
| `--report-every N` | progress line every N generations |
| `--load PATH` | resume from a saved state |
| `--save PATH` | save state when the run completes |
| `--save-every N` | auto-save every N generations |
| `--json-stats PATH` | per-generation stats as JSON |
| `--csv-stats PATH` | per-generation stats as CSV |
| `--experiment FIELD` | sweep a config field (headless) |
| `--experiment-values a,b,c` | values to try for the sweep |
| `--experiment-seeds a,b,c` | seeds to run per value (default 1,2,3) |
| `--experiment-output DIR` | where to write sweep results |
| `--field value` | any other `SimulationConfig` field can be overridden |

Config-file values override built-in defaults; command-line values override
both.

## How the Simulation Works

### Tick order

Each tick (`Engine.step()`):

1. environment updates (events, food regrowth, temperature),
2. every organism ages, pays metabolism, suffers thermal stress and acts
   (graze, seek food, wander, flee, hunt),
3. feeding (everyone near a food patch ingests),
4. predation resolution (when enabled),
5. reproduction (mature, well-fed, cooled-down organisms mate),
6. deaths (starvation, stress, old age),
7. every `generation_length` ticks a generation ends: the population is
   replenished to `target_population` from the fittest survivors (when
   enabled), species are re-detected and stats recorded.

### World, terrain and food

- Terrain is generated with deterministic value noise from the seed: a small
  integer lattice, bilinear interpolation and a few octaves. Same seed, same
  world.
- Biomes: `water` (no food), `barren` (low fertility), `grass`, `forest` (high
  fertility). Water never hosts food patches; fertility scales regrowth.
- Food patches are placed stochastically over fertile cells. Each patch has a
  quantity, a maximum, a per-tick regen rate and nutrition. Total food is
  finite at any moment, so hungry organisms must compete.

### Organism & traits

Every organism has a **genome**: named genes on `[0, 1]` (`size` starts at
0.25, `mutation_rate` is capped at 0.5). Phenotype accessors map genes onto
behaviour:

| Trait | Effect |
| --- | --- |
| `speed` | step length and movement-energy cost (faster = costlier) |
| `vision` | how far food, predators and mates are noticed (`vision_base`…`vision_range`) |
| `size` | survival in fights; a predator's attack bonus vs. prey defence; used by prey aggression counters |
| `metabolism` | base upkeep `0.02 + 0.03·metabolism` per tick |
| `efficiency` | conversion of eaten food into energy; predator hunt efficacy |
| `fertility` | reserved for future models (valid chromosome) |
| `lifespan` | maps linearly to `lifespan_min_ticks`…`lifespan_max_ticks` |
| `aggression` | prey counter-attack chance and defence against predators |
| `temperature_tolerance` | reduces thermal stress in cold/heat |
| `mutation_rate` | own per-gene mutation probability |
| `attack` (predators) | hunt success chance |

Energy: `max_energy` cap, `starting_energy`, movement costs
(`movement_energy_cost · (0.5 + speed)`).
Terrain affects movement: stepping onto water costs `terrain_movement_water_penalty`
more energy than grassland, so the terrain is a genuine selection pressure on
movement/heuristics. Organisms rest below `rest_energy_threshold`, starve at 0.
Infants start with `energy` from the reproduction cost model, never below
`offspring_energy_floor`.

### Genetics & reproduction

Sexual reproduction: a mature (`min_age_to_reproduce`), energy-rich
(`reproduction_energy_threshold`) organism past its `reproduction_cooldown`
picks one of the nearest eligible same-kind mates. The child genome is a
**crossover** of the parents (`uniform`: gene-by-gene pick, or `blend`:
interpolation) then **mutated**:

- each gene mutates with probability derived from the parents' *mutation_rate
  gene* + the configured baseline;
- a mutated gene shifts by up to `mutation_strength` of its range, or with
  `point_mutation_chance` snaps to a brand-new random value;
- `mutation_rate` itself mutates, clamped to `[gene_mutation_rate_min,
  gene_mutation_rate_max]`.

Genomes are immutable (`MappingProxyType`); nothing ever edits a parent's genes
in place. Reproduction costs each parent a fraction of its current energy; the
child starts with a fraction of the parents' residual (floor
`offspring_energy_floor`), so reproduction is a real trade-off between
surviving and breeding. Every birth is recorded in a bounded FIFO pedigree
(`pedigree_record_limit`).

### Predation (disabled by default)

Enable with `--predators-enabled true`. Predators are a second population:

- **Hunger gate**: predators only hunt while below
  `predator_hunt_energy_fraction` of max energy.
- **Detection**: nearest prey within vision; move towards it, attack within
  `predator_attack_range + prey.size·0.4`.
- **Hunt resolution**: success probability
  `0.22 + 0.15·attack + 0.16·(pred.speed − prey.speed) + 0.08·pred.size −
  0.08·prey.size − 0.10·prey.aggression`, clamped to `[0.04, 0.9]`. A
  successful hunt feeds the predator (`predator_hunt_reward`, scaled by
  efficiency); failed hunts cost energy, and strong, aggressive prey may
  counter and wound the hunter. Every attempt sets a `predator_hunt_cooldown`.
- **Numbers**: predators reproduce up to `predator_max_population` and, at
  each generation, are replenished to ≈ `predator_replenish_fraction` of the
  prey count, so predator numbers track food availability instead of snow-
  balling. Predators have a higher metabolism
  (`predator_metabolism_multiplier`) and start from a biased genome
  (`predator_initial_speed_floor`).
- Prey notice predators at `flee_vision_factor × normal vision`, and faster
  prey flee while slower prey stand their ground (and may counter-attack).

### Climate & events

Temperature oscillates seasonally (`base_temperature ± seasonal_amplitude`
over `seasonal_period` ticks). `EventManager` probabilistically schedules
droughts, cold snaps, heat waves, resource booms and collapses — each with a
magnitude and duration. Events change food regrowth and/or temperature, i.e.
they are pure **selection pressure**; they never touch genomes. Thermal stress
costs energy and (above a threshold) health, with `temperature_tolerance`
mitigating it.

### Species, diversity & fitness

- At every generation the living population is clustered into species by
  mean genetic similarity across the six diversity traits, using a
  deterministic greedy clustering (`species_similarity_threshold`). Each
  organism gets a `species_id`; the UI can colour by species.
- **Diversity** is the mean normalised mean-absolute-deviation of genes — 0
  when everyone is identical, 1 when maximally spread.
- **Fitness** is a *reporting* metric, not a selection force:

  ```
  fitness = 0.40 · fecundity   (children produced / expectation)
          + 0.30 · lineage     (living children / expectation)
          + 0.20 · longevity   (age reached vs. genome-implied lifespan)
          + 0.10 · efficiency  (energy harvested / energy spent)
  ```

  All components are in `[0, 1]`, so fitness is a bounded weighted average.
  Selection itself happens purely through death and reproduction.

### Statistics, ledger & event log

Every recorded generation (`GenerationStats`) exposes the usual trait averages,
diversity, species count, predator/prey split — and lifecycle metrics tracked
across the generation's ticks:

- `births` / `deaths` and the mirror `reproductions` (a birth is a reproduction);
- `mutations` — the total number of genes actually mutated (counted in
  `genetics.mutate_with_count`, not inferred);
- `resources_consumed` — food ingested from patches this generation;
- `avg_lifespan` — the *achieved* lifespan, i.e. the mean age at death this
  generation (not the gene value);
- `avg_age`, `avg_energy` — descriptors of the living population;
- `deaths_by_cause` (`starvation`, `stress`, `old age`, `hunted`;
  `old age` ↔ `old_age`, see `statistics.DEATH_CAUSES`) and the derived
  `environmental_deaths` (thermal-stress deaths).

The engine keeps these in a `GenerationLedger` that resets at each generation
boundary; `History.run_summary()` aggregates the whole run into one
`RunSummary` (total births/deaths/mutations, food consumed, min/max/avg
population, overall mean lifespan, extinct flag).

A live **event log** (`analytics/observability.py`) records births, deaths,
mutations, generation boundaries and feeding into a capped FIFO
(`event_log_capacity`, 0 disables it; this is observational only). It is
deterministic for a given seed and is deliberately *not* persisted — it never
enters `EngineState`, so save/resume formats are unchanged.

### Replenishment

When `replenish_to_target` is enabled (default), each new generation keeps the
population within reach of `target_population` by birthing from the fittest
survivors (per kind — prey population is replenished from prey, predators from
predators). This keeps the demo running without balancing every seed by hand,
while *trait* evolution remains fully un-scripted.

## Determinism & Persistence

Everything that makes randomness is driven by one `random.Random` seeded from
the config. The same seed + config produces the **byte-for-byte same state** —
nothing depends on timing, hash ordering or the OS. This covers the world,
every organism, the recorded statistics, *and* the live event log.

- `save_state(engine, path)` / `load_state(path)` persist the full world —
  terrain, food patches, every organism, RNG state, event manager, history,
  pedigree. Loading and continuing is deterministic: continuing a restored
  engine from the same save always produces the same future.
- Format versioned (`FORMAT_VERSION = 1`), JSON-based.

## Configuration

`SimulationConfig` is a single frozen dataclass in
`evolution_sim/config.py` — the source of truth for every tunable. Key groups:

- **World / population**: `world_width`, `world_height`, `initial_population`,
  `target_population`, `max_population`, `seed`.
- **Generation**: `generation_length`, `replenish_to_target`.
- **Genetics**: `mutation_rate`, `mutation_strength`, `point_mutation_chance`,
  `crossover_mode`, `crossover_blend_alpha`, `gene_mutation_rate_min/max`.
- **Energy**: `starting_energy`, `max_energy`, `movement_energy_cost`,
  `ingestion_rate`, `rest_energy_threshold`, `eating_radius`.
- **Reproduction**: `reproduction_energy_threshold`,
  `reproduction_energy_cost_fraction`, `offspring_energy_fraction`,
  `offspring_energy_floor`, `reproduction_cooldown`, `mating_range`,
  `min_age_to_reproduce`.
- **Aging / vision / movement**: `lifespan_min_ticks`, `lifespan_max_ticks`,
  `vision_base`, `vision_range`, `wander_angle_sigma`, `flee_vision_factor`.
- **Climate / events**: `base_temperature`, `seasonal_amplitude`,
  `seasonal_period`, `temperature_stress_energy`, `temperature_stress_health`,
  `terrain_movement_water_penalty`, `events_enabled`, `event_frequency`,
  `event_min_duration`, `event_max_duration`.
- **Resources**: `resource_density`, `resource_quantity`,
  `resource_regen_rate`, `resource_nutrition`, `resource_initial_fill`.
- **Predators**: `predators_enabled`, `predator_initial_population`,
  `predator_max_population`, `predator_attack_range`, `predator_hunt_reward`,
  `predator_hunt_cooldown`, `predator_hunt_energy_fraction`,
  `predator_replenish_fraction`, `predator_metabolism_multiplier`,
  `predator_initial_speed_floor`, `prey_attack_penalty`.
- **Analytics**: `species_enabled`, `species_similarity_threshold`,
  `diversity_warning_threshold`, `ancestry_depth`, `pedigree_record_limit`,
  `report_every`, `event_log_capacity`.
- **Performance**: `spatial_cell_size` (spatial-hash cell size for the hot
  spatial queries), `dead_organism_budget` (how many corpses are kept around
  for the UI; older ones are dropped at each generation so long runs stay
  fast).

`FIELD_GROUPS` in `config.py` maps every field to one of these groups (used by
tooling and validated by the test-suite to stay complete).
`SimulationConfig.validate()` rejects inconsistent values with explicit
messages — e.g. `reproduction_energy_threshold > max_energy`,
`lifespan_min > lifespan_max`, `vision_range < vision_base`,
`spatial_cell_size ≤ 0`, negative budgets/capacities, and out-of-range gene
mutation bounds.

```sh
# JSON config file (any subset of fields; unknown keys are ignored)
python -m evolution_sim --config config.json --headless --generations 100

# or just command-line overrides
python -m evolution_sim --headless --generations 50 --temperature-stress-health 0.004
```

## Performance

Pure Python, one `SimulationConfig`-size world:

- Prey-only: roughly **100–240 ticks/sec** in the same order as the default
  world (≈ 2–9 s per 400-tick generation of 500 organisms — measured on a
  mid-range desktop).
- With predation enabled, spatial queries and per-predator decision-making add
  ~1.5–3× overhead (≈ 8–22 s per generation observed over a 16-generation
  run). Dead bodies accumulate over time in the organism pool; a configurable
  budget (`dead_organism_budget`) trims the oldest corpses each generation, so
  per-tick cost stops growing after a few generations instead of drifting up
  forever.
- 1000 generations is therefore typically tens of minutes, not seconds. For
  long studies prefer `--headless`, dump `--csv-stats`/`--json-stats`, and run
  a `resource_density`/`mutation_rate` sweep with `--experiment`.

The RNG is a single plain `random.Random`, so per-run cost is linear in ticks
× organisms with no thread/GIL surprises.

## Project Layout

```
evolution_sim/
├── cli.py                 # argparse CLI: headless / GUI / experiments
├── config.py              # SimulationConfig (frozen dataclass) + JSON/CLI overrides
├── main.py                # CLI entry point
├── __main__.py            # enables `python -m evolution_sim`
├── analytics/
│   ├── statistics.py      # trait stats, diversity, species, GenerationStats, RunSummary, ledger
│   ├── observability.py   # SimEvent / EventLog (capped, deterministic, not persisted)
│   ├── experiments.py     # parameter sweeps across seeds -> CSV/JSON
│   └── history.py         # per-generation record store
├── persistence/
│   └── save_load.py       # versioned, byte-for-byte deterministic save/resume
├── simulation/            # the engine — imports nothing from pygame
│   ├── engine.py          # tick loop, behaviour, feeding, predation, reproduction, deaths
│   ├── world.py           # world state, organisms & spatial hashing
│   ├── organism.py        # the organism dataclass + phenotype accessors
│   ├── genome.py          # immutable Genome, trait specs, similarity
│   ├── genetics.py        # crossover + mutation
│   ├── reproduction.py    # mate finding, offspring energy budget
│   ├── fitness.py         # reporting fitness metric + lifespan expectation
│   ├── events.py          # drought/cold/heat/boom/collapse event manager
│   ├── environment.py     # value-noise terrain, biomes, climate
│   └── resources.py       # food patches (consume / regrow / total food)
└── visualization/         # Pygame, entirely decoupled from the engine
    ├── app.py             # window loop, input, HUD
    ├── renderer.py        # terrain, food, organisms, selection
    ├── camera.py          # pan / zoom / world<->screen transforms
    ├── ui.py              # info / selection / ancestry panels + history charts
    └── colors.py          # palettes: traits (viridis), species, biomes
tests/                     # 87 tests: engine determinism, genetics, persistence...
```

The engine layer (`simulation/`) has **zero** pygame imports — the
visualisation is a pure consumer of `Engine`.

## Development

```sh
python -m pytest            # 87 tests, deterministic
python -m mypy evolution_sim  # strict typing, no untyped flyovers
python -m ruff check .        # lint
```

The test-suite asserts byte-for-byte determinism (identical history *and*
event log for a seed), that all organism energies stay bounded and finite
across generations, replenishment restores population, lifecycle tallies stay
self-consistent (e.g. `mutations` matches the event stream; births/deaths/
reproductions invariants), config field groups cover every field, and that
save→load→continue reproduces the same future exactly.

## License

MIT.