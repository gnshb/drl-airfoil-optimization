# DRL Airfoil Optimization

Deep reinforcement learning for airfoil shape optimization.
A PPO agent iteratively modifies thickness and camber distributions of an airfoil to maximize its lift-to-drag ratio (L/D).

## How it works

1. The agent starts from a random NACA-like airfoil
2. Each step, it adjusts thickness and camber at 12 control points along the chord
3. An aerodynamic evaluator (neuralfoil or xfoil) computes lift, drag, and moment
4. The reward is the improvement in L/D, with penalties for constraint violations
5. After training, the best airfoil shape is exported as coordinates

The airfoil surface is reconstructed from control points using natural cubic spline interpolation in sqrt(x) space, which gives smooth C2-continuous surfaces with higher resolution near the leading edge.

## Setup

```bash
git clone <repo-url>
cd drl-airfoil-optimization
pip install -r requirements.txt
pip install -e .
```

## Quick start

Train a model:

```bash
python train.py
```

Generate an optimized airfoil:

```bash
python generate.py
```

That's it. Training takes a few minutes and saves a model to `models/`. Generation auto-finds the latest model, runs a rollout, and exports the best airfoil to `outputs/`.

## Training

```bash
python train.py
```

By default this trains with **neuralfoil** (a neural network surrogate for aerodynamic evaluation) for 20,000 timesteps using 4 parallel workers.

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--evaluator` | `neuralfoil` (fast, ~10x faster) or `xfoil` (panel method, more accurate) | `neuralfoil` |
| `--timesteps` | Total training timesteps | `20000` |
| `-j` | Number of parallel environment workers | `4` |
| `--alpha` | Angle of attack in degrees | `0.0` |
| `--re` | Reynolds number | `1000000` |
| `--model-out` | Path to save the trained model | `models/ppo_<evaluator>` |

### Examples

```bash
python train.py --evaluator xfoil --timesteps 50000
python train.py --timesteps 100000 -j 8
python train.py --alpha 2.0 --re 500000
```

### Outputs

After training, you get three files in `models/`:

- `ppo_<evaluator>.zip` — the trained PPO model
- `ppo_<evaluator>.json` — training configuration (so generation can reproduce the setup)
- `ppo_<evaluator>_rewards.png` — episode reward curve over training

## Generation

```bash
python generate.py
```

Loads the latest trained model from `models/`, runs a deterministic rollout, evaluates every step with xfoil, and exports the airfoil with the best L/D.

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Path to a specific model | latest `.zip` in `models/` |
| `--seed` | Random seed for initial airfoil shape | `0` |
| `--out-csv` | Output CSV path | `outputs/generated_airfoil.csv` |
| `--out-png` | Output PNG path | `outputs/generated_airfoil.png` |

### Examples

```bash
python generate.py --model models/ppo_xfoil
python generate.py --seed 42
python generate.py --out-png outputs/my_airfoil.png
```

### Output format

The CSV contains x,y coordinates of the airfoil surface, ordered from trailing edge along the upper surface to the leading edge, then back along the lower surface to the trailing edge. This is the standard Selig format used by xfoil and other tools.

## Evaluators

Two aerodynamic backends are supported:

- **neuralfoil** — Neural network surrogate trained on XFOIL data. Very fast (~10x), good for training. Requires `neuralfoil` package.
- **xfoil** — Industry-standard panel method. More accurate but slower. Final airfoil evaluation always uses xfoil regardless of which backend was used for training. Requires the `xfoil` Python package.

## Benchmarking evaluators

Compare xfoil vs neuralfoil speed and accuracy on random airfoils:

```bash
python -m drl_airfoil.benchmark
python -m drl_airfoil.benchmark --n-cases 100 --alpha 2.0
```

## Constraints

The agent operates under physical constraints:

**Hard constraints** (enforced every step):
- Non-negative thickness everywhere
- Camber bounded to 49% of local thickness
- Minimum trailing edge thickness (0.002c)
- Zero camber at leading edge

**Soft constraints** (penalized in reward):
- Maximum thickness must stay between 3% and 15% of chord
- Violations incur a reward penalty proportional to the violation magnitude

## Project structure

```
train.py                 # entry point: train a model
generate.py              # entry point: generate an airfoil

src/drl_airfoil/
    config.py            # configuration dataclasses
    geometry.py          # airfoil parameterization and surface interpolation
    constraints.py       # hard and soft thickness/camber constraints
    evaluators.py        # xfoil and neuralfoil wrappers
    env.py               # gymnasium reinforcement learning environment
    train.py             # PPO training loop and CLI
    generate.py          # airfoil generation from trained model
    benchmark.py         # evaluator speed/accuracy comparison
    plotting.py          # airfoil and reward curve plotting

tests/
    test_geometry.py     # surface interpolation and action tests
    test_constraints.py  # constraint enforcement tests
```

## Running tests

```bash
pip install -e .[dev]
pytest
```
