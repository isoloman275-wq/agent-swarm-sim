#!/usr/bin/env python3
"""
=====================================================================
 SURVIVAL OF THE FITTEST — AGENT SWARM EVOLUTIONARY SIMULATION  v3
=====================================================================
World layers:
  * PREY swarm  — forages limited, fluctuating food (genome: speed/sense/greed/eff)
  * PREDATORS   — separate trophic layer that HUNTS prey (genome: speed/sense/eff)
  * DISEASE     — pathogen that OUTBREAKS then spreads between crowded prey,
                  draining energy; prey evolve to avoid clustering
  * FLUCTUATING ENVIRONMENT — per-gen food regime (scarce/normal/abundant)
  * LIVE TERMINAL ANIMATION — clears + repaints a dashboard every gen
  * INTERACTIVE KEYS (live):  p = pause/resume   q = quit
  * EXPORT:  --csv FILE  (per-gen stats)   --frames DIR (replayable frame log)

Pure stdlib. Deterministic with --seed.
=====================================================================
"""
import argparse
import csv
import math
import os
import random
import select
import statistics
import sys
import termios
import time
import tty
from dataclasses import dataclass, field

GENES = ["speed", "sense", "greed", "efficiency"]
PGENES = ["speed", "sense", "efficiency"]

SPEED_MIN, SPEED_MAX = 0.004, 0.045
SENSE_MIN, SENSE_MAX = 0.04, 0.40
GREED_MIN, GREED_MAX = 0.01, 0.06
COST_BASE = 0.0040
COST_SPEED = 0.020
COST_SENSE = 0.012

REGIMES = {"scarce": 0.0004, "normal": 0.0015, "abundant": 0.0035}
REGIME_WEIGHTS = [0.30, 0.40, 0.30]
REGIME_NAMES = list(REGIMES.keys())
FOOD_CAP = 1.0

PRED_METAB = 2.0
PRED_CATCH_VALUE = 0.30
CATCH_RADIUS = 0.012
CATCH_PROB = 0.45

# ---- disease layer ----
DIS_OUTBREAK_PROB = 0.45   # chance an outbreak seeds each generation
DIS_OUTBREAK_N = 4          # initial infected when outbreak fires
DIS_INFECT_RADIUS = 0.050  # contact distance for transmission
DIS_INFECT_PROB = 0.30     # per-tick transmission chance on contact
DIS_INFECT_COST = 0.0012   # extra energy drained per tick while infected
DIS_RECOVER_PROB = 0.015   # per-tick chance to shake the infection


@dataclass
class Agent:
    genes: dict
    energy: float = 0.0
    alive: bool = True
    fitness: float = 0.0
    id: int = 0
    infected: bool = False

    @classmethod
    def random(cls, rng, aid, genome):
        return cls(genes={g: rng.random() for g in genome}, id=aid)

    def child(self, rng, aid, rate, scale):
        new = {}
        for g, val in self.genes.items():
            if rng.random() < rate:
                val = min(1.0, max(0.0, val + rng.gauss(0.0, scale)))
            new[g] = val
        return Agent(genes=new, id=aid)


def gene_to(g, vals):
    if g == "speed":
        return SPEED_MIN + (SPEED_MAX - SPEED_MIN) * vals
    if g == "sense":
        return SENSE_MIN + (SENSE_MAX - SENSE_MIN) * vals
    if g == "greed":
        return GREED_MIN + (GREED_MAX - GREED_MIN) * vals
    return vals


@dataclass
class World:
    w: float = 1.0
    h: float = 1.0
    food: list = field(default_factory=list)

    def torus_dist(self, ax, ay, bx, by):
        dx = abs(ax - bx)
        dy = abs(ay - by)
        if dx > 0.5:
            dx = 1.0 - dx
        if dy > 0.5:
            dy = 1.0 - dy
        return math.hypot(dx, dy)

    def step_food(self, regen):
        for i in range(len(self.food)):
            x, y, amt = self.food[i]
            self.food[i] = (x, y, min(FOOD_CAP, amt + regen))


def run_generation(prey, preds, world, rng, ticks, eat_radius, food_value,
                   catch_radius, catch_prob, regen, disease):
    ppos = [(rng.random(), rng.random()) for _ in prey]
    dpos = [(rng.random(), rng.random()) for _ in preds]
    for a in prey:
        a.alive = True
        a.energy = 1.0
        a.infected = False
    for d in preds:
        d.alive = True
        d.energy = 1.0

    # disease outbreak seeding
    if disease and rng.random() < DIS_OUTBREAK_PROB and prey:
        idxs = rng.sample(range(len(prey)), min(DIS_OUTBREAK_N, len(prey)))
        for i in idxs:
            prey[i].infected = True

    for _ in range(ticks):
        world.step_food(regen)
        # ---- prey forage ----
        for i, a in enumerate(prey):
            if not a.alive:
                continue
            ax, ay = ppos[i]
            spd = gene_to("speed", a.genes["speed"])
            sns = gene_to("sense", a.genes["sense"])
            grd = gene_to("greed", a.genes["greed"])
            eff = a.genes["efficiency"]

            target = None
            best = sns
            for (fx, fy, famt) in world.food:
                if famt <= 0:
                    continue
                d = world.torus_dist(ax, ay, fx, fy)
                if d <= best:
                    best = d
                    target = (fx, fy, famt)
            if target:
                fx, fy, famt = target
                dx = fx - ax
                dy = fy - ay
                if dx > 0.5:
                    dx -= 1.0
                elif dx < -0.5:
                    dx += 1.0
                if dy > 0.5:
                    dy -= 1.0
                elif dy < -0.5:
                    dy += 1.0
                dist = math.hypot(dx, dy)
                if dist < eat_radius:
                    bite = min(famt, grd)
                    for k in range(len(world.food)):
                        if world.food[k][0] == fx and world.food[k][1] == fy:
                            world.food[k] = (fx, fy, famt - bite)
                            break
                    a.energy += bite * eff * food_value
                else:
                    nx = (dx / dist) * spd if dist > 0 else 0.0
                    ny = (dy / dist) * spd if dist > 0 else 0.0
                    ppos[i] = ((ax + nx) % 1.0, (ay + ny) % 1.0)
            else:
                ppos[i] = ((ax + rng.uniform(-spd, spd)) % 1.0,
                           (ay + rng.uniform(-spd, spd)) % 1.0)
            a.energy -= (COST_BASE + COST_SPEED * spd + COST_SENSE * sns)
            if a.energy <= 0:
                a.energy = 0.0
                a.alive = False

        # ---- disease transmission (after movement) ----
        if disease:
            inf_pos = [(ppos[j][0], ppos[j][1]) for j, b in enumerate(prey)
                       if b.alive and b.infected]
            for j, b in enumerate(prey):
                if not b.alive or b.infected:
                    continue
                for (ix, iy) in inf_pos:
                    if world.torus_dist(ppos[j][0], ppos[j][1], ix, iy) < DIS_INFECT_RADIUS:
                        if rng.random() < DIS_INFECT_PROB:
                            b.infected = True
                        break
            for b in prey:
                if b.alive and b.infected:
                    b.energy -= DIS_INFECT_COST
                    if rng.random() < DIS_RECOVER_PROB:
                        b.infected = False
                    if b.energy <= 0:
                        b.energy = 0.0
                        b.alive = False

        # ---- predators hunt prey ----
        for q, d in enumerate(preds):
            if not d.alive:
                continue
            dx0, dy0 = dpos[q]
            spd = gene_to("speed", d.genes["speed"])
            sns = gene_to("sense", d.genes["sense"])
            eff = d.genes["efficiency"]

            target = None
            best = sns
            for i, a in enumerate(prey):
                if not a.alive:
                    continue
                d2 = world.torus_dist(dx0, dy0, ppos[i][0], ppos[i][1])
                if d2 <= best:
                    best = d2
                    target = i
            if target is not None:
                ix, iy = ppos[target]
                vx = ix - dx0
                vy = iy - dy0
                if vx > 0.5:
                    vx -= 1.0
                elif vx < -0.5:
                    vx += 1.0
                if vy > 0.5:
                    vy -= 1.0
                elif vy < -0.5:
                    vy += 1.0
                dist = math.hypot(vx, vy)
                if dist < catch_radius and rng.random() < catch_prob:
                    prey[target].alive = False
                    prey[target].energy = 0.0
                    d.energy += PRED_CATCH_VALUE * eff
                else:
                    nx = (vx / dist) * spd if dist > 0 else 0.0
                    ny = (vy / dist) * spd if dist > 0 else 0.0
                    dpos[q] = ((dx0 + nx) % 1.0, (dy0 + ny) % 1.0)
            else:
                dpos[q] = ((dx0 + rng.uniform(-spd, spd)) % 1.0,
                           (dy0 + rng.uniform(-spd, spd)) % 1.0)
            d.energy -= PRED_METAB * (COST_BASE + COST_SPEED * spd + COST_SENSE * sns)
            if d.energy <= 0:
                d.energy = 0.0
                d.alive = False

    for a in prey:
        a.fitness = (a.energy if a.alive else 0.0) + (0.5 if a.alive else 0.0)
    for d in preds:
        d.fitness = (d.energy if d.alive else 0.0) + (0.5 if d.alive else 0.0)

    infected_n = sum(1 for a in prey if a.alive and a.infected)
    return ppos, dpos, infected_n


def breed(agents, rng, n, mut_rate, mut_scale):
    ranked = sorted(agents, key=lambda x: x.fitness, reverse=True)
    pool = [a for a in ranked if a.fitness > 0]
    if not pool:
        pool = ranked[:max(1, len(ranked) // 4)] or ranked
    total = sum(a.fitness for a in pool) or 1.0
    children = []
    aid = 1
    while len(children) < n:
        pick = rng.choices(pool, weights=[a.fitness / total for a in pool])[0]
        children.append(pick.child(rng, aid, mut_rate, mut_scale))
        aid += 1
    return children


def sparkline(values, width=46, lo=None, hi=None):
    if not values:
        return ""
    lo = lo if lo is not None else min(values)
    hi = hi if hi is not None else max(values)
    rng = (hi - lo) or 1.0
    bars = " ▁▂▃▄▅▆▇█"
    out, step = [], max(1, len(values) // width)
    for i in range(0, len(values), step):
        v = values[i]
        out.append(bars[int((v - lo) / rng * (len(bars) - 1))])
    return "".join(out)


def render_frame(gen, gens, prey, preds, world, ppos, dpos, hist, regime,
                 infected_n, G=22):
    prey_alive = [a for a in prey if a.alive]
    pred_alive = [d for d in preds if d.alive]
    grid = [[" "] * G for _ in range(G)]
    for (fx, fy, amt) in world.food:
        if amt > 0.12:
            grid[int(fy * G) % G][int(fx * G) % G] = "·"
    for i, a in enumerate(prey):
        if a.alive:
            cx, cy = int(ppos[i][0] * G) % G, int(ppos[i][1] * G) % G
            grid[cy][cx] = "*" if a.infected else "o"
    for q, d in enumerate(preds):
        if d.alive:
            cx, cy = int(dpos[q][0] * G) % G, int(dpos[q][1] * G) % G
            grid[cy][cx] = "X"
    af = statistics.mean([a.fitness for a in prey if a.alive]) if prey_alive else 0.0
    mf = max((a.fitness for a in prey), default=0.0)
    psurv = len(prey_alive) / len(prey) * 100
    dsurv = len(pred_alive) / len(preds) * 100 if preds else 0.0
    lines = []
    lines.append("=" * 64)
    lines.append("  SURVIVAL OF THE FITTEST — LIVE  (predators · disease · flux env)")
    lines.append("=" * 64)
    lines.append(f"  gen {gen:>3}/{gens}   env: {regime.upper():<8}"
                 f"  prey {len(prey_alive):>2}/{len(prey)}  pred {len(pred_alive):>2}/{len(preds)}"
                 f"  ☣ {infected_n}")
    lines.append(f"  prey surv {psurv:>5.1f}%   pred surv {dsurv:>5.1f}%"
                 f"   avg_fit {af:.3f}   max_fit {mf:.3f}")
    lines.append("-" * 64)
    for row in grid:
        lines.append("  |" + "".join(row) + "|")
    lines.append("-" * 64)
    lines.append(f"  PREY GENES  " + "  ".join(
        f"{g[:4]}={statistics.mean(a.genes[g] for a in prey):.2f}" for g in GENES))
    if preds:
        lines.append(f"  PRED GENES  " + "  ".join(
            f"{g[:4]}={statistics.mean(d.genes[g] for d in preds):.2f}" for g in PGENES))
    lines.append("-" * 64)
    lines.append(f"  FITNESS  {sparkline(hist['fit'])}")
    lines.append(f"  PSURVIV  {sparkline([s*100 for s in hist['psurv']], lo=0, hi=100)}")
    if preds:
        lines.append(f"  DSURVIV  {sparkline([s*100 for s in hist['dsurv']], lo=0, hi=100)}")
    lines.append("  legend: X=predator  o=prey  *=INFECTED  ·=food")
    return "\n".join(lines)


# ---- live keyboard (non-blocking) ----
def _enable_raw():
    if not sys.stdin.isatty():
        return None
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    return old


def _disable_raw(old):
    if old is not None:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old)


def _kbhit():
    if not sys.stdin.isatty():
        return None
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--pred", type=int, default=8)
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--ticks", type=int, default=120)
    ap.add_argument("--food", type=int, default=14)
    ap.add_argument("--mut-rate", type=float, default=0.18)
    ap.add_argument("--mut-scale", type=float, default=0.08)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--delay", type=float, default=0.12)
    ap.add_argument("--no-predators", action="store_true")
    ap.add_argument("--no-disease", action="store_true")
    ap.add_argument("--stable-env", action="store_true")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--csv", type=str, default=None, help="write per-gen stats CSV")
    ap.add_argument("--frames", type=str, default=None, help="write replayable frames to DIR")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    npred = 0 if args.no_predators else args.pred
    disease = not args.no_disease

    world = World()
    for _ in range(args.food):
        world.food.append((rng.random(), rng.random(), rng.random() * 0.8 + 0.2))

    prey = [Agent.random(rng, i + 1, GENES) for i in range(args.pop)]
    preds = [Agent.random(rng, i + 1, PGENES) for i in range(npred)] if npred else []

    hist = {"fit": [], "psurv": [], "dsurv": [],
            "genes": {g: [] for g in GENES},
            "pgenes": {g: [] for g in PGENES}}

    csvf = csv.writer(open(args.csv, "w", newline="")) if args.csv else None
    if csvf:
        csvf.writerow(["gen", "env", "prey_surv", "pred_surv", "avg_fit",
                       "max_fit", "infected"] + GENES + ["p_" + g for g in PGENES])

    if not args.live:
        print("=" * 68)
        print(" SURVIVAL OF THE FITTEST — AGENT SWARM EVOLUTION v3")
        print("=" * 68)
        print(f" seed={args.seed} pop={args.pop} pred={npred} disease={'on' if disease else 'off'}"
              f" gens={args.gens} ticks/gen={args.ticks} food={args.food}"
              f" env={'stable' if args.stable_env else 'fluctuating'}")
        print("-" * 68)
        print(f"{'gen':>4} {'env':<8} {'psurv':>6} {'dsurv':>6} {'☣':>4} {'avg_fit':>8} {'max_fit':>8}  spd sen grd eff")
        print("-" * 68)

    raw_old = _enable_raw() if args.live else None
    paused = False
    quit_ = False
    try:
        for gen in range(1, args.gens + 1):
            regime = "normal" if args.stable_env else rng.choices(REGIME_NAMES, weights=REGIME_WEIGHTS)[0]
            regen = REGIMES[regime]

            ppos, dpos, infected_n = run_generation(
                prey, preds, world, rng, args.ticks,
                eat_radius=0.02, food_value=1.0, catch_radius=CATCH_RADIUS,
                catch_prob=CATCH_PROB, regen=regen, disease=disease)

            prey_alive = [a for a in prey if a.alive]
            pred_alive = [d for d in preds if d.alive] if preds else []
            psurv = len(prey_alive) / len(prey)
            dsurv = (len(pred_alive) / len(preds)) if preds else 0.0
            af = statistics.mean([a.fitness for a in prey if a.alive]) if prey_alive else 0.0
            mf = max((a.fitness for a in prey), default=0.0)

            hist["fit"].append(af)
            hist["psurv"].append(psurv)
            hist["dsurv"].append(dsurv)
            for g in GENES:
                hist["genes"][g].append(statistics.mean(a.genes[g] for a in prey))
            for g in PGENES:
                hist["pgenes"][g].append(statistics.mean(d.genes[g] for d in preds) if preds else 0.0)

            if csvf:
                csvf.writerow([gen, regime, f"{psurv*100:.1f}", f"{dsurv*100:.1f}",
                               f"{af:.3f}", f"{mf:.3f}", infected_n] +
                              [f"{hist['genes'][g][-1]:.3f}" for g in GENES] +
                              [f"{hist['pgenes'][g][-1]:.3f}" for g in PGENES])

            if args.live:
                text = render_frame(gen, args.gens, prey, preds, world, ppos, dpos,
                                    hist, regime, infected_n, G=22)
                os.system("clear")
                print(text)
            else:
                text = render_frame(gen, args.gens, prey, preds, world, ppos, dpos,
                                    hist, regime, infected_n, G=22)
            if args.frames:
                os.makedirs(args.frames, exist_ok=True)
                with open(os.path.join(args.frames, f"gen_{gen:03d}.txt"), "w") as f:
                    f.write(text)
                # ---- interactive keys ----
                while True:
                    ch = _kbhit()
                    if ch in ("p", "P"):
                        paused = not paused
                        print("\n  [PAUSED] press p to resume · q to quit\n" if paused
                              else "\n  [resumed]\n")
                    elif ch in ("q", "Q"):
                        quit_ = True
                        break
                    if not paused:
                        break
                    time.sleep(0.08)
                if quit_:
                    break
                time.sleep(args.delay)
            elif gen == 1 or gen % args.log_every == 0 or gen == args.gens:
                print(f"{gen:>4} {regime:<8} {psurv*100:>5.1f}% {dsurv*100:>5.1f}% "
                      f"{infected_n:>4} {af:>8.3f} {mf:>8.3f}  "
                      f"{hist['genes']['speed'][-1]:.2f} {hist['genes']['sense'][-1]:.2f} "
                      f"{hist['genes']['greed'][-1]:.2f} {hist['genes']['efficiency'][-1]:.2f}")

            if not prey_alive:
                prey = [Agent.random(rng, i + 1, GENES) for i in range(args.pop)]
            else:
                prey = breed(prey, rng, args.pop, args.mut_rate, args.mut_scale)
            if preds:
                if not pred_alive:
                    preds = [Agent.random(rng, i + 1, PGENES) for i in range(npred)]
                else:
                    preds = breed(preds, rng, npred, args.mut_rate, args.mut_scale)
    finally:
        if raw_old is not None:
            _disable_raw(raw_old)

    if not args.live:
        print("-" * 68)
    print()
    print("=" * 68)
    print(" FINAL REPORT")
    print("=" * 68)
    print(f" avg prey fitness (gen1 -> gen{args.gens}): "
          f"{hist['fit'][0]:.3f} -> {hist['fit'][-1]:.3f}")
    print(f" avg prey survival:  {statistics.mean(hist['psurv'])*100:.1f}%")
    if preds:
        print(f" avg pred survival:  {statistics.mean(hist['dsurv'])*100:.1f}%")
        print(" PRED GENE DRIFT:")
        for g in PGENES:
            print(f"   {g:<11} {sparkline(hist['pgenes'][g])}")
    print(" PREY GENE DRIFT:")
    for g in GENES:
        print(f"   {g:<11} {sparkline(hist['genes'][g])}")
    print("=" * 68)
    print(" simulation complete.")
    if args.csv:
        print(f" csv  -> {args.csv}")
    if args.frames:
        print(f" frames -> {args.frames}  (replay: python3 replay_frames.py {args.frames} --delay 0.2)")
    print("=" * 68)


if __name__ == "__main__":
    main()
