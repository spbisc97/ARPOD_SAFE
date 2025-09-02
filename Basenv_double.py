"""
Rendezvous-SE2 Double Integrator Environment + SAC Training (Starter)
---------------------------------------------------------------------
- Underactuated SE2 chaser docking to a static target at the origin.
- Direct actuator commands: main body-forward thruster (0..1) and torque (-1..1).
- Double-integrator translational dynamics; rigid-body rotational dynamics.
- Minimal yet realistic features: actuator saturation, optional first-order lag,
  episode randomization, success & safety conditions, and shaped rewards.

Requires:
    pip install gymnasium numpy stable-baselines3 matplotlib

Usage:
    python this_file.py  # trains SAC for a short run; tweak params below.

Notes:
- Keep dt modest (0.1–0.2 s). We default to 0.2 s to keep rollouts fast.
- Start with easy curriculum (small spawn radius, low initial speed) and expand.
- Tune reward weights first; then expand realism (mass change, actuator lag, noise).
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import gymnasium as gym
import numpy as np

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except Exception:
    SAC = None  # allows import without SB3, e.g., for quick inspection

# ----------------------------
# Environment Implementation
# ----------------------------

@dataclass
class Params:
    dt: float = 0.2                   # [s] integrator step
    m: float = 6.0                    # [kg] 3–6U CubeSat mass range
    I: float = 0.15                   # [kg m^2] planar moment of inertia (about z)
    F_max: float = 0.1                # [N] max main thruster force
    tau_max: float = 0.02             # [N m] max torque (reaction wheel / RCS)
    use_actuator_lag: bool = False    # simple first-order lag on actuators
    u_tau: float = 0.2                # [s] thruster lag time constant
    u_tau_rot: float = 0.15           # [s] torque lag time constant

    # Episode settings
    max_steps: int = 1500             # 1500 * 0.2s = 300s episode max
    spawn_r_min: float = 3.0          # [m]
    spawn_r_max: float = 12.0         # [m]
    spawn_speed_max: float = 0.05     # [m/s]
    spawn_phi_max: float = math.radians(25)  # [rad]
    spawn_omega_max: float = math.radians(1) # [rad/s]

    # Success conditions
    dock_radius: float = 0.20         # [m]
    dock_speed: float = 0.05          # [m/s]
    dock_phi: float = math.radians(5) # [rad]

    # Safety envelope (terminate if violated)
    soft_bounds: float = 25.0         # [m] beyond this -> terminated (lost)

    # Reward weights
    w_dist: float = 1.0
    w_speed: float = 0.2
    w_phi: float = 0.05
    w_omega: float = 0.01
    w_u: float = 0.005
    w_approach: float = 0.25          # reward aligning velocity toward target

    # Big terminal rewards
    r_success: float = 200.0
    r_oom: float = -200.0             # out-of-bounds / crash

    # Noise/randomization
    obs_noise_std: float = 0.0        # set >0 to simulate sensors
    domain_rand: bool = True
    F_max_range: Tuple[float, float] = (0.08, 0.12)
    tau_max_range: Tuple[float, float] = (0.015, 0.025)
    m_range: Tuple[float, float] = (5.5, 7.0)
    I_range: Tuple[float, float] = (0.12, 0.20)


class RendezvousSE2Env(gym.Env):
    """Underactuated SE2 docking with a forward thruster and torque.

    State (obs): [x, y, vx, vy, phi, omega]
      - position and velocity are in target (inertial) frame
      - phi is body heading w.r.t. +x axis of inertial frame

    Action: [a_thrust, a_torque]
      - a_thrust in [-1, 1] mapped to throttle in [0, 1] (forward-only thruster)
      - a_torque in [-1, 1] mapped to physical torque in [-tau_max, +tau_max]
    """
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, params: Optional[Params] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.p = params or Params()
        self.render_mode = render_mode

        # State
        self.state = np.zeros(6, dtype=np.float32)

        # Actuator states (for lag)
        self._F = 0.0
        self._tau = 0.0

        # Randomized physical params per episode (if enabled)
        self._m = self.p.m
        self._I = self.p.I
        self._F_max = self.p.F_max
        self._tau_max = self.p.tau_max

        # Gym spaces
        high_obs = np.array([
            self.p.soft_bounds, self.p.soft_bounds,  # x,y
            3.0, 3.0,                                # vx,vy
            math.pi, 2.0                              # phi, omega
        ], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-high_obs, high=high_obs, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32),
                                           high=np.array([1.0, 1.0], dtype=np.float32), dtype=np.float32)
        self._step_count = 0

    # --------------- Utility ---------------
    def _randomize_physical_params(self):
        if not self.p.domain_rand:
            self._m, self._I, self._F_max, self._tau_max = self.p.m, self.p.I, self.p.F_max, self.p.tau_max
            return
        rng = np.random.default_rng()
        self._m = float(rng.uniform(*self.p.m_range))
        self._I = float(rng.uniform(*self.p.I_range))
        self._F_max = float(rng.uniform(*self.p.F_max_range))
        self._tau_max = float(rng.uniform(*self.p.tau_max_range))

    def _success(self, x, y, vx, vy, phi, omega) -> bool:
        dist = math.hypot(x, y)
        speed = math.hypot(vx, vy)
        return (dist <= self.p.dock_radius) and (speed <= self.p.dock_speed) and (abs(phi) <= self.p.dock_phi)

    def _out_of_bounds(self, x, y) -> bool:
        return (abs(x) > self.p.soft_bounds) or (abs(y) > self.p.soft_bounds)

    # --------------- Gym API ---------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self._step_count = 0
        #self._randomize_physical_params()
        rng = np.random.default_rng(seed)

        # Spawn on a ring sector with small initial speed and heading
        r = float(rng.uniform(self.p.spawn_r_min, self.p.spawn_r_max))
        theta = float(rng.uniform(-math.pi, math.pi))        
        # r = 10
        # theta = 0
        
        x, y = r * math.cos(theta), r * math.sin(theta)
        vx = float(rng.uniform(-self.p.spawn_speed_max, self.p.spawn_speed_max))
        vy = float(rng.uniform(-self.p.spawn_speed_max, self.p.spawn_speed_max))
        # vx = 0
        # vy = 0

        phi = float(rng.uniform(-self.p.spawn_phi_max, self.p.spawn_phi_max))
        omega = float(rng.uniform(-self.p.spawn_omega_max, self.p.spawn_omega_max))
        # phi = 0
        # omega = 0

        self.state = np.array([x, y, vx, vy, phi, omega], dtype=np.float32)
        self._F, self._tau = 0.0, 0.0

        obs = self._observe()
        info = {}
        return obs, info

    def _observe(self):
        obs = self.state.copy()
        if self.p.obs_noise_std > 0.0:
            obs += np.random.normal(0.0, self.p.obs_noise_std, size=obs.shape).astype(np.float32)
        np.clip(obs, self.observation_space.low, self.observation_space.high, out=obs)
        return obs

    def step(self, action: np.ndarray):
        self._step_count += 1
        a_thrust = float(np.clip(action[0], -1.0, 1.0))
        a_torque = float(np.clip(action[1], -1.0, 1.0))

        # Map to physical commands
        throttle = 0.5 * (a_thrust + 1.0)  # [-1,1] -> [0,1]
        F_cmd = throttle * self._F_max
        tau_cmd = a_torque * self._tau_max

        # Actuator dynamics (optional first-order lag)
        if self.p.use_actuator_lag:
            self._F += (self.p.dt / self.p.u_tau) * (F_cmd - self._F)
            self._tau += (self.p.dt / self.p.u_tau_rot) * (tau_cmd - self._tau)
        else:
            self._F, self._tau = F_cmd, tau_cmd

        x, y, vx, vy, phi, omega = map(float, self.state)
        dt = self.p.dt
        inv_m = 1.0 / self._m
        inv_I = 1.0 / self._I
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)

        # Continuous dynamics
        ax = (self._F * inv_m) * cos_phi
        ay = (self._F * inv_m) * sin_phi
        alpha = self._tau * inv_I

        # Semi-implicit (symplectic) Euler integration for better stability
        # example:
        # v_n+1 = v_n + acceleration(v_n, u_n) * dt
        # x_n+1 = u_n + v_n+1 * dt.
        # instead of
        # v_n+1 = v_n + acceleration(v_n, u_n) * dt
        # x_n+1 = u_n + v_n * dt.
        
        vx += ax * dt
        vy += ay * dt
        omega += alpha * dt

        x += vx * dt
        y += vy * dt
        phi += omega * dt
        # wrap angle to [-pi, pi]
        phi = (phi + math.pi) % (2 * math.pi) - math.pi
        # in-place update to avoid realloc
        s = self.state
        s[0] = x; s[1] = y; s[2] = vx; s[3] = vy; s[4] = phi; s[5] = omega
        self.state = s
        # Reward shaping
        dist = math.hypot(x, y)
        speed = math.hypot(vx, vy)
        # Radial approach (positive if velocity reduces distance)
        if dist > 1e-6:
        #     radial_unit = np.array([x, y]) / dist
        #     vel_vec = np.array([vx, vy])
        #     approach_speed = -float(radial_unit @ vel_vec)  # positive if approaching
            approach_speed = -((x * vx + y * vy) / dist)
        else:
            approach_speed = 0.0

        r = 0.0
        r -= self.p.w_dist * dist
        r -= self.p.w_speed * speed
        r -= self.p.w_phi * abs(phi)
        r -= self.p.w_omega * abs(omega)
        r += self.p.w_approach * approach_speed
        r -= self.p.w_u * ((abs(self._F) / (self._F_max + 1e-9)) + (abs(self._tau) / (self._tau_max + 1e-9)))

        terminated = False
        truncated = False
        info: Dict = {}

        if self._success(x, y, vx, vy, phi, omega):
            r += self.p.r_success
            terminated = True
            info["is_success"] = True

        if self._out_of_bounds(x, y):
            r += self.p.r_oom
            terminated = True
            info["failure"] = "out_of_bounds"

        if self._step_count >= self.p.max_steps:
            truncated = True

        obs = self._observe()
        return obs, float(r), terminated, truncated, info

    def render(self):
        # Minimal text render for quick debugging (extend to matplotlib if needed)
        x, y, vx, vy, phi, omega = self.state
        print(f"t={self._step_count*self.p.dt:6.2f}s  pos=({x:6.2f},{y:6.2f})  vel=({vx:5.2f},{vy:5.2f})  phi={math.degrees(phi):6.2f}°  w={math.degrees(omega):6.2f}°/s")


# ----------------------------
# Training Pipeline (SAC)
# ----------------------------

def make_env(params: Optional[Params] = None):
    def _thunk():
        env = RendezvousSE2Env(params=params)
        env = Monitor(env)
        return env
    return _thunk


def train_sac(total_timesteps: int = 300_000, n_envs: int = 8, seed: int = 42,
              normalize: bool = True, params: Optional[Params] = None,
              log_dir: str = "./logs_se2"):
    if SAC is None:
        raise RuntimeError("stable-baselines3 not installed. Run: pip install stable-baselines3")

    vec_env = DummyVecEnv([make_env(params) for _ in range(n_envs)])

    if normalize:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.995)

    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.02,
        gamma=0.995,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        seed=seed,
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    # Save model and normalization (if used)
    ts = int(time.time())
    model.save(f"sac_se2_{ts}")
    if normalize and isinstance(vec_env, VecNormalize):
        vec_env.save(f"vecnorm_se2_{ts}.pkl")

    try:
        vec_env.close()
    except Exception:
        pass




# ----------------------------
# Rollout + Plotting Utilities
# ----------------------------
import matplotlib.pyplot as plt


def rollout(env: RendezvousSE2Env, policy=None, max_steps: int = None, seed: int = 0):
    """Run one episode and collect history.
    - policy: callable(obs)->action in env.action_space; if None, sample random.
    Returns a dict with arrays.
    """
    obs, info = env.reset(seed=seed)
    done = False
    terminated = False
    truncated = False
    steps = 0
    max_steps = max_steps or env.p.max_steps

    hist = {
        "t": [],
        "x": [], "y": [], "vx": [], "vy": [], "phi": [], "omega": [],
        "F": [], "tau": [], "a0": [], "a1": [], "r": [], "dist": [], "speed": [],
    }

    while not (terminated or truncated) and steps < max_steps:
        if policy is None:
            action = env.action_space.sample()
        else:
            action = np.asarray(policy(obs), dtype=np.float32)

        # Step
        obs, reward, terminated, truncated, info = env.step(action)

        # Unpack state for logging
        x, y, vx, vy, phi, omega = env.state
        F = env._F
        tau = env._tau
        dist = float(np.hypot(x, y))
        speed = float(np.hypot(vx, vy))

        t = steps * env.p.dt
        hist["t"].append(t)
        hist["x"].append(float(x)); hist["y"].append(float(y))
        hist["vx"].append(float(vx)); hist["vy"].append(float(vy))
        hist["phi"].append(float(phi)); hist["omega"].append(float(omega))
        hist["F"].append(float(F)); hist["tau"].append(float(tau))
        hist["a0"].append(float(action[0])); hist["a1"].append(float(action[1]))
        hist["r"].append(float(reward))
        hist["dist"].append(dist); hist["speed"].append(speed)

        steps += 1

    for k in hist:
        hist[k] = np.asarray(hist[k])

    hist["is_success"] = bool(info.get("is_success", False))
    hist["failure"] = info.get("failure")
    # Include key environment parameters used for plotting to avoid relying on globals
    try:
        hist["dock_radius"] = float(env.p.dock_radius)
        hist["dock_speed"] = float(env.p.dock_speed)
        hist["dock_phi"] = float(env.p.dock_phi)
        hist["soft_bounds"] = float(env.p.soft_bounds)
    except Exception:
        # Fallbacks if env does not have expected attributes
        hist["dock_radius"] = hist.get("dock_radius", 0.20)
        hist["dock_speed"] = hist.get("dock_speed", 0.05)
        hist["dock_phi"] = hist.get("dock_phi", math.radians(5))
        hist["soft_bounds"] = hist.get("soft_bounds", 25.0)
    return hist


def make_policy_from_sb3(model):
    """Wrap a Stable-Baselines3 model into a callable(obs)->action suitable for rollout."""
    def _pi(obs):
        # SB3 expects batch; deterministic=True for evaluation
        action, _ = model.predict(obs, deterministic=True)
        return action
    return _pi


def plot_rollout(hist: dict, title: str = "SE2 Rendezvous Rollout", show: bool = True, save_path: str = None):
    """Plot XY trajectory, time series of distance/speed, attitude rates, and controls."""
    t = hist["t"]
    x, y = hist["x"], hist["y"]
    vx, vy = hist["vx"], hist["vy"]
    phi, omega = hist["phi"], hist["omega"]
    F, tau = hist["F"], hist["tau"]
    dist, speed = hist["dist"], hist["speed"]

    # Pull plotting params from hist (recorded during rollout)
    dock_radius = float(hist.get("dock_radius", 0.20))
    dock_speed = float(hist.get("dock_speed", 0.05))
    dock_phi = float(hist.get("dock_phi", math.radians(5)))
    soft_bounds = float(hist.get("soft_bounds", 25.0))

    # Figure 1: XY trajectory with headings (quiver every N)
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, linewidth=2)
    # Docking zone & bounds
    ax = plt.gca()
    dock = plt.Circle((0, 0), radius=dock_radius, fill=False, linestyle='--')
    ax.add_artist(dock)
    sb = soft_bounds
    ax.set_xlim(-sb, sb); ax.set_ylim(-sb, sb)
    ax.set_aspect('equal', adjustable='box')
    plt.title(f"{title} — XY path (success={hist['is_success']})")
    plt.xlabel("x [m]"); plt.ylabel("y [m]")
    # Heading arrows
    N = max(1, len(x)//25)
    u = np.cos(phi)[::N]
    v = np.sin(phi)[::N]
    plt.quiver(x[::N], y[::N], u, v, angles='xy', scale_units='xy', scale=5)
    if save_path:
        plt.savefig(f"{save_path}_xy.png", dpi=150, bbox_inches='tight')

    # Figure 2: Distance & speed
    plt.figure(figsize=(8, 4))
    plt.plot(t, dist, label='distance [m]')
    plt.plot(t, speed, label='speed [m/s]')
    plt.axhline(dock_radius, linestyle='--', linewidth=1)
    plt.axhline(dock_speed, linestyle='--', linewidth=1)
    plt.xlabel('time [s]'); plt.ylabel('metric'); plt.legend(); plt.title('Range & Speed vs. Time')
    if save_path:
        plt.savefig(f"{save_path}_range_speed.png", dpi=150, bbox_inches='tight')

    # Figure 3: Attitude & rate
    plt.figure(figsize=(8, 4))
    plt.plot(t, np.degrees(phi), label='phi [deg]')
    plt.plot(t, np.degrees(omega), label='omega [deg/s]')
    plt.axhline(np.degrees(dock_phi), linestyle='--', linewidth=1)
    plt.axhline(-np.degrees(dock_phi), linestyle='--', linewidth=1)
    plt.xlabel('time [s]'); plt.ylabel('attitude'); plt.legend(); plt.title('Attitude vs. Time')
    if save_path:
        plt.savefig(f"{save_path}_attitude.png", dpi=150, bbox_inches='tight')

    # Figure 4: Controls & reward
    plt.figure(figsize=(8, 4))
    plt.plot(t, F, label='F [N]')
    plt.plot(t, tau, label='tau [N·m]')
    plt.xlabel('time [s]'); plt.ylabel('actuators'); plt.legend(); plt.title('Actuators vs. Time')
    if save_path:
        plt.savefig(f"{save_path}_actuators.png", dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close('all')


# if __name__ == "__main__":
#     # Example: evaluate a random policy (replace with trained SB3 model load)
#     env = RendezvousSE2Env()

#     # To use a trained model, uncomment and set the path:
#     from stable_baselines3 import SAC
#     model = SAC.load("sac_se2_1756564950")
#     policy = make_policy_from_sb3(model)
#     hist = rollout(env, policy=policy, seed=123)

#     hist = rollout(env, policy=None, seed=123)
#     plot_rollout(hist, title="Debug Rollout", show=True, save_path=None)


# # ----------------------------
# # Quick smoke test & training
# # ----------------------------
# if __name__ == "__main__":
#     # Quick environment sanity check
#     env = RendezvousSE2Env()
#     obs, info = env.reset(seed=0)
#     for _ in range(5):
#         a = env.action_space.sample()
#         obs, r, term, trunc, info = env.step(a)
#         env.render()
#         if term or trunc:
#             obs, info = env.reset()

#     # Kick off a short train by default (adjust timesteps for serious runs)
#     if SAC is not None:
#         p = Params()
#         train_sac(total_timesteps=100_000, n_envs=4, params=p)
#     else:
#         print("Stable-Baselines3 not found; skipping training. Install it to train the agent.")









"""
Enhanced Training Loop with Periodic Evaluation & Plotting
---------------------------------------------------------
- Extends the quick training main to:
  * periodically evaluate the SAC agent on a single environment
  * generate rollout plots every N steps
  * log intermediate results
"""

import os
import matplotlib.pyplot as plt


def evaluate_and_plot(model, env, step, outdir="evals"):
    os.makedirs(outdir, exist_ok=True)
    policy = make_policy_from_sb3(model)
    hist = rollout(env, policy=policy, seed=step)
    save_path = os.path.join(outdir, f"eval_{step}")
    plot_rollout(hist, title=f"Eval at step {step}", show=False, save_path=save_path)
    print(f"[Eval] step={step}, success={hist['is_success']}, saved plots to {save_path}_*.png")


if __name__ == "__main__":
    """Enhanced training main with periodic evaluation plots.
    - Trains SAC with VecNormalize
    - Every `plot_every_steps`, runs a few eval episodes and saves plots under ./plots_sac/<ts>/
    """
    import os
    from stable_baselines3.common.callbacks import BaseCallback

    class TrainPlotCallback(BaseCallback):
        def __init__(self, params: Params, plot_every_steps: int = 50_000, eval_episodes: int = 2, out_dir: str = "./plots_sac", verbose: int = 0):
            super().__init__(verbose)
            self.params = params
            self.plot_every_steps = int(plot_every_steps)
            self.eval_episodes = int(eval_episodes)
            self.out_dir = out_dir
            self.ts = str(int(time.time()))
            os.makedirs(os.path.join(out_dir, self.ts), exist_ok=True)

        def _init_callback(self) -> None:
            # nothing special
            return None

        def _on_step(self) -> bool:
            if self.num_timesteps % self.plot_every_steps != 0:
                return True
            # Build a fresh eval env
            base_env = RendezvousSE2Env(self.params)
            eval_env = DummyVecEnv([lambda: Monitor(base_env)])
            # If training uses VecNormalize, mirror obs stats for proper eval
            if isinstance(self.model.get_env(), VecNormalize):
                eval_env = VecNormalize(eval_env, training=False, norm_obs=False, norm_reward=False)
                # eval_env.obs_rms = self.model.get_env().obs_rms

            # Roll a few eval episodes and plot
            pi = make_policy_from_sb3(self.model)
            for ep in range(self.eval_episodes):
                hist = rollout(base_env, policy=pi, seed=ep)
                tag = f"step_{self.num_timesteps}_ep_{ep}"
                save_path = os.path.join(self.out_dir, self.ts, tag)
                plot_rollout(hist, title=f"SAC eval @ {self.num_timesteps}", show=False, save_path=save_path)
            if self.verbose:
                print(f"[PlotCallback] Saved eval plots at {self.num_timesteps} steps.")
            return True

    # -------- Quick environment sanity check (optional) --------
    env_debug = RendezvousSE2Env()
    obs, info = env_debug.reset(seed=0)
    for _ in range(3):
        a = env_debug.action_space.sample()
        obs, r, term, trunc, info = env_debug.step(a)
        if term or trunc:
            obs, info = env_debug.reset()

    # -------- Training with periodic plots --------
    if SAC is not None:
        p = Params()
        n_envs = 4
        total_timesteps = 60_000_000

        # Vectorized env with normalization
        vec = DummyVecEnv([make_env(p) for _ in range(n_envs)])
        vec = VecNormalize(vec, norm_obs=False, norm_reward=True, clip_obs=10.0, gamma=0.995)

        model = SAC(
            policy="MlpPolicy",
            env=vec,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            tau=0.05,
            gamma=0.99,
            train_freq=(10, "step"),
            gradient_steps=-1,
            policy_kwargs=dict(net_arch=[256, 256, 64]),
            verbose=1,
            seed=42,
            tensorboard_log="./tb_logs_se2",
        )

        cb = TrainPlotCallback(params=p, plot_every_steps=10_000, eval_episodes=3, out_dir="./plots_sac", verbose=1)
        model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=cb)

        # Save artifacts
        ts = int(time.time())
        model.save(f"sac_se2_{ts}")
        if isinstance(vec, VecNormalize):
            vec.save(f"vecnorm_se2_{ts}.pkl")
    else:
        print("Stable-Baselines3 not found; skipping training. Install it to train the agent.")
