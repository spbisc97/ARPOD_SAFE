"""
Safe SQRL Training for SE2 Rendezvous (Double Integrator)
---------------------------------------------------------
- Reuses the environment and plotting utilities from `Basenv_double.py`.
- Trains a SafeGym SQRL agent with a simple safety signal.

Safety signal (default):
- unsafe = 1 when radial distance exceeds a margin of the soft bounds
  (|p| > unsafe_radius = 0.9 * soft_bounds) OR when an out-of-bounds termination happens.
- You can tune the margin or plug a different rule as needed.

Requires:
    pip install git+https://github.com/spbisc97/SafeGym.git
    pip install torch --index-url https://download.pytorch.org/whl/cpu
"""
from __future__ import annotations

import os
import time
import math
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

# Local env and utils
from Basenv_double import RendezvousSE2Env, Params, rollout, plot_rollout

# SafeGym SQRL
from safegym.models.sqrl import SQRLAgent
import torch
import pickle
import json


@dataclass
class SQRLConfig:
    total_timesteps: int = 300_000
    seed: int = 42
    batch_size: int = 256
    buffer_size: int = 1_000_000
    warmup_steps: int = 10_000
    train_every: int = 1
    gradient_steps: int = 1
    # SAC/SQRL core
    gamma: float = 0.99
    safe_gamma: float = 0.99
    tau: float = 0.02
    alpha: float = 0.2
    epsilon_safe: float = 0.02
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    auto_entropy_tuning: bool = True
    nu: float = 0.1
    auto_nu_tuning: bool = True
    target_entropy: str | float = "auto"
    actor_hidden: int = 256
    critic_hidden: int = 256
    alpha_lr: float = 1e-4
    nu_lr: float = 1e-5
    # Eval/plots
    eval_every_steps: int = 50_000
    eval_episodes: int = 2
    out_dir: str = "./plots_sqrl"
    # Safety rule
    unsafe_margin: float = 0.9  # fraction of soft_bounds -> unsafe if dist > margin*soft_bounds
    # Resume
    resume_from: Optional[str] = None  # path to checkpoint dir to resume from


def make_sqrl_agent(env: RendezvousSE2Env, cfg: SQRLConfig) -> SQRLAgent:
    agent = SQRLAgent(
        env=env,
        gamma=cfg.gamma,
        safe_gamma=cfg.safe_gamma,
        tau=cfg.tau,
        alpha=cfg.alpha,
        epsilon_safe=cfg.epsilon_safe,
        actor_learning_rate=cfg.actor_lr,
        critic_learning_rate=cfg.critic_lr,
        buffer_size=cfg.buffer_size,
        auto_entropy_tuning=cfg.auto_entropy_tuning,
        nu=cfg.nu,
        auto_nu_tuning=cfg.auto_nu_tuning,
        gradient_steps=cfg.gradient_steps,
        target_entropy=cfg.target_entropy,
        actor_hidden_size=cfg.actor_hidden,
        critic_hidden_size=cfg.critic_hidden,
        alpha_learning_rate=cfg.alpha_lr,
        nu_learning_rate=cfg.nu_lr,
    )
    return agent


def compute_unsafe(env: RendezvousSE2Env, info: dict, margin: float) -> int:
    # Unsafe if env declared out-of-bounds termination or if we exceed a distance margin
    if info.get("failure") == "out_of_bounds":
        return 1
    x, y, *_ = env.state
    dist = float(math.hypot(float(x), float(y)))
    unsafe_radius = float(env.p.soft_bounds) * float(margin)
    return int(dist > unsafe_radius)


def evaluate_and_plot_sqrl(agent: SQRLAgent, params: Params, step: int, out_dir: str, episodes: int = 2):
    ts_dir = os.path.join(out_dir, str(int(time.time())))
    os.makedirs(ts_dir, exist_ok=True)

    def policy(obs):
        return agent.select_action(obs)

    base_env = RendezvousSE2Env(params)
    for ep in range(episodes):
        hist = rollout(base_env, policy=policy, seed=ep)
        tag = f"step_{step}_ep_{ep}"
        save_path = os.path.join(ts_dir, tag)
        plot_rollout(hist, title=f"SQRL eval @ {step}", show=False, save_path=save_path)
    print(f"[SQRL Eval] step={step}, saved plots under {ts_dir}")


def train_sqrl(params: Optional[Params] = None, cfg: Optional[SQRLConfig] = None):
    p = params or Params()
    c = cfg or SQRLConfig()

    # Single-environment training (SQRLAgent is single-env)
    env = RendezvousSE2Env(p)
    agent = make_sqrl_agent(env, c)

    # Optionally resume
    if c.resume_from:
        _resume_from_checkpoint(agent, c.resume_from)

    obs, info = env.reset(seed=c.seed)
    episode_steps = 0

    os.makedirs(c.out_dir, exist_ok=True)
    start_time = time.time()

    for t in range(1, c.total_timesteps + 1):
        # Action selection
        if t <= c.warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)

        # Step env
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        unsafe = compute_unsafe(env, info, c.unsafe_margin)

        # Store transition
        agent.update_replay_buffer(obs, action, reward, next_obs, done, unsafe)

        obs = next_obs
        episode_steps += 1

        # Learn
        if t > c.warmup_steps and (t % c.train_every == 0):
            agent.train(batch_size=c.batch_size, pretrain=False)

        # Episode reset
        if done:
            obs, info = env.reset()
            episode_steps = 0

        # Periodic evaluation + plots
        if (t % c.eval_every_steps) == 0:
            evaluate_and_plot_sqrl(agent, p, step=t, out_dir=c.out_dir, episodes=c.eval_episodes)

        # Light progress
        if (t % 10_000) == 0:
            elapsed = time.time() - start_time
            print(f"[Progress] t={t}/{c.total_timesteps}  elapsed={elapsed/60:.1f} min")

    # Save full checkpoint (weights, optimizers, buffers, metadata)
    ts = int(time.time())
    ckpt_dir = f"sqrl_se2_{ts}"
    _save_checkpoint(agent, ckpt_dir, params=p, cfg=c)
    print(f"[Save] Saved SQRL checkpoint under ./{ckpt_dir}")



# --------------- Checkpointing Utils ---------------

def _save_checkpoint(agent: SQRLAgent, ckpt_dir: str, params: Params, cfg: SQRLConfig):
    os.makedirs(ckpt_dir, exist_ok=True)
    # Models
    torch.save(agent.actor.state_dict(), os.path.join(ckpt_dir, "actor.pt"))
    torch.save(agent.critic1.state_dict(), os.path.join(ckpt_dir, "critic1.pt"))
    torch.save(agent.critic2.state_dict(), os.path.join(ckpt_dir, "critic2.pt"))
    torch.save(agent.safety_critic.state_dict(), os.path.join(ckpt_dir, "safety_critic.pt"))
    # Optimizers (when present)
    torch.save(agent.actor_optimizer.state_dict(), os.path.join(ckpt_dir, "actor_optim.pt"))
    torch.save(agent.critic1_optimizer.state_dict(), os.path.join(ckpt_dir, "critic1_optim.pt"))
    torch.save(agent.critic2_optimizer.state_dict(), os.path.join(ckpt_dir, "critic2_optim.pt"))
    torch.save(agent.safety_critic_optimizer.state_dict(), os.path.join(ckpt_dir, "safety_critic_optim.pt"))
    if getattr(agent, "auto_entropy_tuning", False) and hasattr(agent, "alpha_optimizer"):
        torch.save(agent.alpha_optimizer.state_dict(), os.path.join(ckpt_dir, "alpha_optim.pt"))
    if getattr(agent, "auto_nu_tuning", False) and hasattr(agent, "log_nu_optimizer"):
        torch.save(agent.log_nu_optimizer.state_dict(), os.path.join(ckpt_dir, "nu_optim.pt"))
    # Scalars
    torch.save(agent.log_alpha, os.path.join(ckpt_dir, "log_alpha.pt"))
    torch.save(agent.log_nu, os.path.join(ckpt_dir, "log_nu.pt"))
    # Replay buffer
    try:
        with open(os.path.join(ckpt_dir, "replay_buffer.pkl"), "wb") as f:
            pickle.dump(agent.replay_buffer, f)
    except Exception as e:
        print(f"[Warn] Could not save replay buffer: {e}")
    # Meta
    meta = {
        "params": asdict(params),
        "cfg": {k: v for k, v in asdict(cfg).items() if k != "resume_from"},
        "ts": int(time.time()),
        "version": 1,
    }
    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def _resume_from_checkpoint(agent: SQRLAgent, ckpt_dir: str):
    def _try_load(path, load_fn, name):
        full = os.path.join(ckpt_dir, path)
        if os.path.exists(full):
            try:
                load_fn(full)
                print(f"[Resume] Loaded {name} from {path}")
            except Exception as e:
                print(f"[Resume] Failed to load {name}: {e}")

    # Models
    _try_load("actor.pt", lambda p: agent.actor.load_state_dict(torch.load(p, map_location=agent.device)), "actor")
    _try_load("critic1.pt", lambda p: agent.critic1.load_state_dict(torch.load(p, map_location=agent.device)), "critic1")
    _try_load("critic2.pt", lambda p: agent.critic2.load_state_dict(torch.load(p, map_location=agent.device)), "critic2")
    _try_load("safety_critic.pt", lambda p: agent.safety_critic.load_state_dict(torch.load(p, map_location=agent.device)), "safety_critic")
    # Optimizers
    _try_load("actor_optim.pt", lambda p: agent.actor_optimizer.load_state_dict(torch.load(p, map_location=agent.device)), "actor_optim")
    _try_load("critic1_optim.pt", lambda p: agent.critic1_optimizer.load_state_dict(torch.load(p, map_location=agent.device)), "critic1_optim")
    _try_load("critic2_optim.pt", lambda p: agent.critic2_optimizer.load_state_dict(torch.load(p, map_location=agent.device)), "critic2_optim")
    _try_load("safety_critic_optim.pt", lambda p: agent.safety_critic_optimizer.load_state_dict(torch.load(p, map_location=agent.device)), "safety_critic_optim")
    _try_load("alpha_optim.pt", lambda p: agent.alpha_optimizer.load_state_dict(torch.load(p, map_location=agent.device)), "alpha_optim")
    _try_load("nu_optim.pt", lambda p: agent.log_nu_optimizer.load_state_dict(torch.load(p, map_location=agent.device)), "nu_optim")
    # Scalars
    _try_load("log_alpha.pt", lambda p: setattr(agent, "log_alpha", torch.load(p, map_location=agent.device)), "log_alpha")
    _try_load("log_nu.pt", lambda p: setattr(agent, "log_nu", torch.load(p, map_location=agent.device)), "log_nu")
    # Replay buffer
    rb_path = os.path.join(ckpt_dir, "replay_buffer.pkl")
    if os.path.exists(rb_path):
        try:
            with open(rb_path, "rb") as f:
                agent.replay_buffer = pickle.load(f)
            print("[Resume] Loaded replay buffer")
        except Exception as e:
            print(f"[Resume] Failed to load replay buffer: {e}")

if __name__ == "__main__":
    # Default run: quick SQRL training
    params = Params()
    cfg = SQRLConfig(
        total_timesteps=300_000,
        eval_every_steps=50_000,
        eval_episodes=2,
        unsafe_margin=0.9,
        resume_from="sqrl_se2_1756662046",
    )
    train_sqrl(params=params, cfg=cfg)