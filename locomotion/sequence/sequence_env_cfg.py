"""Four-stage command-sequence task built on the S1 velocity configuration.

The low-level policy still controls 12 joint targets; what changes is the task:
instead of tracking a randomly resampled velocity command, the agent must
complete a fixed sequence of four stages (forward, turn, forward, stop) and
the observation carries the stage context explicitly.

Only the environment is defined here.  Group A (single layer) and group B
(layered) both read the same 53-dim policy observation (48 base terms, with
the 3-dim velocity command replaced by a 2-dim stage command, plus a 4-dim
stage one-hot and a 2-dim target delta), so the comparison in
docs/RESEARCH.md uses identical task information.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from locomotion.isaac.velocity_env_cfg import Go2BlindVelocityEnvCfg
from locomotion.sequence import mdp as seq_mdp


@configclass
class Go2SequenceEnvCfg(Go2BlindVelocityEnvCfg):
    """Stage machine on flat ground, blind observation plus stage context."""

    def __post_init__(self):
        super().__post_init__()

        # The stage machine supplies [vx, wz]; the random command sampler is off.
        self.commands.base_velocity = None

        # Observation: replace the command term and append stage context.
        self.observations.policy.velocity_commands = ObsTerm(func=seq_mdp.stage_command)
        self.observations.policy.stage_onehot = ObsTerm(func=seq_mdp.stage_onehot)
        self.observations.policy.target_delta = ObsTerm(func=seq_mdp.target_delta)

        # Reward: drop velocity tracking, keep posture/smoothness terms, add the
        # stage progress term with the completion bonus folded in.
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        # feet_air_time reads the velocity command to decide when to reward
        # swing; without a command manager it cannot run.  Other gait and
        # smoothness terms (feet_slide, torques, action rate) remain.
        self.rewards.feet_air_time = None
        self.rewards.stage = RewTerm(func=seq_mdp.stage_reward, weight=1.0)

        # Termination: completing the sequence ends the episode successfully;
        # base contact and the time-out remain as failures.
        self.terminations.sequence_complete = DoneTerm(func=seq_mdp.sequence_complete)

        # Per-episode stage state.
        self.events.reset_stage = EventTerm(func=seq_mdp.reset_stage, mode="reset")

        # A four-stage sequence needs more than the 20 s tracking episode.
        self.episode_length_s = 60.0
