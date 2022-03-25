import tensorflow as tf
from tensorflow.keras.optimizers.schedules import (
    PiecewiseConstantDecay,
)
from tensorflow.keras import mixed_precision as prec
from tensorflow_text import sliding_window

import common
import expl


class Agent(common.Module):
    def __init__(self, config, obs_space, act_space, step):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.step = step
        self.tfstep = tf.Variable(int(self.step), tf.int64)
        self.wm = WorldModel(config, obs_space, self.tfstep)
        self._task_behavior = ActorCritic(config, self.act_space, self.tfstep)
        if config.expl_behavior == "greedy":
            self._expl_behavior = self._task_behavior
        else:
            self._expl_behavior = getattr(expl, config.expl_behavior)(
                self.config,
                self.act_space,
                self.wm,
                self.tfstep,
                lambda seq: self.wm.heads["reward"](seq["feat"]).mode(),
            )

    @tf.function
    def policy(self, obs, state=None, mode="train"):
        obs = tf.nest.map_structure(tf.tensor, obs)
        tf.py_function(
            lambda: self.tfstep.assign(int(self.step), read_value=False), [], []
        )
        if state is None:
            latent = self.wm.rssm.initial(len(obs["reward"]))
            af_latent = self.wm.af_rssm.initial(len(obs["reward"]))
            action = tf.zeros((len(obs["reward"]),) + self.act_space.shape)
            state = af_latent, latent, action
        af_latent, latent, action = state

        af_sample = (mode == "train") or not self.config.eval_state_mean
        embed = self.wm.encoder(self.wm.preprocess(obs))
        af_latent, _ = self.wm.af_rssm.obs_step(
            af_latent, action, embed, obs["is_first"], af_sample
        )
        af_embed = self.wm.af_rssm.get_feat(af_latent)
        sample = (mode == "train") or not self.config.eval_state_mean
        if self.config.concat_embed:
            af_embed = tf.concat([embed, af_embed], -1)
        latent, _ = self.wm.rssm.obs_step(
            latent, action, af_embed, obs["is_first"], sample
        )
        feat = self.wm.rssm.get_feat(latent)
        if mode == "eval":
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
            noise = self.config.eval_noise
        elif mode == "explore":
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
            noise = self.config.expl_noise
        elif mode == "train":
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
            noise = self.config.expl_noise
        action = common.action_noise(action, noise, self.act_space)
        outputs = {"action": action}
        state = (af_latent, latent, action)
        return outputs, state

    @tf.function
    def train(self, data, state=None):
        metrics = {}
        state, outputs, mets = self.wm.train(data, state)
        metrics.update(mets)
        start = outputs["post"]
        reward = lambda seq: self.wm.heads["reward"](seq["feat"]).mode()
        metrics.update(
            self._task_behavior.train(self.wm, start, data["is_terminal"], reward)
        )
        if self.config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, outputs, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        return state, metrics

    @tf.function
    def report(self, data):
        report = {}
        data = self.wm.preprocess(data)
        for key in self.wm.heads["decoder"].cnn_keys:
            name = key.replace("/", "_")
            report[f"openl_{name}"] = self.wm.video_pred(data, key)
        return report

    def save_all(self, logdir):
        self.save(logdir / "variables.pkl")
        self.wm.af_rssm.save(logdir / "af_rssm_variables.pkl", verbose=False)
        self.wm.rssm.save(logdir / "rssm_variables.pkl", verbose=False)
        self.wm.encoder.save(logdir / "encoder_variables.pkl", verbose=False)
        self.wm.heads["decoder"].save(logdir / "decoder_variables.pkl", verbose=False)
        self.wm.heads["reward"].save(logdir / "reward_variables.pkl", verbose=False)
        self._task_behavior.actor.save(logdir / "actor_variables.pkl", verbose=False)
        self._task_behavior.critic.save(logdir / "critic_variables.pkl", verbose=False)


class WorldModel(common.Module):
    def __init__(self, config, obs_space, tfstep):
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        self.config = config
        self.tfstep = tfstep
        self.af_rssm = common.EnsembleRSSM(**config.af_rssm)
        self.rssm = common.EnsembleRSSM(**config.rssm)
        self.encoder = common.Encoder(shapes, **config.encoder)

        self.heads = {}
        self.heads["decoder"] = common.Decoder(shapes, **config.decoder)
        self.heads["reward"] = common.MLP([], **config.reward_head)
        if config.pred_discount:
            self.heads["discount"] = common.MLP([], **config.discount_head)
        for name in config.grad_heads:
            assert name in self.heads, name

        self.tf_queue_step = tf.Variable(int(0), dtype=tf.int32)
        self.tf_queue_size = tf.Variable(int(config.queue_size), dtype=tf.int32)
        self.random_projection_matrix = tf.Variable(
            tf.random.normal(
                (
                    config.rssm.deter + config.rssm.stoch * config.rssm.discrete,
                    config.queue_dim,
                ),
                mean=0.0,
                stddev=1.0 / config.queue_dim,
                dtype=prec.global_policy().compute_dtype,
            ),
            trainable=False,
        )
        self.queue = tf.Variable(
            tf.zeros(
                (config.queue_size, config.queue_dim),
                dtype=prec.global_policy().compute_dtype,
            ),
            trainable=False,
        )
        self.intr_rewnorm = common.StreamNorm(**self.config.intr_reward_norm)

        custom_enc_opt_config = {
            k: v for k, v in self.config.enc_model_opt.items() if k != "lr"
        }
        if self.config.enc_lr_type == "no_pretrain":
            learning_rate_fn = PiecewiseConstantDecay(
                boundaries=[self.config.pretrain],
                values=[0.0, self.config.enc_model_opt.lr],
            )
            custom_enc_opt_config["lr"] = learning_rate_fn
        else:
            custom_enc_opt_config["lr"] = self.config.enc_model_opt.lr

        self.enc_model_opt = common.Optimizer("enc_model", **custom_enc_opt_config)
        self.model_opt = common.Optimizer("model", **config.model_opt)

    def construct_queue(self, seq_feat):
        seq_size = tf.shape(seq_feat)[0]
        self.queue[seq_size:].assign(self.queue[:-seq_size])
        self.queue[:seq_size].assign(seq_feat)

        self.tf_queue_step.assign(self.tf_queue_step + seq_size)
        self.tf_queue_step.assign(
            tf.reduce_min([self.tf_queue_step, self.tf_queue_size])
        )
        return self.queue[: self.tf_queue_step]

    def train(self, data, state=None):
        with tf.GradientTape(persistent=True) as model_tape:
            model_loss, state, outputs, metrics = self.loss(data, state)
        enc_modules = [self.encoder, self.af_rssm]
        modules = [self.rssm, *self.heads.values()]
        metrics.update(self.enc_model_opt(model_tape, model_loss, enc_modules))
        metrics.update(self.model_opt(model_tape, model_loss, modules))
        del model_tape
        return state, outputs, metrics

    def loss(self, data, state=None):
        data = self.preprocess(data)
        embed = self.encoder(data)
        af_post, af_prior = self.af_rssm.observe(
            embed, data["action"], data["is_first"], state, sample=True
        )
        af_embed = self.af_rssm.get_feat(af_post)
        af_feat = af_embed
        if self.config.concat_embed:
            af_embed = tf.concat([embed, af_embed], -1)
        post, prior = self.rssm.observe(
            af_embed, data["action"], data["is_first"], state, sample=True
        )
        kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl)
        assert len(kl_loss.shape) == 0
        af_kl_loss, af_kl_value = self.rssm.kl_loss(af_post, af_prior, **self.config.kl)
        assert len(af_kl_loss.shape) == 0
        likes = {}
        losses = {"kl": kl_loss, "af_kl": af_kl_loss}
        feat = self.rssm.get_feat(post)

        ###############################################################################
        ### Intrinsic Bonus Computation
        seq_feat = af_feat
        seq_feat = sliding_window(seq_feat, self.config.intr_seq_length, axis=1).mean(
            axis=2
        )
        seq_feat = tf.matmul(seq_feat, self.random_projection_matrix)
        b, t, d = (tf.shape(seq_feat)[0], tf.shape(seq_feat)[1], tf.shape(seq_feat)[2])
        seq_feat = tf.reshape(seq_feat, (b * t, d))
        queue = self.construct_queue(seq_feat)
        dist = tf.norm(seq_feat[:, None, :] - queue[None, :, :], axis=-1)
        int_rew = -1.0 * tf.math.top_k(
            -dist, k=tf.reduce_min([self.config.k, tf.shape(queue)[0]])
        ).values.mean(1)
        int_rew = tf.stop_gradient(int_rew)
        int_rew, int_rew_mets = self.intr_rewnorm(int_rew)
        int_rew_mets = {f"intr_{k}": v for k, v in int_rew_mets.items()}
        int_rew = tf.reshape(int_rew, (b, t))
        data["reward"] = data["reward"][:, :t] + self.config.beta * tf.cast(
            tf.stop_gradient(int_rew), data["reward"].dtype
        )
        ###############################################################################

        for name, head in self.heads.items():
            grad_head = name in self.config.grad_heads
            inp = feat if grad_head else tf.stop_gradient(feat)
            if name == "reward":
                inp = inp[:, :t]
            out = head(inp)
            dists = out if isinstance(out, dict) else {name: out}
            for key, dist in dists.items():
                like = tf.cast(dist.log_prob(data[key]), tf.float32)
                likes[key] = like
                losses[key] = -like.mean()
        model_loss = sum(
            self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items()
        )
        outs = dict(
            embed=embed, feat=feat, post=post, prior=prior, likes=likes, kl=kl_value
        )
        metrics = {f"{name}_loss": value for name, value in losses.items()}
        metrics["model_kl"] = kl_value.mean()
        metrics["af_model_kl"] = af_kl_value.mean()
        metrics["prior_ent"] = self.rssm.get_dist(prior).entropy().mean()
        metrics["post_ent"] = self.rssm.get_dist(post).entropy().mean()
        metrics.update(**int_rew_mets)
        last_state = {k: v[:, -1] for k, v in post.items()}
        return model_loss, last_state, outs, metrics

    def imagine(self, policy, start, is_terminal, horizon):
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}
        start["feat"] = self.rssm.get_feat(start)
        start["action"] = tf.zeros_like(policy(start["feat"]).mode())
        seq = {k: [v] for k, v in start.items()}
        for _ in range(horizon):
            action = policy(tf.stop_gradient(seq["feat"][-1])).sample()
            state = self.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
            feat = self.rssm.get_feat(state)
            for key, value in {**state, "action": action, "feat": feat}.items():
                seq[key].append(value)
        seq = {k: tf.stack(v, 0) for k, v in seq.items()}
        if "discount" in self.heads:
            disc = self.heads["discount"](seq["feat"]).mean()
            if is_terminal is not None:
                # Override discount prediction for the first step with the true
                # discount factor from the replay buffer.
                true_first = 1.0 - flatten(is_terminal).astype(disc.dtype)
                true_first *= self.config.discount
                disc = tf.concat([true_first[None], disc[1:]], 0)
        else:
            disc = self.config.discount * tf.ones(seq["feat"].shape[:-1])
        seq["discount"] = disc
        # Shift discount factors because they imply whether the following state
        # will be valid, not whether the current state is valid.
        seq["weight"] = tf.math.cumprod(
            tf.concat([tf.ones_like(disc[:1]), disc[:-1]], 0), 0
        )
        return seq

    @tf.function
    def preprocess(self, obs):
        dtype = prec.global_policy().compute_dtype
        obs = obs.copy()
        for key, value in obs.items():
            if key.startswith("log_"):
                continue
            if value.dtype == tf.int32:
                value = value.astype(dtype)
            if value.dtype == tf.uint8:
                value = value.astype(dtype) / 255.0 - 0.5
            obs[key] = value
        if self.config.clip_rewards in ["identity", "sign", "tanh"]:
            obs["reward"] = {
                "identity": tf.identity,
                "sign": tf.sign,
                "tanh": tf.tanh,
            }[self.config.clip_rewards](obs["reward"])
        else:
            obs["reward"] /= float(self.config.clip_rewards)
        obs["discount"] = 1.0 - obs["is_terminal"].astype(dtype)
        obs["discount"] *= self.config.discount
        return obs

    @tf.function
    def video_pred(self, data, key):
        decoder = self.heads["decoder"]
        truth = data[key][:6] + 0.5
        embed = self.encoder(data)
        af_post, _ = self.af_rssm.observe(embed, data["action"], data["is_first"])
        af_embed = self.af_rssm.get_feat(af_post)
        if self.config.concat_embed:
            af_embed = tf.concat([embed, af_embed], -1)
        states, _ = self.rssm.observe(
            af_embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = decoder(self.rssm.get_feat(states))[key].mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.rssm.imagine(data["action"][:6, 5:], init)
        openl = decoder(self.rssm.get_feat(prior))[key].mode()
        model = tf.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
        error = (model - truth + 1) / 2
        video = tf.concat([truth, model, error], 2)
        B, T, H, W, C = video.shape
        return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))


class ActorCritic(common.Module):
    def __init__(self, config, act_space, tfstep):
        self.config = config
        self.act_space = act_space
        self.tfstep = tfstep
        discrete = hasattr(act_space, "n")
        if self.config.actor.dist == "auto":
            self.config = self.config.update(
                {"actor.dist": "onehot" if discrete else "trunc_normal"}
            )
        if self.config.actor_grad == "auto":
            self.config = self.config.update(
                {"actor_grad": "reinforce" if discrete else "dynamics"}
            )
        self.actor = common.MLP(act_space.shape[0], **self.config.actor)
        self.critic = common.MLP([], **self.config.critic)
        if self.config.slow_target:
            self._target_critic = common.MLP([], **self.config.critic)
            self._updates = tf.Variable(0, tf.int64)
        else:
            self._target_critic = self.critic
        self.actor_opt = common.Optimizer("actor", **self.config.actor_opt)
        self.critic_opt = common.Optimizer("critic", **self.config.critic_opt)
        self.rewnorm = common.StreamNorm(**self.config.reward_norm)

    def train(self, world_model, start, is_terminal, reward_fn):
        metrics = {}
        hor = self.config.imag_horizon
        # The weights are is_terminal flags for the imagination start states.
        # Technically, they should multiply the losses from the second trajectory
        # step onwards, which is the first imagined step. However, we are not
        # training the action that led into the first step anyway, so we can use
        # them to scale the whole sequence.
        with tf.GradientTape() as actor_tape:
            seq = world_model.imagine(self.actor, start, is_terminal, hor)
            reward = reward_fn(seq)
            seq["reward"], mets1 = self.rewnorm(reward)
            mets1 = {f"reward_{k}": v for k, v in mets1.items()}
            target, mets2 = self.target(seq)
            actor_loss, mets3 = self.actor_loss(seq, target)
        with tf.GradientTape() as critic_tape:
            critic_loss, mets4 = self.critic_loss(seq, target)
        metrics.update(self.actor_opt(actor_tape, actor_loss, self.actor))
        metrics.update(self.critic_opt(critic_tape, critic_loss, self.critic))
        metrics.update(**mets1, **mets2, **mets3, **mets4)
        self.update_slow_target()  # Variables exist after first forward pass.
        return metrics

    def actor_loss(self, seq, target):
        # Actions:      0   [a1]  [a2]   a3
        #                  ^  |  ^  |  ^  |
        #                 /   v /   v /   v
        # States:     [z0]->[z1]-> z2 -> z3
        # Targets:     t0   [t1]  [t2]
        # Baselines:  [v0]  [v1]   v2    v3
        # Entropies:        [e1]  [e2]
        # Weights:    [ 1]  [w1]   w2    w3
        # Loss:              l1    l2
        metrics = {}
        # Two states are lost at the end of the trajectory, one for the boostrap
        # value prediction and one because the corresponding action does not lead
        # anywhere anymore. One target is lost at the start of the trajectory
        # because the initial state comes from the replay buffer.
        policy = self.actor(tf.stop_gradient(seq["feat"][:-2]))
        if self.config.actor_grad == "dynamics":
            objective = target[1:]
        elif self.config.actor_grad == "reinforce":
            baseline = self._target_critic(seq["feat"][:-2]).mode()
            advantage = tf.stop_gradient(target[1:] - baseline)
            objective = policy.log_prob(seq["action"][1:-1]) * advantage
        elif self.config.actor_grad == "both":
            baseline = self._target_critic(seq["feat"][:-2]).mode()
            advantage = tf.stop_gradient(target[1:] - baseline)
            objective = policy.log_prob(seq["action"][1:-1]) * advantage
            mix = common.schedule(self.config.actor_grad_mix, self.tfstep)
            objective = mix * target[1:] + (1 - mix) * objective
            metrics["actor_grad_mix"] = mix
        else:
            raise NotImplementedError(self.config.actor_grad)
        ent = policy.entropy()
        ent_scale = common.schedule(self.config.actor_ent, self.tfstep)
        objective += ent_scale * ent
        weight = tf.stop_gradient(seq["weight"])
        actor_loss = -(weight[:-2] * objective).mean()
        metrics["actor_ent"] = ent.mean()
        metrics["actor_ent_scale"] = ent_scale
        return actor_loss, metrics

    def critic_loss(self, seq, target):
        # States:     [z0]  [z1]  [z2]   z3
        # Rewards:    [r0]  [r1]  [r2]   r3
        # Values:     [v0]  [v1]  [v2]   v3
        # Weights:    [ 1]  [w1]  [w2]   w3
        # Targets:    [t0]  [t1]  [t2]
        # Loss:        l0    l1    l2
        dist = self.critic(seq["feat"][:-1])
        target = tf.stop_gradient(target)
        weight = tf.stop_gradient(seq["weight"])
        critic_loss = -(dist.log_prob(target) * weight[:-1]).mean()
        metrics = {"critic": dist.mode().mean()}
        return critic_loss, metrics

    def target(self, seq):
        # States:     [z0]  [z1]  [z2]  [z3]
        # Rewards:    [r0]  [r1]  [r2]   r3
        # Values:     [v0]  [v1]  [v2]  [v3]
        # Discount:   [d0]  [d1]  [d2]   d3
        # Targets:     t0    t1    t2
        reward = tf.cast(seq["reward"], tf.float32)
        disc = tf.cast(seq["discount"], tf.float32)
        value = self._target_critic(seq["feat"]).mode()
        # Skipping last time step because it is used for bootstrapping.
        target = common.lambda_return(
            reward[:-1],
            value[:-1],
            disc[:-1],
            bootstrap=value[-1],
            lambda_=self.config.discount_lambda,
            axis=0,
        )
        metrics = {}
        metrics["critic_slow"] = value.mean()
        metrics["critic_target"] = target.mean()
        return target, metrics

    def update_slow_target(self):
        if self.config.slow_target:
            if self._updates % self.config.slow_target_update == 0:
                mix = (
                    1.0
                    if self._updates == 0
                    else float(self.config.slow_target_fraction)
                )
                for s, d in zip(self.critic.variables, self._target_critic.variables):
                    d.assign(mix * s + (1 - mix) * d)
            self._updates.assign_add(1)
