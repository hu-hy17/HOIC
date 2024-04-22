import os

import joblib
import pickle
from collections import defaultdict
import os.path as osp
import wandb

from uhc.khrylib.utils import ZFilter, get_eta_str
from uhc.khrylib.utils.memory import Memory
from uhc.khrylib.utils.torch import *
from uhc.khrylib.rl.agents import AgentPPO
from uhc.khrylib.rl.core.policy_gaussian import PolicyGaussian
from uhc.khrylib.rl.core.critic import Value
from uhc.khrylib.models.mlp import MLP
from uhc.envs.ho_im4 import HandObjMimic4
from uhc.envs.ho_reward import *
from uhc.data_loaders.dataset_singledepth import DatasetSingleDepth
import multiprocessing
from uhc.utils.tools import CustomUnpickler
from loguru import logger


class AgentHandMimic(AgentPPO):

    def __init__(self, cfg, dtype, device, training=True, checkpoint_epoch=0):
        self.cfg = cfg
        self.cc_cfg = cfg
        self.device = device
        self.dtype = dtype
        self.training = training
        self.max_freq = 50
        self.with_obj = self.cfg.data_specs['with_obj']

        self.setup_vars()
        self.setup_data_loader()
        self.setup_env()
        self.setup_policy()
        self.setup_value()
        self.setup_optimizer()
        self.setup_logging()
        self.setup_reward()
        self.seed(cfg.seed)
        self.print_config()
        if checkpoint_epoch > 0:
            self.load_checkpoint(checkpoint_epoch)
            self.epoch = checkpoint_epoch

        super().__init__(
            env=self.env,
            dtype=dtype,
            device=device,
            running_state=self.running_state,
            custom_reward=self.expert_reward,
            mean_action=cfg.render and not cfg.show_noise,
            render=cfg.render,
            num_threads=cfg.num_threads,
            data_loader=self.data_loader,
            policy_net=self.policy_net,
            value_net=self.value_net,
            optimizer_policy=self.optimizer_policy,
            optimizer_value=self.optimizer_value,
            opt_num_epochs=cfg.num_optim_epoch,
            gamma=cfg.gamma,
            tau=cfg.tau,
            clip_epsilon=cfg.clip_epsilon,
            policy_grad_clip=[(self.policy_net.parameters(), 40)],
            end_reward=cfg.end_reward,
            use_mini_batch=False,
            mini_batch_size=0,
        )

        # fork plan when thread_num == 64
        self.fork_plan = self.get_fork_plan()

    def get_fork_plan(self):
        thread_num = self.num_threads
        child_thread_arr = [[] for _ in range(thread_num)]
        cur_thread = [0]
        while len(cur_thread) < thread_num:
            c_num = len(cur_thread)
            for i in range(c_num):
                t_idx = cur_thread[i]
                new_t_idx = len(cur_thread)
                child_thread_arr[t_idx].append(new_t_idx)
                cur_thread.append(new_t_idx)
                if len(cur_thread) >= thread_num:
                    break
        return child_thread_arr

    def setup_vars(self):
        self.epoch = 0
        self.running_state = None
        self.fit_single_key = ""
        self.precision_mode = self.cc_cfg.get("precision_mode", False)

        pass

    def print_config(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        logger.info("==========================Agent HandMimic===========================")
        logger.info(f"Feature Version: {cfg.obs_v}")
        logger.info(f"Meta Pd: {cfg.meta_pd}")
        logger.info(f"Meta Pd Joint: {cfg.meta_pd_joint}")
        logger.info(f"Actor_type: {cfg.actor_type}")
        logger.info(f"Precision mode: {self.precision_mode}")
        logger.info(f"State_dim: {self.state_dim}")
        logger.info("============================================================")

    def setup_data_loader(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        self.data_loader = DatasetSingleDepth(cfg.mujoco_model_file, cfg.data_specs)

    def setup_env(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        expert_seq = self.data_loader.load_seq(0)
        self.env = HandObjMimic4(
            cfg,
            expert_seq=expert_seq,
            data_specs=cfg.data_specs,
            model_xml=self.data_loader.model_path,
            mode="train",
        )

    def setup_policy(self):
        cfg, device, dtype, env = self.cfg, self.device, self.dtype, self.env
        actuators = env.model.actuator_names
        self.state_dim = state_dim = env.observation_space.shape[0]
        self.action_dim = action_dim = env.action_space.shape[0]
        """define actor and critic"""
        self.policy_net = PolicyGaussian(cfg, action_dim=action_dim, state_dim=state_dim)
        self.running_state = ZFilter((state_dim,), clip=5)
        to_device(device, self.policy_net)

    def setup_value(self):
        cfg, device, dtype, env = self.cfg, self.device, self.dtype, self.env
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.value_net = Value(MLP(state_dim, cfg.value_hsize, cfg.value_htype))
        to_device(device, self.value_net)

    def setup_optimizer(self):
        cfg, device, dtype, env = self.cfg, self.device, self.dtype, self.env
        if cfg.policy_optimizer == "Adam":
            self.optimizer_policy = torch.optim.Adam(
                self.policy_net.parameters(),
                lr=cfg.policy_lr,
                weight_decay=cfg.policy_weightdecay,
            )
        else:
            self.optimizer_policy = torch.optim.SGD(
                self.policy_net.parameters(),
                lr=cfg.policy_lr,
                momentum=cfg.policy_momentum,
                weight_decay=cfg.policy_weightdecay,
            )
        if cfg.value_optimizer == "Adam":
            self.optimizer_value = torch.optim.Adam(
                self.value_net.parameters(),
                lr=cfg.value_lr,
                weight_decay=cfg.value_weightdecay,
            )
        else:
            self.optimizer_value = torch.optim.SGD(
                self.value_net.parameters(),
                lr=cfg.value_lr,
                momentum=cfg.value_momentum,
                weight_decay=cfg.value_weightdecay,
            )

    def setup_reward(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        self.expert_reward = reward_list[cfg.reward_type - 1]

    def save_checkpoint(self, epoch):
        cfg = self.cfg
        # self.tb_logger.flush()
        with to_cpu(self.policy_net, self.value_net):
            cp_path = "%s/iter_%04d.p" % (cfg.model_dir, epoch + 1)
            model_cp = {
                "policy_dict": self.policy_net.state_dict(),
                "value_dict": self.value_net.state_dict(),
                "running_state": self.running_state,
            }
            pickle.dump(model_cp, open(cp_path, "wb"))
            joblib.dump(self.freq_dict, osp.join(cfg.result_dir, "freq_dict.pt"))

    def save_singles(self, epoch, key):
        cfg = self.cfg
        # self.tb_logger.flush()
        with to_cpu(self.policy_net, self.value_net):
            cp_path = f"{cfg.model_dir}_singles/{key}.p"

            model_cp = {
                "policy_dict": self.policy_net.state_dict(),
                "value_dict": self.value_net.state_dict(),
                "running_state": self.running_state,
            }
            pickle.dump(model_cp, open(cp_path, "wb"))

    def save_curr(self):
        cfg = self.cfg
        # self.tb_logger.flush()
        with to_cpu(self.policy_net, self.value_net):
            cp_path_best = f"{cfg.model_dir}/iter_best.p"

            model_cp = {
                "policy_dict": self.policy_net.state_dict(),
                "value_dict": self.value_net.state_dict(),
                "running_state": self.running_state,
            }
            pickle.dump(model_cp, open(cp_path_best, "wb"))

    def load_curr(self):
        cfg = self.cfg
        cp_path_best = f"{cfg.model_dir}/iter_best.p"
        logger.info("loading model from checkpoint: %s" % cp_path_best)
        model_cp = CustomUnpickler(open(cp_path_best, "rb")).load()
        self.policy_net.load_state_dict(model_cp["policy_dict"])
        self.value_net.load_state_dict(model_cp["value_dict"])
        self.running_state = model_cp["running_state"]

    def load_singles(self, epoch, key):
        cfg = self.cfg
        # self.tb_logger.flush()
        if epoch > 0:
            cp_path = f"{cfg.model_dir}/iter_{(epoch + 1):04d}_{key}.p"
            logger.info("loading model from checkpoint: %s" % cp_path)
            model_cp = CustomUnpickler(open(cp_path, "rb")).load()
            self.policy_net.load_state_dict(model_cp["policy_dict"])
            self.value_net.load_state_dict(model_cp["value_dict"])
            self.running_state = model_cp["running_state"]

    def load_checkpoint(self, iter):
        cfg, device, dtype, env = self.cfg, self.device, self.dtype, self.env
        cfg = self.cfg
        if iter > 0:
            cp_path = "%s/iter_%04d.p" % (cfg.model_dir, iter)
            logger.info("loading model from checkpoint: %s" % cp_path)
            model_cp = CustomUnpickler(open(cp_path, "rb")).load()
            self.policy_net.load_state_dict(model_cp["policy_dict"])
            self.value_net.load_state_dict(model_cp["value_dict"])
            self.running_state = model_cp["running_state"]

        to_device(device, self.policy_net, self.value_net)

    def setup_logging(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        freq_path = osp.join(cfg.result_dir, "freq_dict.pt")
        try:
            self.freq_dict = (
                {k: [] for k in self.data_loader.data_keys} if not osp.exists(freq_path) else joblib.load(freq_path))
            for k in self.data_loader.data_keys:
                if not k in self.freq_dict:
                    raise Exception("freq_dict is not initialized")

            for k in self.freq_dict:
                if not k in self.data_loader.data_keys:
                    raise Exception("freq_dict is not initialized")
        except:
            print("error parsing freq_dict, using empty one")
            self.freq_dict = {k: [] for k in self.data_loader.data_keys}

    def per_epoch_update(self, epoch):
        cfg, device, dtype, env = self.cfg, self.device, self.dtype, self.env
        cfg.update_adaptive_params(epoch)
        self.set_noise_rate(cfg.adp_noise_rate)
        set_optimizer_lr(self.optimizer_policy, cfg.adp_policy_lr)
        if cfg.rfc_decay:
            if self.epoch < cfg.get("rfc_decay_max", 10000):
                self.env.rfc_rate = lambda_rule(self.epoch, cfg.get("rfc_decay_max", 10000), cfg.num_epoch_fix)
            else:
                self.env.rfc_rate = 0.0

        # epoch
        # adative_iter = cfg.data_specs.get("adaptive_iter", -1)
        # if epoch != 0 and adative_iter != -1 and epoch % adative_iter == 0 :
        # agent.data_loader.hard_negative_mining(agent.value_net, agent.env, device, dtype, running_state = running_state, sampling_temp = cfg.sampling_temp)

        if cfg.fix_std:
            self.policy_net.action_log_std.fill_(cfg.adp_log_std)
        return

    def log_train(self, info):
        """logging"""
        cfg, device, dtype = self.cfg, self.device, self.dtype
        log = info["log"]

        c_info = log.avg_c_info
        log_str = f"Ep: {self.epoch}\t {cfg.id} \tT_s {info['T_sample']:.2f}\t \
                    T_u {info['T_update']:.2f}\tETA {get_eta_str(self.epoch, cfg.num_epoch, info['T_total'])} \
                \texpert_R_avg {log.avg_c_reward:.4f} {np.array2string(c_info, formatter={'all': lambda x: '%.4f' % x}, separator=',')}\
                 \texpert_R_range ({log.min_c_reward:.4f}, {log.max_c_reward:.4f})\teps_len {log.avg_episode_len:.2f}"

        logger.info(log_str)

        if not cfg.no_log:
            wandb.log(
                data={
                    "rewards": log.avg_c_info,
                    "eps_len": log.avg_episode_len,
                    "avg_rwd": log.avg_c_reward,
                    # "rfc_rate": self.env.rfc_rate,
                },
                step=self.epoch,
            )

            if "log_eval" in info:
                wandb.log(data=info["log_eval"], step=self.epoch)

    def optimize_policy(self, epoch, save_model=True):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        self.epoch = epoch
        t0 = time.time()
        self.per_epoch_update(epoch)
        batch, log = self.sample(cfg.min_batch_size)

        if cfg.end_reward:
            self.env.end_reward = log.avg_c_reward * cfg.gamma / (1 - cfg.gamma)
        """update networks"""
        t1 = time.time()
        self.update_params(batch)
        t2 = time.time()
        info = {
            "log": log,
            "T_sample": t1 - t0,
            "T_update": t2 - t1,
            "T_total": t2 - t0,
        }

        if save_model and (self.epoch + 1) % cfg.save_n_epochs == 0:
            self.save_checkpoint(epoch)
            log_eval = self.eval_policy(epoch)
            info["log_eval"] = log_eval

        self.log_train(info)
        # joblib.dump(self.freq_dict, osp.join(cfg.result_dir, "freq_dict.pt"))

    def eval_policy(self, epoch=0, dump=False):
        self.env.set_mode("test")
        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                res = defaultdict(list)
                seq_idx = self.data_loader.seq_num - 1
                # seq_idx = 14
                test_expert = self.data_loader.load_seq(seq_idx, start_idx=0, full_seq=True)
                # test_expert = self.data_loader.load_seq(seq_idx, start_idx=550, full_seq=False, end_idx=650)
                self.env.set_expert(test_expert)
                state = self.env.reset()
                if self.running_state is not None:
                    state = self.running_state(state)

                for t in range(10000):
                    res["gt"].append(self.env.get_expert_attr("hand_dof_seq", t % self.env.expert_len).copy())
                    res["pred"].append(self.env.data.qpos[:self.env.hand_qpos_dim].copy())
                    res["gt_jpos"].append(
                        self.env.get_expert_attr("body_pos_seq", t % self.env.expert_len).copy().reshape(-1, 3))
                    res["pred_jpos"].append(self.env.get_wbody_pos().copy().reshape(-1, 3))
                    state_var = tensor(state).unsqueeze(0)
                    trans_out = self.trans_policy(state_var)

                    action = (self.policy_net.select_action(trans_out, mean_action=True)[0].cpu().numpy())
                    next_state, env_reward, done, info = self.env.step(action)

                    c_reward, c_info = self.custom_reward(self.env, state, action, info)
                    res["reward"].append(c_reward)
                    res["pose_reward"].append(c_info[0])
                    res["wpose_reward"].append(c_info[1])
                    res["jpos_reward"].append(c_info[2])
                    res["vel_reward"].append(c_info[3])
                    if self.with_obj:
                        res["obj_pos_reward"].append(c_info[4])
                        res["obj_rot_reward"].append(c_info[5])
                        res["obj_vel_reward"].append(c_info[6])
                        res["obj_rfc_reward"].append(c_info[7])
                        res["contact_reward"].append(c_info[8])

                    if self.running_state is not None:
                        next_state = self.running_state(next_state, update=False)

                    if done:
                        res = {k: np.vstack(v) for k, v in res.items()}

                        # compute metrics
                        metrics = {}
                        metrics["pose_err"] = np.linalg.norm(res["gt"][6:] - res["pred"][6:], axis=-1).mean()
                        metrics["mpjpe"] = np.linalg.norm(res["gt_jpos"] - res["pred_jpos"], axis=-1).mean()
                        metrics["avg_reward"] = np.array(res["reward"]).mean()
                        metrics["total_reward"] = np.array(res["reward"]).sum()
                        metrics["pose_reward"] = np.array(res["pose_reward"]).mean()
                        metrics["wpose_reward"] = np.array(res["wpose_reward"]).mean()
                        metrics["jpos_reward"] = np.array(res["jpos_reward"]).mean()
                        metrics["vel_reward"] = np.array(res["vel_reward"]).mean()
                        metrics["percent"] = info["percent"]
                        if self.with_obj:
                            metrics["obj_pos_reward"] = np.array(res["obj_pos_reward"]).mean()
                            metrics["obj_rot_reward"] = np.array(res["obj_rot_reward"]).mean()
                            metrics["obj_vel_reward"] = np.array(res["obj_vel_reward"]).mean()
                            metrics["obj_rfc_reward"] = np.array(res["obj_rfc_reward"]).mean()

                        return metrics

                    state = next_state

    def seed(self, seed):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.env.seed(seed)

    def eval_seqs(self, take_keys, data_loader, queue, id=0):
        res = {}
        for take_key in take_keys:
            res[take_key] = self.eval_seq(take_key, data_loader)

        if queue == None:
            return res
        else:
            queue.put(res)

    def sample_worker(self, pid, queue, min_batch_size):
        for c in self.fork_plan[pid]:
            args = (c, queue, min_batch_size)

            worker = multiprocessing.Process(target=self.sample_worker, args=args)
            worker.start()

        self.sample_process(pid, queue, min_batch_size)

    def sample_process(self, pid, queue, min_batch_size):
        self.seed_worker(pid)
        if hasattr(self.env, "np_random"):
            self.env.np_random.random(pid)
        memory = Memory()
        sample_logger = self.logger_cls()
        freq_dict = defaultdict(list)
        while sample_logger.num_steps < min_batch_size:
            # sample random seq
            if self.data_loader.seq_num == 1:
                seq_idx = 0
                start_idx = 0
                end_idx = -1
            else:
                seq_idx = np.random.randint(0, self.data_loader.seq_num - 1)
                # seq_idx = 14
                # start_idx = 0
                # end_idx = 650
                start_idx = np.random.randint(0, self.data_loader.get_len(seq_idx) - 200)
                # end_idx = np.random.randint(200, self.data_loader.get_len(0) - 1)
            sample_expert = self.data_loader.load_seq(seq_idx, start_idx=start_idx, full_seq=True)
            # sample_expert = self.data_loader.load_seq(seq_idx, start_idx=start_idx, full_seq=False, end_idx=end_idx)
            self.env.set_expert(sample_expert)

            state = self.env.reset()
            if self.running_state is not None:
                state = self.running_state(state)
            sample_logger.start_episode(self.env)
            self.pre_episode()

            for t in range(10000):
                state = state.astype(self.cfg.dtype)
                state_var = tensor(state).unsqueeze(0)
                trans_out = self.trans_policy(state_var)
                mean_action = self.mean_action or self.env.np_random.binomial(1, 1 - self.noise_rate)
                action = self.policy_net.select_action(trans_out, mean_action)[0].numpy()
                action = (int(action) if self.policy_net.type == "discrete" else action.astype(np.float64))
                next_state, env_reward, done, info = self.env.step(action)
                if self.running_state is not None:
                    next_state = self.running_state(next_state)
                # use custom or env reward
                if self.custom_reward is not None:
                    c_reward, c_info = self.custom_reward(self.env, state, action, info)
                    reward = c_reward
                else:
                    c_reward, c_info = 0.0, np.array([0.0])
                    reward = env_reward

                # add end reward
                if self.end_reward and info.get("end", False):
                    reward += self.env.end_reward
                # logging
                sample_logger.step(self.env, env_reward, c_reward, c_info, info)

                mask = 0 if done else 1
                exp = 1 - mean_action
                self.push_memory(memory, state, action, mask, next_state, reward, exp)

                if done:
                    freq_dict[self.data_loader.curr_key].append([info["percent"], self.data_loader.fr_start])
                    break
                state = next_state

            sample_logger.end_episode(self.env)
        sample_logger.end_sampling()

        # print(pid)

        if queue is not None:
            queue.put([pid, memory, sample_logger, freq_dict])
        else:
            return memory, sample_logger, freq_dict

    def sample(self, min_batch_size):
        self.env.set_mode("train")
        t_start = time.time()
        self.pre_sample()
        to_test(*self.sample_modules)
        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
                queue = multiprocessing.Queue()
                memories = [None] * self.num_threads
                sample_loggers = [None] * self.num_threads
                for c in self.fork_plan[0]:
                    worker_args = (c, queue, thread_batch_size)
                    worker = multiprocessing.Process(target=self.sample_worker, args=worker_args)
                    worker.start()
                memories[0], sample_loggers[0], freq_dict = self.sample_process(0, None, thread_batch_size)

                self.freq_dict = {k: v + freq_dict[k] for k, v in self.freq_dict.items()}

                for i in range(self.num_threads - 1):
                    pid, worker_memory, worker_logger, freq_dict = queue.get()
                    memories[pid] = worker_memory
                    sample_loggers[pid] = worker_logger

                    self.freq_dict = {k: v + freq_dict[k] for k, v in self.freq_dict.items()}

                # print(np.sum([len(v) for k, v in self.freq_dict.items()]), np.mean(np.concatenate([self.freq_dict[k] for k in self.freq_dict.keys()])))
                traj_batch = self.traj_cls(memories)
                sample_logger = self.logger_cls.merge(sample_loggers)

        self.freq_dict = {k: v if len(v) < self.max_freq else v[-self.max_freq:] for k, v in self.freq_dict.items()}
        sample_logger.sample_time = time.time() - t_start
        return traj_batch, sample_logger
