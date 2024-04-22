import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import platform
if platform.system() == 'Windows':
    os.add_dll_directory(os.path.abspath("mujoco210//bin"))
import argparse
import sys
sys.path.append(os.getcwd())

import torch
import numpy as np
import wandb

from uhc.utils.config_utils.handmimic_config import Config
from uhc.agents.agent_handmimic import AgentHandMimic

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--num_threads", type=int, default=16)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--show_noise", action="store_true", default=False)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_log", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--full_eval", action="store_true", default=False)
    args = parser.parse_args()

    cfg = Config(cfg_id=args.cfg, create_dirs=not (args.render or args.epoch > 0))
    cfg.update(args)

    if args.debug:
        # cfg.data_specs["file_path"] = "sample_data/amass_copycat_take5_test.pkl"
        # cfg.num_threads = 1
        cfg.no_log = True

    if not cfg.no_log:
        wandb.init(
            project="copycat",
            resume=not args.resume is None,
            id=args.resume,
            notes=cfg.notes,
        )
        wandb.config.update(vars(cfg), allow_val_change=True)
        wandb.config.update(args, allow_val_change=True)
        wandb.run.name = args.cfg
        wandb.run.save()

    if cfg.render:
        print("Rendering!!")
        from mujoco_py import load_model_from_path, MjSim
        from uhc.khrylib.rl.envs.common.mjviewer import MjViewer

        model = load_model_from_path(cfg.mujoco_model_file)
        sim = MjSim(model)
        viewer = MjViewer(sim)
        cfg.num_threads = 1

    dtype = torch.float64
    cfg.dtype = np.float64
    torch.set_default_dtype(dtype)
    device = (torch.device("cuda", index=args.gpu_index) if torch.cuda.is_available() else torch.device("cpu"))
    # device = torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    print(f"Using: {device}")
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    agent = AgentHandMimic(cfg, dtype, device, training=True, checkpoint_epoch=args.epoch)
    # agent.optimize_policy(0)
    # torch.cuda.empty_cache()

    for i_iter in range(args.epoch, cfg.num_epoch):
        agent.optimize_policy(i_iter)
        """clean up gpu memory"""
        torch.cuda.empty_cache()

    print("training done!")
