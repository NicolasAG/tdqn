import os
from os.path import join as pjoin
import json
import argparse
import jericho
from tdqn import TDQN_Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rom_path', default='zork1.z5')
    parser.add_argument('--output_dir', default='logs')
    parser.add_argument('--spm_path', default='../spm_models/unigram_8k.model')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--env_step_limit', default=100, type=int)
    parser.add_argument('--lr', default=0.0003, type=float)
    parser.add_argument('--gamma', default=.95, type=float)
    parser.add_argument('--rho', default=.5, type=float)
    parser.add_argument('--embedding_size', default=64, type=int)
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--steps', default=1000000, type=int)
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--update_freq_td', default=4, type=int)
    parser.add_argument('--update_freq_tar', default=1000, type=int)
    parser.add_argument('--replay_buffer_size', default=100000, type=int)
    parser.add_argument('--replay_buffer_type', default='priority')
    parser.add_argument('--clip', default=40, type=float)
    parser.add_argument('--max_seq_len', default=300, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    assert jericho.__version__ == '2.1.0', "This code is designed to be run with Jericho version 2.1.0."
    args = parse_args()
    print(args)
    trainer = TDQN_Trainer(args)
    trainer.train()

    with open(pjoin(args.output_dir, trainer.filename + '_trajectories.json'), "r") as f:
        trajectories = json.load(f)
    print(f"generated {len(trajectories)} during training.")
    max_score = max([traj['score'] for traj in trajectories])

    n_max = len([t for t in trajectories if t['score'] == max_score])
    print(f"Only {n_max} of them have a max score of {max_score}.")

    extra = 1000
    print(f"try to generate {extra} more with score >= {max_score}...")
    new_trajectories = trainer.generate(n=extra)
    cnt = 0
    for traj in new_trajectories:
        if traj['score'] >= max_score:
            trajectories.append(traj)
            cnt += 1
    print(f"generated {cnt} more.")
    print(f"min | avg | max score of new trajectories: "
          f"{min([t['score'] for t in new_trajectories])} | "
          f"{sum([t['score'] for t in new_trajectories]) / len(new_trajectories)} | "
          f"{max([t['score'] for t in new_trajectories])}")

    print(f"save new trajectories...")
    with open(pjoin(args.output_dir, trainer.filename + '_trajectories.json'), "w") as f:
        json.dump(trajectories, f)
    print("done.")
