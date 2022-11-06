import argparse
import os
import sys

import numpy as np

from utils import ATTACKS, NUM_CHECKPOINTS


def main(argv=sys.argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str, default='workspace')
    args = parser.parse_args(argv[1:])
    os.makedirs(args.workspace, exist_ok=True)
    eval_dir = os.path.join(args.workspace, 'eval')
    attack_dir = os.path.join(args.workspace, 'attack')
    distances_dir = os.path.join(args.workspace, 'distances')
    os.makedirs(distances_dir, exist_ok=True)
    correct_csv_path = os.path.join(eval_dir, 'correct.csv')
    eval_correct = np.loadtxt(correct_csv_path, dtype=bool, delimiter=',')
    root_reprs_dir = os.path.join(eval_dir, 'representations')
    for attack in ATTACKS:
        outdir = os.path.join(distances_dir, attack)
        os.makedirs(outdir, exist_ok=True)
        root_adv_reprs_dir = os.path.join(attack_dir, attack, 'representations')
        for checkpoint in range(NUM_CHECKPOINTS):
            print(attack, checkpoint)
            repr_path = os.path.join(root_reprs_dir, f'{checkpoint}.npy')
            with open(repr_path, 'rb') as f:
                # Limit to images that were correctly classified initially.
                # This was already done earlier in the pipeline for the adversarial images.
                representations = np.load(f)[eval_correct]
            adv_repr_path = os.path.join(root_adv_reprs_dir, f'{checkpoint}.npy')
            with open(adv_repr_path, 'rb') as f:
                adv_representations = np.load(f)
            euc_distances = np.linalg.norm(representations - adv_representations, axis=1, ord=2)
            cos_distances = (representations * adv_representations).sum(axis=1)
            cos_distances = cos_distances / np.linalg.norm(representations, axis=1, ord=2)
            cos_distances = cos_distances / np.linalg.norm(adv_representations, axis=1, ord=2)
            cos_distances = 1 - cos_distances
            with open(os.path.join(outdir, f'{checkpoint}.npz'), 'wb') as f:
                np.savez(f, euclidean=euc_distances.astype(np.float32), cosine=cos_distances.astype(np.float32))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
