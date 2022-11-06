import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from utils import NUM_CHECKPOINTS, set_seed


def main(argv=sys.argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument(
        '--num-instances', type=int, default=0,
        help='Number of instances for pairwise calculations. Use 0 for all.'
    )
    parser.add_argument('--seed', type=int, default=0)  # only relevant along with num_instances>0
    args = parser.parse_args(argv[1:])
    os.makedirs(args.workspace, exist_ok=True)
    eval_dir = os.path.join(args.workspace, 'eval')
    set_seed(args.seed)
    correct_csv_path = os.path.join(eval_dir, 'correct.csv')
    eval_correct = np.loadtxt(correct_csv_path, dtype=bool, delimiter=',')
    root_reprs_dir = os.path.join(eval_dir, 'representations')
    distances_dict = {
        'checkpoint': [],
        'euclidean_mean': [],
        'euclidean_median': [],
        'euclidean_std': [],
        'cosine_mean': [],
        'cosine_median': [],
        'cosine_std': [],
        'count': []
    }
    # Limit to images that were correctly classified initially.
    idxs = np.where(eval_correct)[0]
    if args.num_instances > 0:
        idxs = np.sort(np.random.choice(idxs, replace=False, size=args.num_instances))
    for checkpoint in range(NUM_CHECKPOINTS):
        print('checkpoint:', checkpoint, time.time())
        repr_path = os.path.join(root_reprs_dir, f'{checkpoint}.npy')
        with open(repr_path, 'rb') as f:
            representations = np.load(f)[idxs]
            euclidean_pdists = pdist(representations, metric='euclidean')
            cosine_pdists = pdist(representations, metric='cosine')
            distances_dict['checkpoint'].append(checkpoint)
            distances_dict['euclidean_mean'].append(euclidean_pdists.mean())
            distances_dict['euclidean_median'].append(np.median(euclidean_pdists))
            distances_dict['euclidean_std'].append(euclidean_pdists.std(ddof=1))
            distances_dict['cosine_mean'].append(cosine_pdists.mean())
            distances_dict['cosine_median'].append(np.median(cosine_pdists))
            distances_dict['cosine_std'].append(cosine_pdists.std(ddof=1))
            assert len(euclidean_pdists) == len(cosine_pdists)
            distances_dict['count'].append(len(euclidean_pdists))
    df = pd.DataFrame.from_dict(distances_dict)
    print(df)
    pairwise_path = os.path.join(args.workspace, 'pairwise_distances.csv')
    os.makedirs(os.path.dirname(pairwise_path), exist_ok=True)
    df.to_csv(pairwise_path, index=False)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
