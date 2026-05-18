import glob
from argparse import ArgumentParser
from typing import Iterable, Tuple

import numpy as np
from tqdm import tqdm
from numpy import ndarray

from .__main__ import pickle_load, write_ranking


def _topk_merge(existing_scores: ndarray, existing_indices: ndarray,
                new_scores: ndarray, new_indices: ndarray):
    concat_scores = np.concatenate([existing_scores, new_scores], axis=1)
    concat_indices = np.concatenate([existing_indices, new_indices], axis=1)

    k = existing_scores.shape[1]
    top_idx = np.argpartition(concat_scores, -k, axis=1)[:, -k:]
    rows = np.arange(concat_scores.shape[0])[:, None]
    selected_scores = np.take_along_axis(concat_scores, top_idx, axis=1)
    selected_indices = np.take_along_axis(concat_indices, top_idx, axis=1)

    order = np.argsort(selected_scores, axis=1)[:, ::-1]
    merged_scores = np.take_along_axis(selected_scores, order, axis=1)
    merged_indices = np.take_along_axis(selected_indices, order, axis=1)
    return merged_scores, merged_indices


def combine_faiss_results(results: Iterable[Tuple[ndarray, ndarray]]):
    combined_scores = None
    combined_indices = None
    for scores, indices in results:
        scores = np.array(scores)
        indices = np.array(indices)
        if combined_scores is None:
            print(f'Initializing merge. Assuming {scores.shape[0]} queries.')
            combined_scores = scores
            combined_indices = indices
            continue
        if scores.shape[0] != combined_scores.shape[0]:
            raise ValueError('Mismatched number of queries across result shards.')
        combined_scores, combined_indices = _topk_merge(
            combined_scores, combined_indices, scores, indices
        )

    return combined_scores, combined_indices


def main():
    parser = ArgumentParser()
    parser.add_argument('--score_dir', required=True)
    parser.add_argument('--query', required=True)
    parser.add_argument('--save_ranking_to', required=True)
    args = parser.parse_args()

    partitions = glob.glob(f'{args.score_dir}/*')

    corpus_scores, corpus_indices = combine_faiss_results(map(pickle_load, tqdm(partitions)))

    _, q_lookup = pickle_load(args.query)
    write_ranking(corpus_indices, corpus_scores, q_lookup, args.save_ranking_to)


if __name__ == '__main__':
    main()


# import glob
# import faiss
# from argparse import ArgumentParser
# from tqdm import tqdm
# from typing import Iterable, Tuple
# from numpy import ndarray
# from .__main__ import pickle_load, write_ranking


# def combine_faiss_results(results: Iterable[Tuple[ndarray, ndarray]]):
#     rh = None
#     for scores, indices in results:
#         # ResultHeap needs the number of queries and the per-shard top-k width.
#         if rh is None:
#             print(f'Initializing Heap. Assuming {scores.shape[0]} queries.')
#             # scores.shape[1] is the retrieval depth returned by each shard.
#             rh = faiss.ResultHeap(scores.shape[0], scores.shape[1])

#         # ResultHeap expects negated scores because it maintains a min-heap
#         # internally; indices carry the corresponding passage ids.
#         rh.add_result(-scores, indices)

#     # finalize merges all shard heaps into final score and index matrices.
#     rh.finalize()
#     # Scores are negated back to the original inner-product values.
#     corpus_scores, corpus_indices = -rh.D, rh.I

#     return corpus_scores, corpus_indices


# def main():
#     parser = ArgumentParser()
#     parser.add_argument('--score_dir', required=True)
#     parser.add_argument('--query', required=True)
#     parser.add_argument('--save_ranking_to', required=True)
#     args = parser.parse_args()

#     # Load score shards, each saved as pickle.dump((scores, indices)).
#     partitions = glob.glob(f'{args.score_dir}/*')

#     # Merge all partial shard rankings into a final top-k ranking.
#     corpus_scores, corpus_indices = combine_faiss_results(map(pickle_load, tqdm(partitions)))

#     # Reuse query ids from encoding and write the merged text ranking.
#     _, q_lookup = pickle_load(args.query)
#     write_ranking(corpus_indices, corpus_scores, q_lookup, args.save_ranking_to)


# if __name__ == '__main__':
#     main()
