import numpy as np
import faiss

import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BaseFaissIPRetriever:
    def __init__(
        self,
        reps_dim: int,
        use_gpu: bool = False,
        gpu_shard: bool = False,
        gpu_fp16: bool = False,
        faiss_omp_num_threads: int = 64,
    ):
        # Limit FAISS OpenMP parallelism so retrieval runs are reproducible and
        # do not oversubscribe CPU cores on shared machines.
        faiss.omp_set_num_threads(faiss_omp_num_threads)

        # IndexFlatIP performs exact inner-product search. This matches the
        # dense retriever scoring function, and also works as cosine search when
        # vectors are normalized before indexing.
        index = faiss.IndexFlatIP(reps_dim)

        # Optionally move the index to all visible GPUs. The helper preserves the
        # CPU index when GPU search is disabled.
        self.index = self._maybe_move_to_gpu(index, use_gpu, gpu_shard, gpu_fp16)

    @staticmethod
    def _maybe_move_to_gpu(index: faiss.Index, use_gpu: bool, gpu_shard: bool, gpu_fp16: bool):
        if not use_gpu:
            return index

        co = faiss.GpuMultipleClonerOptions()
        co.shard = gpu_shard
        co.useFloat16 = gpu_fp16
        mode = 'sharded' if gpu_shard else 'replicated'
        logger.info(f'Using FAISS GPU with {mode} index (fp16={gpu_fp16})')
        return faiss.index_cpu_to_all_gpus(index, co=co)

    def add(self, p_reps: np.ndarray):
        # Add passage vectors to the index. Multiple calls append shards in order,
        # which is why the caller must keep a matching id lookup list.
        self.index.add(p_reps)

    def search(self, q_reps: np.ndarray, k: int):
        # Return top-k inner-product scores and zero-based index offsets for each
        # query vector.
        return self.index.search(q_reps, k)

    def batch_search(self, q_reps: np.ndarray, k: int, batch_size: int, quiet: bool=False):
        num_query = q_reps.shape[0]
        all_scores = []
        all_indices = []
        # Split large query sets into smaller FAISS calls to reduce peak memory
        # and make GPU searches less likely to fail.
        for start_idx in tqdm(range(0, num_query, batch_size), disable=quiet):
            nn_scores, nn_indices = self.search(q_reps[start_idx: start_idx + batch_size], k)
            all_scores.append(nn_scores)
            all_indices.append(nn_indices)

        # Reassemble batch results into the original query order.
        all_scores = np.concatenate(all_scores, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)

        return all_scores, all_indices


class FaissRetriever(BaseFaissIPRetriever):

    def __init__(
        self,
        init_reps: np.ndarray,
        factory_str: str,
        use_gpu: bool = False,
        gpu_shard: bool = False,
        gpu_fp16: bool = False,
        faiss_omp_num_threads: int = 64,
    ):
        # Build an index from a FAISS factory string. Some factories, such as IVF
        # or PQ variants, require a training pass before vectors can be added.
        faiss.omp_set_num_threads(faiss_omp_num_threads)
        index = faiss.index_factory(init_reps.shape[1], factory_str)
        # Train indexes that are not ready immediately after construction.
        if not index.is_trained:
            index.train(init_reps)
        configured_index = self._maybe_move_to_gpu(index, use_gpu, gpu_shard, gpu_fp16)
        self.index = configured_index
        # Surface FAISS logs while building and querying trained index types.
        self.index.verbose = True
