import pandas as pd
import numpy as np
import logging
from typing import List
from tqdm import tqdm
from clusters import HierarchicalCluster
from collections import defaultdict
import bisect
import os
import argparse


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def find_closest_idx(x: float, idxs: List[float]) -> int:
    """
    Goal: Identify which slide the tile is originating from
    The full dataset of tile embeddings is constructed as follows:
    |    slide 1    |    slide 2    |    slide 3    |    slide 4    | ....
    |0 1 ....  12412|12413 ... 17009|17010 ... 23734|23735 ... 34986| ....
    This function identifies which slides start_index is closest to the global_tile_idx we are looking for,
    and therefore which slide it is coming from.
    """
    pos = bisect.bisect_left(idxs, x)
    array_idx = (
        (pos - 1 if pos > 0 and abs(idxs[pos - 1] - x) <= abs(idxs[pos] - x) else pos)
        if pos < len(idxs)
        else len(idxs) - 1
    )
    closest_idx = idxs[array_idx]
    if closest_idx > x:
        return idxs[array_idx - 1]
    else:
        return closest_idx

class ClusterAssociator:
    def __init__(
        self,
        clustering_dir: str,
        slide_tile_mapping_csv: str,
        id_column: str = "slide_id",
    ):
        self._clustering_dir = clustering_dir
        self._output_file = f"{clustering_dir}/all_cluster_associations2.csv"
        self._id_column = id_column
        self._slide_tile_mapping_df = pd.read_csv(slide_tile_mapping_csv)
        self._n_samples = self._slide_tile_mapping_df.iloc[-1]["end_index"]

    def associate_clusters(self):
        if os.path.exists(self._output_file):
            df = pd.read_csv(self._output_file)
            if "tissue" not in df.columns:
                resume_idx = len(df)
                if resume_idx < self._n_samples:
                    cl = HierarchicalCluster.from_file(
                        cluster_path=self._clustering_dir,
                        cluster_fname="sorted_clusters.npy",
                    )
                    cl.build_sample_id_cluster_lookup()
                    cluster_origin = {}
                    start_idxs_wsi = df["start_index"].tolist()
                    tile_idxs = df["tile_idx"].tolist()
                    for level in range(cl.n_levels):
                        cluster_origin[level + 1] = df[f"level_{level+1}"].tolist()
                    logging.info(f"Continue from resume_idx {resume_idx}")
            else:
                logging.info("CSV with metadata has already fully been generated!")
                exit()
        else:
            resume_idx = 0
            start_idxs_wsi = []
            tile_idxs = []
            cluster_origin = defaultdict(list)
            logging.info(f"Start from scratch")
            cl = HierarchicalCluster.from_file(
                cluster_path=self._clustering_dir,
                cluster_fname="sorted_clusters.npy",
            )
            cl.build_sample_id_cluster_lookup()
            
        if resume_idx < self._n_samples:
            try:
                idxs = np.arange(self._n_samples, dtype="int")
                start_idxs = self._slide_tile_mapping_df["start_index"]
                for global_tile_idx in tqdm(idxs[resume_idx:]):
                    for level in range(cl.n_levels):
                        cluster_origin[level + 1].append(
                            cl.sample_id_cluster_mapping[global_tile_idx, level]
                        )
                    start_index = find_closest_idx(global_tile_idx, start_idxs)
                    local_tile_index = global_tile_idx - start_index
                    tile_idxs.append(local_tile_index)
                    start_idxs_wsi.append(start_index)

                df_dict = {
                    "tile_idx": tile_idxs,
                    "start_index": start_idxs_wsi,
                }
                for level, cluster_idxs in cluster_origin.items():
                    df_dict[f"level_{level}"] = cluster_idxs
                df = pd.DataFrame(df_dict)
                df.to_csv(self._output_file, index=False)
            except:
                if len(tile_idxs) > 0:
                    logging.info(
                        f"Code got interrupted, save intermediate result to {self._output_file}"
                    )
                    df_dict = {
                        "tile_idx": tile_idxs,
                        "start_index": start_idxs_wsi,
                    }
                    for level, cluster_idxs in cluster_origin.items():
                        df_dict[f"level_{level}"] = cluster_idxs[: len(tile_idxs)]
                    df = pd.DataFrame(df_dict)
                    df.to_csv(self._output_file, index=False)
                raise

        df = df.merge(
            self._slide_tile_mapping_df[[self._id_column, "start_index"]],
            on="start_index",
            how="left",
        )
        df = df.drop("start_index", axis=1)
        df.to_csv(self._output_file, index=False)


if __name__ == "__main__":
    ####### Configs ############################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clustering_dir",
        type=str,
        help="Path to clustering result dir, e.g. ../ssl-data-curation/outputs/clustering-t1",
    )
    parser.add_argument(
        "--slide_tile_mapping_csv",
        type=str,
        default="./slide_tile_mapping_csv.csv",
        help="Path to csv mapping slide id to the slides start index in the large numpy embeddings file",
    )
    ################################################################################################

    args = parser.parse_args()
    clustering_dir = args.clustering_dir
    slide_tile_mapping_csv = args.slide_tile_mapping_csv

    cluster_associator = ClusterAssociator(clustering_dir, slide_tile_mapping_csv)
    cluster_associator.associate_clusters()