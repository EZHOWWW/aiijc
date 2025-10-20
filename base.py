# %% [markdown]
# # AIIJC Multimodal Retrieval for CAD Objects case solution
# ## Team: IDR

# %% [markdown]
# ### Install libs

# %%
# !pip

# %% [markdown]
# ### Import libs


# %%
import io
import json
import logging
import multiprocessing
import os
import random
import shutil
import tempfile
import urllib.request
import zipfile
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from glob import glob
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import py7zr  # Added for ABC dataset extraction
import requests
import scipy.io
import thingi10k
import torch
import trimesh

# Mesh processing libraries
from datasets import load_dataset
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# %% [markdown]
# ### Global variables and preferences

# %%
DATA_ROOT = Path("/mnt/new_disk/aiijc_data")


IMG_SIZE = (512, 512)
NUM_SAMPLE_POINTS = 2048
BATCH_SIZE = 32


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# %% [markdown]
# ## Download and prepare data

# %%
# Setup clear logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------------
# Worker functions (must be at the module's top level for process pools to work)
# ---------------------------------------------------------------------------------


def _download_worker(url, filename, chunk_size=8192):
    """Securely downloads a file, supporting HTTP/S and FTP."""
    try:
        if url.startswith("ftp://"):
            urllib.request.urlretrieve(url, filename)
        else:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))
                with (
                    open(filename, "wb") as f,
                    tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        leave=False,
                        desc=os.path.basename(filename),
                    ) as pbar,
                ):
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        pbar.update(len(chunk))
        return filename, None
    except Exception as e:
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except OSError:
                pass
        return filename, str(e)


def _convert_to_stl_worker(src_path, dst_path, target_faces=2048):
    """Loads any mesh file, simplifies it if needed, and saves as STL."""
    try:
        mesh_obj = trimesh.load_mesh(src_path)

        if isinstance(mesh_obj, trimesh.Scene):
            if len(mesh_obj.geometry) > 0:
                mesh_obj = trimesh.util.concatenate(
                    tuple(
                        trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                        for g in mesh_obj.geometry.values()
                    )
                )
            else:
                return src_path, "Scene is empty"

        if (
            isinstance(mesh_obj, trimesh.Trimesh)
            and len(mesh_obj.faces) > target_faces
        ):
            mesh_obj = mesh_obj.simplify_quadric_decimation(
                face_count=target_faces
            )

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        mesh_obj.export(dst_path)
        return src_path, None
    except Exception as e:
        return src_path, str(e)


def _thingi10k_worker(npz_path, dst_path, target_faces=2048):
    """Worker for Thingi10k: loads .npz, converts, simplifies, and saves."""
    try:
        if os.path.exists(dst_path):
            return npz_path, None
        with np.load(npz_path) as data:
            vertices = np.asarray(data["vertices"], dtype=np.float64)
            facets = np.asarray(data["facets"], dtype=np.int32)

        if vertices.shape[0] < 3 or facets.shape[0] == 0:
            return npz_path, "Insufficient geometry"

        mesh_obj = trimesh.Trimesh(vertices=vertices, faces=facets)

        if len(mesh_obj.faces) > target_faces:
            mesh_obj = mesh_obj.simplify_quadric_decimation(
                face_count=target_faces
            )

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        mesh_obj.export(dst_path)
        return npz_path, None
    except Exception as e:
        return npz_path, str(e)


def _process_shapenet_model_worker(
    zip_filepath, category_id, model_id, models_dir, images_dir, target_faces
):
    """Worker to process a single model from a ShapeNet zip file."""
    temp_obj_path = ""
    try:
        with zipfile.ZipFile(zip_filepath, "r") as z:
            all_files = z.namelist()

            model_prefix = f"{category_id}/{model_id}/models/"
            model_norm = f"{model_prefix}model_normalized.obj"
            model_path_in_zip = (
                model_norm
                if model_norm in all_files
                else next(
                    (
                        f
                        for f in all_files
                        if f.startswith(model_prefix) and f.endswith(".obj")
                    ),
                    None,
                )
            )
            if not model_path_in_zip:
                return model_id, "No .obj file found"

            pid = os.getpid()
            temp_obj_path = os.path.join(
                tempfile.gettempdir(), f"shapenet_{pid}_{model_id}.obj"
            )
            with (
                z.open(model_path_in_zip) as source,
                open(temp_obj_path, "wb") as target,
            ):
                shutil.copyfileobj(source, target)

            dst_stl_path = os.path.join(models_dir, f"{model_id}.stl")
            _, error = _convert_to_stl_worker(
                temp_obj_path, dst_stl_path, target_faces
            )
            if error:
                return model_id, error

            screenshot_prefix = f"{category_id}/{model_id}/screenshots/"
            screenshot_files = [
                f
                for f in all_files
                if f.startswith(screenshot_prefix)
                and (f.endswith((".png", ".jpg")))
            ]

            for i, screenshot_path in enumerate(screenshot_files):
                ext = ".png" if screenshot_path.endswith(".png") else ".jpg"
                img_dest_path = os.path.join(images_dir, f"{model_id}_{i}{ext}")
                with (
                    z.open(screenshot_path) as source,
                    open(img_dest_path, "wb") as target,
                ):
                    shutil.copyfileobj(source, target)

        return model_id, None
    except Exception as e:
        return model_id, str(e)
    finally:
        if os.path.exists(temp_obj_path):
            os.remove(temp_obj_path)


def _extract_7z_worker(archive_path, extract_to):
    """Extracts a .7z archive to a specified directory."""
    try:
        with py7zr.SevenZipFile(archive_path, mode="r") as z:
            z.extractall(path=extract_to)
        return archive_path, None
    except Exception as e:
        return archive_path, str(e)


# ---------------------------------------------------------------------------------
# DatasetsManager Class
# ---------------------------------------------------------------------------------
class DatasetsManager:
    """Manages downloading, processing, and structuring of 3D datasets with parallel execution."""

    def __init__(
        self,
        root_dir="data",
        max_workers=None,
        target_faces=2048,
        temp_dir="/tmp/",
    ):
        self.root_dir = root_dir
        self.target_faces = target_faces
        os.makedirs(self.root_dir, exist_ok=True)

        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)

        cpu_count = os.cpu_count() or 4
        self.max_workers = max_workers or cpu_count
        self.max_io_workers = self.max_workers * 2

        logging.info(f"Initialized DatasetsManager in '{self.root_dir}'")
        logging.info(
            f"Using up to {self.max_workers} CPU workers and {self.max_io_workers} IO workers."
        )
        logging.info(
            f"All models will be simplified to a maximum of {self.target_faces} faces."
        )

        self.config = {
            "thingi10k": {
                "path": os.path.join(root_dir, "thingi10k"),
                "description": "~10,000 diverse CAD models from Thingiverse. Contains only 3D geometries.",
            },
            "modelnet": {
                "path": os.path.join(root_dir, "modelnet40"),
                "url": "http://modelnet.cs.princeton.edu/ModelNet40.zip",
                "description": "A classic benchmark with 40 categories of clean CAD models. Used for 3D shape classification.",
            },
            "abc": {
                "path": os.path.join(root_dir, "abc_dataset"),
                "manifest_url": "https://deep-geometry.github.io/abc-dataset/data/stl2_v00.txt",
                "description": "A large-scale dataset of over 1 million CAD models with geometric features. Used for pretraining the 3D encoder.",
            },
            "objectnet": {
                "path": os.path.join(root_dir, "objectnet3d"),
                "urls": {
                    "annotations": "ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_annotations.zip",
                    "cads": "ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_cads.zip",
                    "images": "ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_images.zip",
                },
                "description": "Large-scale dataset of CAD models aligned with real-world photographs. Good for pretraining the image encoder.",
            },
            "shapenet": {
                "path": os.path.join(root_dir, "shapenet"),
                "repo_id": "ShapeNet/ShapeNetCore",
                "description": "A large, richly-annotated dataset of 3D models with categories. A primary source for 3D model data.",
            },
            "shapenet_captions": {
                "path": os.path.join(root_dir, "shapenet"),
                "repo_id": "Rohan3/ShapeNetCore_Captions",
                "description": "Provides textual captions for models in the ShapeNetCore dataset. Used for pretraining text-3D alignment.",
            },
            "text2cad": {
                "path": os.path.join(root_dir, "text2cad"),
                "repo_id": "SadilKhan/Text2CAD",
                "description": "A dataset of CAD models paired with synthetic text descriptions, useful for pretraining text encoders.",
            },
            "custom": {
                "path": root_dir,
                "urls": {
                    "train_data": "https://disk.360.yandex.ru/d/7FWyaYtL5OxRVQ",
                    "test_data": "https://disk.yandex.ru/d/DyHVEO6lZg95MA",
                },
                "description": "The primary hackathon dataset with ~500 simple CAD models, 26 stylized renders per model, and text descriptions.",
            },
        }

    def _get_process_pool(self):
        try:
            context = multiprocessing.get_context("spawn")
            return ProcessPoolExecutor(
                max_workers=self.max_workers, mp_context=context
            )
        except Exception:
            logging.warning(
                "ProcessPoolExecutor failed. Falling back to ThreadPoolExecutor for CPU tasks (slower)."
            )
            return None

    def _parallel_execute(self, jobs, desc, is_io_bound=False):
        results = {}
        if not jobs:
            return results

        use_processes = not is_io_bound and self._get_process_pool() is not None
        Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        max_workers = self.max_workers if use_processes else self.max_io_workers

        with Executor(max_workers=max_workers) as pool:
            futures = {pool.submit(job[0], *job[1]): job for job in jobs}

            for future in tqdm(
                as_completed(futures), total=len(jobs), desc=desc
            ):
                original_job = futures[future]
                try:
                    key, error = future.result()
                    if error:
                        logging.warning(f"Task failed for '{key}': {error}")
                    results[key] = error
                except Exception as e:
                    key = original_job[1][0]
                    logging.error(f"Task for '{key}' raised an exception: {e}")
                    results[key] = str(e)
        return results

    def prepare_all_datasets(self):
        pipeline = [
            self.prepare_thingi10k,
            self.prepare_modelnet40,
            self.prepare_abc_dataset,
            self.prepare_text2cad,
            self.prepare_objectnet3d,
            self.prepare_shapenet,
            self.prepare_shapenet_captions,
            self.prepare_custom_dataset,
        ]
        for i, func in enumerate(pipeline, 1):
            logging.info(
                f"--- [{i}/{len(pipeline)}] Starting: {func.__name__} ---"
            )
            try:
                func()
            except Exception as e:
                logging.error(
                    f"FATAL ERROR in {func.__name__}: {e}", exc_info=True
                )
            logging.info(f"--- Finished: {func.__name__} ---")
        logging.info("All dataset preparations complete.")

    def prepare_thingi10k(self):
        path = self.config["thingi10k"]["path"]
        models_out_dir = os.path.join(path, "models")
        os.makedirs(models_out_dir, exist_ok=True)
        try:
            thingi10k.init()
        except Exception as e:
            logging.error(f"Thingi10k initialization failed: {e}")
            return
        jobs = []
        for entry in tqdm(
            thingi10k.dataset(), desc="Collecting Thingi10k jobs"
        ):
            npz_path, file_id = entry.get("file_path"), entry.get("file_id")
            if not npz_path or not file_id:
                continue
            dst_path = os.path.join(models_out_dir, f"{file_id}.stl")
            if not os.path.exists(dst_path):
                jobs.append(
                    (_thingi10k_worker, (npz_path, dst_path, self.target_faces))
                )
        if not jobs:
            logging.info("Thingi10k is already up to date.")
            return
        self._parallel_execute(jobs, "Processing Thingi10k")

    def prepare_modelnet40(self):
        cfg = self.config["modelnet"]
        out_dir = cfg["path"]
        if os.path.exists(out_dir) and any(os.scandir(out_dir)):
            logging.info("ModelNet40 appears processed. Skipping.")
            return
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "ModelNet40.zip")
            logging.info("Downloading ModelNet40...")
            _, error = _download_worker(cfg["url"], zip_path)
            if error:
                logging.error(f"Failed to download ModelNet40: {error}")
                return
            logging.info("Extracting ModelNet40...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(temp_dir)
            source_dir = os.path.join(temp_dir, "ModelNet40")
            if not os.path.exists(source_dir):
                logging.error("Extracted 'ModelNet40' folder not found.")
                return
            jobs = []
            for cat_dir in os.scandir(source_dir):
                if not cat_dir.is_dir():
                    continue
                for split in ["train", "test"]:
                    split_dir = os.path.join(cat_dir.path, split)
                    if not os.path.exists(split_dir):
                        continue
                    for off_file in os.scandir(split_dir):
                        if off_file.name.endswith(".off"):
                            dst_path = os.path.join(
                                out_dir,
                                cat_dir.name,
                                "models",
                                off_file.name.replace(".off", ".stl"),
                            )
                            if not os.path.exists(dst_path):
                                jobs.append(
                                    (
                                        _convert_to_stl_worker,
                                        (
                                            off_file.path,
                                            dst_path,
                                            self.target_faces,
                                        ),
                                    )
                                )
            if jobs:
                self._parallel_execute(jobs, "Processing ModelNet40")

    def prepare_abc_dataset(self, archives=range(35, 41)):
        cfg = self.config["abc"]
        models_out_dir = os.path.join(cfg["path"], "models")
        os.makedirs(models_out_dir, exist_ok=True)
        if (
            os.path.exists(models_out_dir)
            and len(os.listdir(models_out_dir)) > 1000
        ):
            logging.info("ABC dataset appears to be processed. Skipping.")
            return
        temp_dir_path = os.path.join(self.root_dir, "data_temp")
        os.makedirs(temp_dir_path, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="abc_dataset_", dir=temp_dir_path
        ) as temp_dir:
            logging.info("Downloading ABC dataset manifest...")
            manifest_path = os.path.join(temp_dir, "stl2_v00.txt")
            _, error = _download_worker(cfg["manifest_url"], manifest_path)
            if error:
                logging.error(f"Failed to download ABC manifest: {error}")
                return

            download_jobs = []
            with open(manifest_path, "r") as f:
                urls = [line.strip().split()[0] for line in f if line.strip()]

            req_urls = [urls[i] for i in archives]
            for i, url in enumerate(req_urls):
                archive_path = os.path.join(temp_dir, f"abc_{i}.7z")
                if not os.path.exists(archive_path):
                    download_jobs.append(
                        (_download_worker, (url, archive_path))
                    )

            if download_jobs:
                self._parallel_execute(
                    download_jobs, "Downloading ABC archives", is_io_bound=True
                )

            extraction_jobs = [
                (_extract_7z_worker, (os.path.join(temp_dir, f), temp_dir))
                for f in os.listdir(temp_dir)
                if f.endswith(".7z")
            ]
            if extraction_jobs:
                self._parallel_execute(
                    extraction_jobs, "Extracting ABC archives"
                )

            final_jobs = []
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith(".stl"):
                        src_path = os.path.join(root, file)
                        dst_path = os.path.join(models_out_dir, file)
                        if not os.path.exists(dst_path):
                            # ABC models are already STL, so we just copy and simplify
                            final_jobs.append(
                                (
                                    _convert_to_stl_worker,
                                    (src_path, dst_path, self.target_faces),
                                )
                            )
            if final_jobs:
                self._parallel_execute(
                    final_jobs, "Simplifying and moving ABC models"
                )

    def prepare_objectnet3d(self):
        """Downloads, extracts, converts, and links ObjectNet3D data."""
        cfg = self.config["objectnet"]
        out_dir = cfg["path"]
        if os.path.exists(out_dir) and any(os.scandir(out_dir)):
            logging.info("ObjectNet3D directory not empty; skipping.")
            return

        with tempfile.TemporaryDirectory(prefix="objectnet_") as temp_dir:
            # 1. Download all zip files in parallel
            download_jobs = []
            for name, url in cfg["urls"].items():
                zip_path = os.path.join(temp_dir, f"{name}.zip")
                download_jobs.append((_download_worker, (url, zip_path)))

            self._parallel_execute(
                download_jobs, "Downloading ObjectNet3D", is_io_bound=True
            )

            # 2. Extract all zip files serially
            for name in cfg["urls"]:
                zip_path = os.path.join(temp_dir, f"{name}.zip")
                if os.path.exists(zip_path):
                    with zipfile.ZipFile(zip_path, "r") as z:
                        z.extractall(temp_dir)

            # 3. Collect and run CAD conversion jobs in parallel
            source_cad_dir = os.path.join(temp_dir, "ObjectNet3D", "CAD", "off")
            conversion_jobs = []
            if os.path.exists(source_cad_dir):
                for cat_dir in os.scandir(source_cad_dir):
                    if not cat_dir.is_dir():
                        continue
                    for off_file in os.scandir(cat_dir.path):
                        if (
                            off_file.name.endswith(".off")
                            and len(off_file.name.split(".")[0]) == 2
                        ):
                            dst_path = os.path.join(
                                out_dir,
                                cat_dir.name,
                                "models",
                                off_file.name.replace(".off", ".stl"),
                            )
                            if not os.path.exists(dst_path):
                                conversion_jobs.append(
                                    (
                                        _convert_to_stl_worker,
                                        (
                                            off_file.path,
                                            dst_path,
                                            self.target_faces,
                                        ),
                                    )
                                )
            if conversion_jobs:
                self._parallel_execute(
                    conversion_jobs, "Processing ObjectNet3D CADs"
                )

            # 4. Link images based on annotations serially
            ann_dir = os.path.join(temp_dir, "ObjectNet3D", "Annotations")
            img_dir = os.path.join(temp_dir, "ObjectNet3D", "Images")
            if os.path.exists(ann_dir) and os.path.exists(img_dir):
                for mat_file in tqdm(
                    os.scandir(ann_dir), desc="Linking ObjectNet3D Images"
                ):
                    if not mat_file.name.endswith(".mat"):
                        continue
                    try:
                        mat = scipy.io.loadmat(mat_file.path)
                        record = mat["record"][0, 0]
                        img_filename = str(record["filename"][0])

                        # Handle multiple objects in one image; link to all relevant categories
                        for obj in record["objects"][0]:
                            category_name = str(obj["class"][0])
                            src_img_path = os.path.join(img_dir, img_filename)
                            if os.path.exists(src_img_path):
                                dest_img_dir = os.path.join(
                                    out_dir, category_name, "images"
                                )
                                os.makedirs(dest_img_dir, exist_ok=True)
                                shutil.copy2(
                                    src_img_path,
                                    os.path.join(dest_img_dir, img_filename),
                                )
                    except Exception as e:
                        logging.warning(
                            f"Could not process annotation {mat_file.name}: {e}"
                        )

    def prepare_shapenet(self):
        cfg = self.config["shapenet"]
        out_dir = cfg["path"]
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            logging.warning(
                "HF_TOKEN environment variable not set. Skipping ShapeNet."
            )
            return
        try:
            from huggingface_hub import login, snapshot_download

            login(token=hf_token)
            cache_dir = snapshot_download(
                repo_id=cfg["repo_id"], repo_type="dataset"
            )
        except Exception as e:
            logging.error(f"Could not download ShapeNet from Hugging Face: {e}")
            return
        zip_files = sorted(
            [f for f in os.listdir(cache_dir) if f.endswith(".zip")]
        )
        for zip_filename in tqdm(
            zip_files, desc="Processing ShapeNet Categories"
        ):
            category_id = zip_filename.replace(".zip", "")
            models_out_dir = os.path.join(out_dir, category_id, "models")
            if os.path.exists(models_out_dir) and any(
                os.scandir(models_out_dir)
            ):
                continue
            images_out_dir = os.path.join(out_dir, category_id, "images")
            os.makedirs(models_out_dir, exist_ok=True)
            os.makedirs(images_out_dir, exist_ok=True)
            zip_filepath = os.path.join(cache_dir, zip_filename)
            jobs = []
            try:
                with zipfile.ZipFile(zip_filepath, "r") as z:
                    all_files = z.namelist()
                    model_ids = sorted(
                        list(
                            {
                                f.split("/")[1]
                                for f in all_files
                                if f.count("/") >= 2
                            }
                        )
                    )
                    for model_id in model_ids:
                        screenshot_prefix = (
                            f"{category_id}/{model_id}/screenshots/"
                        )
                        if (
                            any(
                                f.startswith(screenshot_prefix)
                                for f in all_files
                            )
                            or True
                        ):
                            jobs.append(
                                (
                                    _process_shapenet_model_worker,
                                    (
                                        zip_filepath,
                                        category_id,
                                        model_id,
                                        models_out_dir,
                                        images_out_dir,
                                        self.target_faces,
                                    ),
                                )
                            )
            except Exception as e:
                logging.error(f"Failed to read zip {zip_filename}: {e}")
                continue
            if jobs:
                self._parallel_execute(
                    jobs, f"Processing models in {category_id}"
                )
            if not any(os.scandir(models_out_dir)):
                logging.info(
                    f"Removing empty category directory: {category_id}"
                )
                shutil.rmtree(
                    os.path.join(out_dir, category_id), ignore_errors=True
                )

    def prepare_shapenet_captions(self):
        cfg = self.config["shapenet_captions"]
        out_dir = cfg["path"]
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            logging.warning(
                "HF_TOKEN environment variable not set. Skipping ShapeNet Captions."
            )
            return

        try:
            from huggingface_hub import login

            login(token=hf_token)
            dataset = load_dataset(cfg["repo_id"])
        except Exception as e:
            logging.error(
                f"Could not download ShapeNet Captions from Hugging Face: {e}"
            )
            return

        logging.info("Processing ShapeNet Captions...")
        df = dataset["train"].to_pandas()

        for category_id, group in tqdm(
            df.groupby("Class"), desc="Processing ShapeNet Captions"
        ):
            category_id = category_id.replace("'", "")
            category_path = os.path.join(out_dir, category_id)
            if not os.path.exists(category_path):
                logging.warning(
                    f"Category directory {category_path} does not exist. Skipping captions."
                )
                continue

            captions = {}
            for _, row in group.iterrows():
                model_id = row["Subclass"]
                caption = row["Caption"]
                if model_id in captions:
                    if isinstance(captions[model_id], list):
                        captions[model_id].append(caption)
                    else:
                        captions[model_id] = [captions[model_id], caption]
                else:
                    captions[model_id] = caption

            captions_path = os.path.join(category_path, "captions.json")
            try:
                with open(captions_path, "w") as f:
                    json.dump(captions, f, indent=2)
            except Exception as e:
                logging.error(
                    f"Failed to write captions for category {category_id}: {e}"
                )

    def prepare_custom_dataset(self):
        cfg = self.config["custom"]
        out_dir = cfg["path"]
        # if os.path.exists(out_dir) and any(os.scandir(out_dir)):
        #     logging.info("Custom dataset directory not empty. Skipping.")
        #     return
        download_jobs = []
        for name, public_url in cfg["urls"].items():
            try:
                api_url = f"https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={public_url}"
                response = requests.get(api_url, timeout=15)
                response.raise_for_status()
                href = response.json().get("href")
                if href:
                    zip_path = os.path.join(self.root_dir, f"temp_{name}.zip")
                    download_jobs.append((_download_worker, (href, zip_path)))
            except Exception as e:
                logging.error(f"Could not get Yandex.Disk URL for {name}: {e}")
        if not download_jobs:
            return
        results = self._parallel_execute(
            download_jobs, "Downloading Custom Data", is_io_bound=True
        )
        for zip_path, error in results.items():
            if error:
                logging.error(
                    f"Failed to download {os.path.basename(zip_path)}: {error}"
                )
                continue
            try:
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(out_dir)
            except Exception as e:
                logging.error(f"Failed to extract {zip_path}: {e}")
            finally:
                if os.path.exists(zip_path):
                    os.remove(zip_path)

    def prepare_text2cad(self):
        """Downloads and processes Text2CAD dataset."""
        try:
            import pandas as pd
            from huggingface_hub import hf_hub_download
        except ImportError:
            logging.error(
                "huggingface_hub or pandas is not installed. Skipping Text2CAD."
            )
            return

        dataset_name = "text2cad"
        download_path = os.path.join(self.root_dir, dataset_name)
        os.makedirs(download_path, exist_ok=True)
        logging.info(f"Start preparing {dataset_name}")

        repo_id = "SadilKhan/Text2CAD"

        try:
            v1_0_path = hf_hub_download(
                repo_id=repo_id,
                filename="text2cad_v1.0/text2cad_v1.0.csv",
                repo_type="dataset",
                local_dir=download_path,
            )
            v1_1_path = hf_hub_download(
                repo_id=repo_id,
                filename="text2cad_v1.1/text2cad_v1.1.csv",
                repo_type="dataset",
                local_dir=download_path,
            )
        except Exception as e:
            logging.error(f"Could not download Text2CAD from Hugging Face: {e}")
            return

        captions = {}

        # Process v1.0
        df_v1_0 = pd.read_csv(v1_0_path)
        description_cols_v1_0 = [
            "abstract",
            "beginner",
            "intermediate",
            "expert",
        ]
        for _, row in tqdm(
            df_v1_0.iterrows(),
            total=len(df_v1_0),
            desc="Processing Text2CAD v1.0",
        ):
            if not isinstance(row["uid"], str):
                continue
            model_id = row["uid"].split("/")[-1]
            if model_id not in captions:
                captions[model_id] = []
            for col in description_cols_v1_0:
                if col in row and pd.notna(row[col]):
                    captions[model_id].append(row[col])

        # Process v1.1
        df_v1_1 = pd.read_csv(v1_1_path)
        description_cols_v1_1 = [
            "abstract",
            "beginner",
            "intermediate",
            "expert",
            "description",
            "keywords",
        ]
        for _, row in tqdm(
            df_v1_1.iterrows(),
            total=len(df_v1_1),
            desc="Processing Text2CAD v1.1",
        ):
            if not isinstance(row["uid"], str):
                continue
            model_id = row["uid"].split("/")[-1]
            if model_id not in captions:
                captions[model_id] = []
            for col in description_cols_v1_1:
                if col in row and pd.notna(row[col]):
                    captions[model_id].append(row[col])

        # Remove duplicates
        for model_id in captions:
            captions[model_id] = sorted(list(set(captions[model_id])))

        abc_path = self.config["abc"]["path"]
        os.makedirs(abc_path, exist_ok=True)
        with open(os.path.join(abc_path, "captions.json"), "w") as f:
            json.dump(captions, f, indent=2)

        logging.info(f"Finished preparing {dataset_name}")


if __name__ == "__main__":
    manager = DatasetsManager(
        root_dir="/mnt/new_disk/aiijc_data",
        max_workers=None,
        target_faces=NUM_SAMPLE_POINTS,
    )
    # manager.prepare_custom_dataset()


# %% [markdown]
# ## Define data classes
# ### Train datasets


# %%
class MultimodalDataset(Dataset):
    """
    A PyTorch Dataset for loading multimodal data (3D models, images, text).

    This dataset is designed to handle the specific data structure outlined in
    the project's documentation (USE_DATA.md). It supports loading data from
    different datasets (e.g., CustomDataset, Thingi10k) and splits (train, val, test).

    The directory structure is expected to be:
    <root_dir>/<dataset_name>/[category]/models/<model_name>.stl
    <root_dir>/<dataset_name>/[category]/images/<model_name>_[number].<jpeg/png>
    <root_dir>/<dataset_name>/[category]/captions.json

    Args:
        root_dir (str): The root directory where all datasets are stored.
        dataset_name (str): The name of the dataset to load.
        split (str): The data split to use ('train', 'validation', or 'test').
        transform_3d (Optional[Callable]): A function/transform to apply to the 3D model data.
        transform_image (Optional[Callable]): A function/transform to apply to the image data.
        transform_text (Optional[Callable]): A function/transform to apply to the text data.
        load_images (bool): Whether to load images.
        load_models (bool): Whether to load 3D models.
    """

    def __init__(
        self,
        root_dir: str = "/mnt/new_disk/aiijc_data",
        dataset_name: str = "train",
        split: str = "",
        description: str = "No description provided.",
        transform_3d: Optional[Callable] = None,
        transform_image: Optional[Callable] = None,
        transform_text: Optional[Callable] = None,
        load_images: bool = True,
        load_models: bool = True,
    ):
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.split = split
        self.description = description
        self.transform_3d = transform_3d
        self.transform_image = transform_image
        self.transform_text = transform_text
        self.load_images = load_images
        self.load_models = load_models

        self.dataset_path = os.path.join(self.root_dir, self.dataset_name)
        self.split_path = os.path.join(self.dataset_path, self.split)

        self.samples = self._create_samples()

    def _create_samples(self) -> List[Dict[str, Any]]:
        """
        Scans the dataset directory and creates a list of samples.
        Each sample is a dictionary containing paths to the model, images, and text.
        """
        samples = []
        model_paths = glob(
            os.path.join(self.split_path, "**/models/*.stl"), recursive=True
        )

        captions = {}
        caption_files = glob(
            os.path.join(self.split_path, "**/captions.json"), recursive=True
        )
        for caption_file in caption_files:
            with open(caption_file, "r") as f:
                captions.update(json.load(f))

        for model_path in model_paths:
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            image_dir = os.path.join(
                os.path.dirname(os.path.dirname(model_path)), "images"
            )
            image_paths = glob(os.path.join(image_dir, f"{model_name}_*.*"))
            text_data = captions.get(model_name, [])
            if isinstance(text_data, str):
                text_data = [text_data]

            samples.append(
                {
                    "model_name": model_name,
                    "model_path": model_path,
                    "image_paths": image_paths,
                    "texts": text_data,
                }
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Fetches a single sample from the dataset.

        Args:
            idx (int): The index of the sample to fetch.

        Returns:
            A dictionary containing the processed data for the sample:
            {
                "model_name": str,
                "model": (Optional) Tensor for the 3D model,
                "images": (Optional) List of Tensors for the images,
                "texts": List of strings
            }
        """
        sample = self.samples[idx]
        model_data, image_data, text_data = None, [], sample["texts"]

        if self.load_models:
            try:
                mesh = trimesh.load(sample["model_path"], force="mesh")
                model_data = mesh.sample(1024)
                if self.transform_3d:
                    model_data = self.transform_3d(model_data)
            except Exception as e:
                logging.warning(
                    f"Could not load model {sample['model_path']}: {e}"
                )

        if self.load_images:
            for img_path in sample["image_paths"]:
                try:
                    image = Image.open(img_path).convert("RGB")
                    if self.transform_image:
                        image = self.transform_image(image)
                    image_data.append(image)
                except Exception as e:
                    logging.warning(f"Could not load image {img_path}: {e}")

        if self.transform_text:
            text_data = [self.transform_text(t) for t in text_data]

        return {
            "model_name": sample["model_name"],
            "model": model_data,
            "images": image_data,
            "texts": text_data,
        }

    def describe(self):
        """Prints the dataset's description."""
        print(f"\n--- Description for Dataset: {self.dataset_name} ---")
        print(self.description)
        print("-" * (33 + len(self.dataset_name)))

    def analyze(self, num_examples=3):
        """
        Analyzes the dataset, prints statistics, and shows some examples.
        Returns the statistics for aggregation.
        """
        if not self.samples:
            logging.warning(
                f"Dataset {self.dataset_name} is empty. Skipping analysis."
            )
            return 0, 0, 0, 0, 0

        num_models = len(self)
        num_images = sum(len(s["image_paths"]) for s in self.samples)
        num_texts = sum(len(s["texts"]) for s in self.samples)

        total_vertices, models_with_vertices = 0, 0
        for sample in tqdm(
            self.samples, desc=f"Analyzing vertices in {self.dataset_name}"
        ):
            try:
                mesh = trimesh.load(sample["model_path"], force="mesh")
                if isinstance(mesh, trimesh.Trimesh):
                    total_vertices += len(mesh.vertices)
                    models_with_vertices += 1
            except Exception as e:
                logging.warning(
                    f"Could not load model {sample['model_path']} to count vertices: {e}"
                )

        avg_vertices = (
            total_vertices / models_with_vertices
            if models_with_vertices > 0
            else 0
        )

        print(f"\n--- Analysis for Dataset: {self.dataset_name} ---")
        print(f"Total 3D Models: {num_models}")
        print(f"Total Images: {num_images}")
        print(f"Total Text Captions: {num_texts}")
        print(f"Average Vertices per Model: {avg_vertices:.2f}")
        print("-" * (28 + len(self.dataset_name)))

        if num_examples > 0 and num_models > 0:
            print(
                f"\n--- Displaying {min(num_examples, num_models)} random examples ---"
            )
            random_samples = random.sample(
                self.samples, min(num_examples, num_models)
            )

            for i, sample in enumerate(random_samples):
                print(f"\n--- Example {i + 1}: {sample['model_name']} ---")
                print("Texts:")
                print(
                    "\n".join(f"- {t}" for t in sample["texts"])
                    if sample["texts"]
                    else "- No text available."
                )

                if sample["image_paths"]:
                    num_images_to_show = min(len(sample["image_paths"]), 4)
                    fig, axs = plt.subplots(
                        1, num_images_to_show, figsize=(15, 4)
                    )
                    if num_images_to_show == 1:
                        axs = [axs]
                    fig.suptitle(f"Images for {sample['model_name']}")
                    for j, img_path in enumerate(
                        sample["image_paths"][:num_images_to_show]
                    ):
                        try:
                            axs[j].imshow(Image.open(img_path))
                            axs[j].axis("off")
                        except Exception as e:
                            axs[j].set_title("Could not load")
                            logging.warning(
                                f"Could not load image {img_path}: {e}"
                            )
                    plt.tight_layout()
                    plt.show()

                try:
                    mesh = trimesh.load(sample["model_path"], force="mesh")
                    scene = trimesh.Scene(mesh)
                    scene.camera_transform = scene.camera.look_at(
                        center=mesh.bounds[0], fov=60, points=mesh
                    )
                    img = Image.open(
                        io.BytesIO(scene.save_image(resolution=(600, 450)))
                    )
                    plt.figure(figsize=(8, 6))
                    plt.imshow(img)
                    plt.title(f"Render of {sample['model_name']}")
                    plt.axis("off")
                    plt.show()
                except Exception as e:
                    logging.warning(
                        f"Could not render model {sample['model_path']}: {e}"
                    )

        return (
            num_models,
            num_images,
            num_texts,
            total_vertices,
            models_with_vertices,
        )


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Creates a DataLoader for a given dataset.

    This is optimized for performance, especially when dealing with data on a slow HDD,
    by using multiple workers and pinning memory.

    Args:
        dataset (Dataset): The dataset to load data from.
        batch_size (int): The number of samples per batch.
        shuffle (bool): Whether to shuffle the data at every epoch.
        num_workers (int): The number of subprocesses to use for data loading.
        pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory
                           before returning them.

    Returns:
        A DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,  # Use a custom collate function for flexible batching
    )


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function to handle variable numbers of images and texts per sample.
    """
    model_names = [item["model_name"] for item in batch]
    models = [item["model"] for item in batch if item["model"] is not None]
    images = [
        img for item in batch for img in item["images"]
    ]  # Flatten list of images
    texts = [
        txt for item in batch for txt in item["texts"]
    ]  # Flatten list of texts

    # Stack models and images if they are tensors
    if models:
        models = torch.stack(models)
    # Note: Stacking images might require them to be the same size.
    # The transform_image should handle this (e.g., resize and convert to tensor).
    if images and all(isinstance(i, torch.Tensor) for i in images):
        images = torch.stack(images)

    return {
        "model_name": model_names,
        "model": models,
        "images": images,
        "texts": texts,
    }


# %% [markdown]
# ## Initialize All Datasets
# This section demonstrates how to initialize all prepared datasets into a single dictionary
# for easy access and use.

# %%
if __name__ == "__main__":
    all_datasets = {}
    logging.info("Initializing all datasets...")

    # Ensure the manager is created if this cell is run independently
    if "manager" not in locals():
        manager = DatasetsManager(
            root_dir="/mnt/new_disk/aiijc_data",
            max_workers=None,
            target_faces=NUM_SAMPLE_POINTS,
        )

    for name, cfg in manager.config.items():
        path = cfg["path"]
        if not os.path.exists(path):
            logging.warning(
                f"Path for dataset {name} ({path}) not found, skipping initialization."
            )
            continue

        description = cfg.get("description", "No description provided.")

        if name == "custom":
            for split in ["train", "test"]:
                split_path = os.path.join(manager.root_dir, split)
                if os.path.exists(split_path):
                    dataset_key = f"custom_{split}"
                    logging.info(f"Initializing {dataset_key}...")
                    all_datasets[dataset_key] = MultimodalDataset(
                        root_dir=manager.root_dir,
                        dataset_name=split,
                        split="",
                        description=description,
                    )
        elif name != "text2cad" and name != "shapenet_captions":
            logging.info(f"Initializing {name}...")
            all_datasets[name] = MultimodalDataset(
                root_dir=manager.root_dir,
                dataset_name=name,
                split="",
                description=description,
            )

    logging.info(
        f"Successfully initialized {len(all_datasets)} datasets: {list(all_datasets.keys())}"
    )


# %% [markdown]
# ## In-depth Analysis of a Single Dataset

# %%
if __name__ == "__main__":
    # Select a dataset to analyze
    target_dataset_name = "custom_train"

    if target_dataset_name in all_datasets:
        dataset_to_analyze = all_datasets[target_dataset_name]
        dataset_to_analyze.describe()
        dataset_to_analyze.analyze(num_examples=2)  # Show 2 random examples
    else:
        logging.warning(
            f"Dataset '{target_dataset_name}' not found. Available datasets: {list(all_datasets.keys())}"
        )


# %% [markdown]
# ## Summary Analysis of All Datasets

# %%
if __name__ == "__main__":
    total_models = 0
    total_images = 0
    total_texts = 0
    total_vertices_all = 0
    total_models_counted_all = 0

    print("\n\n==============================================")
    print("--- Starting Summary Analysis of All Datasets ---")
    print("==============================================")

    for name, dataset in all_datasets.items():
        # Get stats from the analyze method without showing examples
        (
            num_models,
            num_images,
            num_texts,
            total_vertices,
            models_with_vertices,
        ) = dataset.analyze(num_examples=0)

        # Aggregate stats
        total_models += num_models
        total_images += num_images
        total_texts += num_texts
        total_vertices_all += total_vertices
        total_models_counted_all += models_with_vertices

    print("\n===========================================")
    print("--- Total Summary Across All Datasets ---")
    print(f"Total 3D Models: {total_models}")
    print(f"Total Images: {total_images}")
    print(f"Total Text Captions: {total_texts}")
    if total_models_counted_all > 0:
        avg_vertices_total = total_vertices_all / total_models_counted_all
        print(f"Overall Average Vertices per Model: {avg_vertices_total:.2f}")
    print("===========================================")
