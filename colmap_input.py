import argparse
import cv2
import multiprocessing as mp
import numpy as np
import os
import shutil
import struct
import time
from typing import Dict, List, NamedTuple, Tuple
from PIL import Image

# ============================ read_model.py ============================#
class CameraModel(NamedTuple):
    model_id: int
    model_name: str
    num_params: int


class Camera(NamedTuple):
    id: int
    model: str
    width: int
    height: int
    params: List[float]


class Images(NamedTuple):
    id: int
    qvec: List[float]
    tvec: List[float]
    camera_id: int
    name: str
    point3d_ids: List[int] = []


class Point3D(NamedTuple):
    id: int
    xyz: List[float]
    rgb: List[int]
    error: float
    image_ids: List[int]
    point2d_ids: List[int]


CAMERA_MODELS = {
    CameraModel(0, "SIMPLE_PINHOLE", 3),
    CameraModel(1, "PINHOLE", 4),
    CameraModel(2, "SIMPLE_RADIAL", 4),
    CameraModel(3, "RADIAL", 5),
    CameraModel(4, "OPENCV", 8),
    CameraModel(5, "OPENCV_FISHEYE", 8),
    CameraModel(6, "FULL_OPENCV", 12),
    CameraModel(7, "FOV", 5),
    CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4),
    CameraModel(9, "RADIAL_FISHEYE", 5),
    CameraModel(10, "THIN_PRISM_FISHEYE", 12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])


def read_next_bytes(fid, num_bytes: int, format_char_sequence: str) -> Tuple:
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack("<" + format_char_sequence, data)


def read_cameras_text(path: str) -> Dict[int, Camera]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    model_cameras: Dict[int, Camera] = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elements = line.split()
                cam_id = int(elements[0])
                model = elements[1]
                width = int(elements[2])
                height = int(elements[3])
                params = list(map(float, elements[4:]))
                model_cameras[cam_id] = Camera(cam_id, model, width, height, params)
    return model_cameras


def read_cameras_binary(path: str) -> Dict[int, Camera]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    model_cameras: Dict[int, Camera] = {}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        print("num of cameras")
        print(num_cameras)
        for camera_line_index in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            cam_id = camera_properties[0]

            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = list(read_next_bytes(fid, 8 * num_params, "d" * num_params))
            model_cameras[cam_id] = Camera(cam_id, model_name, width, height, params)
        assert len(model_cameras) == num_cameras
    return model_cameras


def read_images_text(path: str) -> List[Images]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    model_images: List[Images] = []
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elements = line.split()
                im_id = int(elements[0])
                qvec = list(map(float, elements[1:5]))
                tvec = list(map(float, elements[5:8]))
                cam_id = int(elements[8])
                image_name = elements[9]
                elements = fid.readline().split()
                point3d_ids = list(map(int, elements[2::3]))
                model_images.append(Images(im_id, qvec, tvec, cam_id, image_name, point3d_ids))
    return model_images


def read_images_binary(path: str) -> List[Images]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    model_images: List[Images] = []
    with open(path, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            im_id = binary_image_properties[0]
            qvec = binary_image_properties[1:5]
            tvec = binary_image_properties[5:8]
            cam_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points_2d = read_next_bytes(fid, 8, "Q")[0]
            x_y_id_s = read_next_bytes(fid, 24 * num_points_2d, "ddq" * num_points_2d)
            point3d_ids = list(map(int, x_y_id_s[2::3]))
            model_images.append(Images(im_id, qvec, tvec, cam_id, image_name, point3d_ids))
    return model_images


def read_points_3d_text(path: str) -> Dict[int, Point3D]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    model_points3d: Dict[int, Point3D] = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elements = line.split()
                point_id = int(elements[0])
                xyz = list(map(float, elements[1:4]))
                rgb = list(map(int, elements[4:7]))
                error = float(elements[7])
                image_ids = list(map(int, elements[8::2]))
                point2d_ids = list(map(int, elements[9::2]))
                model_points3d[point_id] = Point3D(point_id, xyz, rgb, error, image_ids, point2d_ids)
    return model_points3d


def read_points3d_binary(path: str) -> Dict[int, Point3D]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    model_points3d: Dict[int, Point3D] = {}
    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for point_line_index in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, 43, "QdddBBBd")
            point_id = binary_point_line_properties[0]
            xyz = list(binary_point_line_properties[1:4])
            rgb = list(binary_point_line_properties[4:7])
            error = binary_point_line_properties[7]
            track_length = read_next_bytes(fid, 8, "Q")[0]
            track_elements = read_next_bytes(fid, 8 * track_length, "ii" * track_length)
            image_ids = list(map(int, track_elements[0::2]))
            point2d_ids = list(map(int, track_elements[1::2]))
            model_points3d[point_id] = Point3D(point_id, xyz, rgb, error, image_ids, point2d_ids)
    return model_points3d


def read_model(path: str, ext: str) -> Tuple[Dict[int, Camera], List[Images], Dict[int, Point3D]]:
    if ext == ".txt":
        model_cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        model_images = read_images_text(os.path.join(path, "images" + ext))
        model_points_3d = read_points_3d_text(os.path.join(path, "points3D") + ext)
    else:
        model_cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        model_images = read_images_binary(os.path.join(path, "images" + ext))
        model_points_3d = read_points3d_binary(os.path.join(path, "points3D") + ext)
    return model_cameras, model_images, model_points_3d


def quaternion_to_rotation_matrix(qvec: List[float]) -> np.ndarray:
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


_WORKER_VALID_POINT_IDS = None
_WORKER_CAMERA_CENTERS = None
_WORKER_POINT_XYZ = None
_WORKER_THETA0 = None
_WORKER_SIGMA1 = None
_WORKER_SIGMA2 = None


def log(message: str) -> None:
    print(f"[colmap_input] {message}", flush=True)


def progress_iter(iterable, total: int, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)
    except ImportError:
        def fallback():
            start = time.monotonic()
            last = start
            for idx, item in enumerate(iterable, 1):
                now = time.monotonic()
                if idx == 1 or idx == total or now - last >= 10:
                    elapsed = max(now - start, 1e-9)
                    rate = idx / elapsed
                    remaining = (total - idx) / rate if rate > 0 else 0
                    log(
                        f"{desc}: {idx}/{total} "
                        f"({100.0 * idx / max(total, 1):.1f}%) "
                        f"rate={rate:.1f}/s eta={remaining / 60:.1f}m"
                    )
                    last = now
                yield item

        return fallback()


def build_pair_queue(num_images: int, pair_window: int) -> List[Tuple[int, int]]:
    queue: List[Tuple[int, int]] = []
    for i in range(num_images):
        max_j = num_images if pair_window <= 0 else min(num_images, i + pair_window + 1)
        for j in range(i + 1, max_j):
            queue.append((i, j))
    return queue


def resolve_num_workers(num_workers: int, total_pairs: int) -> int:
    if total_pairs == 0:
        return 1
    if num_workers < 0:
        raise RuntimeError(f"num_workers must be >= 0: {num_workers}")
    if num_workers > 0:
        return num_workers
    return max(1, min((os.cpu_count() or 1) - 1, total_pairs))


def select_source_views(
    sorted_scores: List[Tuple[int, float]],
    max_sources: int,
    min_score: float,
    min_sources: int,
) -> Tuple[List[Tuple[int, float]], bool]:
    high_score = [item for item in sorted_scores if item[1] >= min_score]
    fallback = sorted_scores[:min_sources]
    fallback_used = len(high_score) < min_sources

    selected: List[Tuple[int, float]] = []
    seen = set()
    for image_id, score in fallback + high_score:
        if image_id in seen:
            continue
        selected.append((image_id, score))
        seen.add(image_id)

    selected = sorted(selected, key=lambda item: item[1], reverse=True)
    if max_sources >= 0:
        selected = selected[:max_sources]
    return selected, fallback_used


def log_view_selection_summary(view_sel: List[List[Tuple[int, float]]], fallback_refs: int) -> None:
    source_counts = [len(items) for items in view_sel]
    scores = [score for items in view_sel for _, score in items]
    zero_scores = sum(1 for score in scores if score == 0)
    if source_counts:
        log(
            "View selection summary: refs={} src/ref min/avg/max={}/{:.1f}/{} fallback_refs={}".format(
                len(source_counts),
                min(source_counts),
                sum(source_counts) / len(source_counts),
                max(source_counts),
                fallback_refs,
            )
        )
    if scores:
        log(
            "View selection scores: min/avg/max={:.6f}/{:.6f}/{:.6f} zero_scores={}/{}".format(
                min(scores),
                sum(scores) / len(scores),
                max(scores),
                zero_scores,
                len(scores),
            )
        )


def init_score_worker(
    valid_point_ids,
    camera_centers,
    point_xyz,
    theta0: float,
    sigma1: float,
    sigma2: float,
) -> None:
    global _WORKER_VALID_POINT_IDS
    global _WORKER_CAMERA_CENTERS
    global _WORKER_POINT_XYZ
    global _WORKER_THETA0
    global _WORKER_SIGMA1
    global _WORKER_SIGMA2

    _WORKER_VALID_POINT_IDS = valid_point_ids
    _WORKER_CAMERA_CENTERS = camera_centers
    _WORKER_POINT_XYZ = point_xyz
    _WORKER_THETA0 = theta0
    _WORKER_SIGMA1 = sigma1
    _WORKER_SIGMA2 = sigma2


def calc_score_fast(ind1: int, ind2: int) -> float:
    common_ids = _WORKER_VALID_POINT_IDS[ind1] & _WORKER_VALID_POINT_IDS[ind2]
    if not common_ids:
        return 0.0

    cam_center_i = _WORKER_CAMERA_CENTERS[ind1]
    cam_center_j = _WORKER_CAMERA_CENTERS[ind2]
    view_score = 0.0
    for pid in common_ids:
        p = _WORKER_POINT_XYZ[pid]
        ray_i = cam_center_i - p
        ray_j = cam_center_j - p
        denom = np.linalg.norm(ray_i) * np.linalg.norm(ray_j)
        if denom <= 0:
            continue
        cosine = np.clip(np.dot(ray_i, ray_j) / denom, -1.0, 1.0)
        theta = (180 / np.pi) * np.arccos(cosine)
        sigma = _WORKER_SIGMA1 if theta <= _WORKER_THETA0 else _WORKER_SIGMA2
        view_score += np.exp(-((theta - _WORKER_THETA0) ** 2) / (2 * sigma ** 2))
    return float(view_score)


def score_pair(pair: Tuple[int, int]) -> Tuple[int, int, float]:
    i, j = pair
    return i, j, calc_score_fast(i, j)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert colmap results into input for PatchmatchNet")

    parser.add_argument("--input_folder", type=str, help="Project input dir.")
    parser.add_argument("--output_folder", type=str, default="", help="Project output dir.")
    parser.add_argument("--num_src_images", type=int, default=-1, help="Related images")
    parser.add_argument("--theta0", type=float, default=5)
    parser.add_argument("--sigma1", type=float, default=1)
    parser.add_argument("--sigma2", type=float, default=10)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Pair-scoring worker processes. 0 auto-selects CPU threads - 1.")
    parser.add_argument("--min_src_score", type=float, default=0.0,
                        help="Minimum source-view score to keep before fallback.")
    parser.add_argument("--min_src_images", type=int, default=0,
                        help="Keep at least this many top-ranked source images when available.")
    parser.add_argument("--pair_window", type=int, default=0,
                        help="Only score image pairs within this index distance. 0 scores all pairs.")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Rewrite converted images even if numbered jpg outputs already exist.")
    parser.add_argument("--convert_format", action="store_true", default=False,
                        help="If set, convert image to jpg format.")
    parser.add_argument("--VGGT", action="store_true", default=False,
                        help="whether to use vggt.")
    parser.add_argument('--checkpoint', type=str, default='CVPR23_DeitS_Rerank.pth',
                        help='path to the checkpoint for R2Former')

    args = parser.parse_args()

    if not args.output_folder:
        args.output_folder = args.input_folder

    if args.input_folder is None or not os.path.isdir(args.input_folder):
        raise Exception("Invalid input folder")

    os.makedirs(args.output_folder, exist_ok=True)

    image_dir = os.path.join(args.input_folder, "images")
    model_dir = os.path.join(args.input_folder, "sparse")
    cam_dir = os.path.join(args.output_folder, "cams")
    renamed_dir = os.path.join(args.output_folder, "images")

    log(f"Reading COLMAP binary model from {model_dir}")
    cameras, images, points3d = read_model(model_dir, ".bin")
    num_images = len(images)
    log(f"Loaded {len(cameras)} cameras, {num_images} registered images, {len(points3d)} sparse points")
    if args.pair_window < 0:
        raise RuntimeError(f"pair_window must be >= 0: {args.pair_window}")
    if args.num_src_images == 0:
        args.num_src_images = -1
    if args.min_src_score < 0:
        raise RuntimeError(f"min_src_score must be >= 0: {args.min_src_score}")
    if args.min_src_images < 0:
        raise RuntimeError(f"min_src_images must be >= 0: {args.min_src_images}")

    param_type: Dict[str, List[str]] = {
        "SIMPLE_PINHOLE": ["f", "cx", "cy"],
        "PINHOLE": ["fx", "fy", "cx", "cy"],
        "SIMPLE_RADIAL": ["f", "cx", "cy", "k"],
        "SIMPLE_RADIAL_FISHEYE": ["f", "cx", "cy", "k"],
        "RADIAL": ["f", "cx", "cy", "k1", "k2"],
        "RADIAL_FISHEYE": ["f", "cx", "cy", "k1", "k2"],
        "OPENCV": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"],
        "OPENCV_FISHEYE": ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"],
        "FULL_OPENCV": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"],
        "FOV": ["fx", "fy", "cx", "cy", "omega"],
        "THIN_PRISM_FISHEYE": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1"]
    }

    if args.VGGT:
        from r2former import DistilledVisionTransformer
        from functools import partial
        import torch
        from torch import nn
        from torchvision import transforms
        model = DistilledVisionTransformer(
            img_size=[480, 640],
            patch_size=16,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=256
        )
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = {k.replace('module.backbone.', ''): v for k, v in checkpoint['model_state_dict'].items() if k.startswith('module.backbone')}
        model.load_state_dict(state_dict)
        device = torch.device("cuda")
        model.to(device)
        model = model.eval()

        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize([480, 640], antialias=False)
        ])

        global_feats = {}
        log("Extracting VGGT/R2Former retrieval features")
        for i in progress_iter(range(num_images), num_images, "r2former features"):
            img_path = os.path.join(image_dir, images[i].name)
            img = base_transform(Image.open(img_path).convert("RGB")).to(device)
            global_feat = model(img.unsqueeze(0))
            global_feats[images[i].name] = global_feat.squeeze(0).detach().cpu().numpy()

    # intrinsic
    intrinsic: Dict[int, np.ndarray] = {}
    log("Building intrinsics")
    for camera_id, cam in progress_iter(cameras.items(), len(cameras), "intrinsics"):
        params_dict = {key: value for key, value in zip(param_type[cam.model], cam.params)}
        if "f" in param_type[cam.model]:
            params_dict["fx"] = params_dict["f"]
            params_dict["fy"] = params_dict["f"]
        i = np.array([
            [params_dict["fx"], 0, params_dict["cx"]],
            [0, params_dict["fy"], params_dict["cy"]],
            [0, 0, 1]
        ])
        intrinsic[camera_id] = i
    print("intrinsic[1]\n", intrinsic[1], end="\n\n")

    # extrinsic
    extrinsic: List[np.ndarray] = []
    log("Building extrinsics")
    for i in progress_iter(range(num_images), num_images, "extrinsics"):
        e = np.zeros((4, 4))
        e[:3, :3] = quaternion_to_rotation_matrix(images[i].qvec)
        e[:3, 3] = images[i].tvec
        e[3, 3] = 1
        extrinsic.append(e)
    print("extrinsic[0]\n", extrinsic[0], end="\n\n")

    # depth range and interval
    depth_ranges: List[Tuple[float, float]] = []
    log("Estimating per-image sparse depth ranges")
    for i in progress_iter(range(num_images), num_images, "depth ranges"):
        zs = []
        for p3d_id in images[i].point3d_ids:
            if p3d_id == -1:
                continue
            transformed: np.ndarray = np.matmul(
                extrinsic[i], [points3d[p3d_id].xyz[0], points3d[p3d_id].xyz[1], points3d[p3d_id].xyz[2], 1])
            zs.append(transformed[2].item())
        zs_sorted = sorted(zs)
        if not zs_sorted:
            raise RuntimeError(f"No sparse depths available for image {images[i].name}")
        # relaxed depth range
        depth_min = zs_sorted[int(len(zs) * .01)]
        depth_max = zs_sorted[int(len(zs) * .99)]

        depth_ranges.append((depth_min, depth_max))
    print("depth_ranges[0]\n", depth_ranges[0], end="\n\n")

    def calc_score_vggt(ind1: int, ind2: int) -> float:
        view_score = float(np.dot(global_feats[images[ind1].name],
                                   global_feats[images[ind2].name].T))
        return view_score

    # view selection
    queue = build_pair_queue(num_images, args.pair_window)
    total_pairs = len(queue)
    pair_mode = "all-pairs" if args.pair_window <= 0 else f"window={args.pair_window}"
    log(f"Scoring source-view pairs: mode={pair_mode}, total_pairs={total_pairs}")

    view_scores: List[List[Tuple[int, float]]] = [[] for _ in range(num_images)]
    if args.VGGT:
        for i, j in progress_iter(queue, total_pairs, "pair scores"):
            s = calc_score_vggt(i, j)
            view_scores[i].append((j, s))
            view_scores[j].append((i, s))
    else:
        log("Precomputing sparse point id sets and camera centers")
        valid_point_ids = [set(p for p in image.point3d_ids if p != -1) for image in images]
        camera_centers = [
            -np.matmul(extrinsic[i][:3, :3].transpose(), extrinsic[i][:3, 3:4])[:, 0]
            for i in range(num_images)
        ]
        point_xyz = {pid: np.array(point.xyz, dtype=np.float64) for pid, point in points3d.items()}
        worker_count = resolve_num_workers(args.num_workers, total_pairs)
        log(f"Pair-scoring workers: {worker_count}")
        init_args = (
            valid_point_ids,
            camera_centers,
            point_xyz,
            args.theta0,
            args.sigma1,
            args.sigma2,
        )
        if worker_count == 1:
            init_score_worker(*init_args)
            results = (score_pair(pair) for pair in queue)
            for i, j, s in progress_iter(results, total_pairs, "pair scores"):
                view_scores[i].append((j, s))
                view_scores[j].append((i, s))
        else:
            ctx = mp.get_context("fork") if hasattr(os, "fork") else mp.get_context()
            chunksize = max(1, min(1000, total_pairs // (worker_count * 16) if total_pairs else 1))
            with ctx.Pool(worker_count, initializer=init_score_worker, initargs=init_args) as pool:
                results = pool.imap_unordered(score_pair, queue, chunksize=chunksize)
                for i, j, s in progress_iter(results, total_pairs, "pair scores"):
                    view_scores[i].append((j, s))
                    view_scores[j].append((i, s))

    max_sources = args.num_src_images if args.num_src_images >= 0 else -1

    view_sel: List[List[Tuple[int, float]]] = []
    fallback_refs = 0
    log("Sorting source-view candidates")
    for i in progress_iter(range(num_images), num_images, "view selection"):
        sorted_score = sorted(view_scores[i], key=lambda item: item[1], reverse=True)
        selected_views, fallback_used = select_source_views(
            sorted_score,
            max_sources,
            args.min_src_score,
            args.min_src_images,
        )
        if fallback_used:
            fallback_refs += 1
        view_sel.append(selected_views)
    print("view_sel[0]\n", view_sel[0], end="\n\n")
    log_view_selection_summary(view_sel, fallback_refs)

    # write
    os.makedirs(cam_dir, exist_ok=True)
    os.makedirs(renamed_dir, exist_ok=True)
    log(f"Writing camera files to {cam_dir}")
    for i in progress_iter(range(num_images), num_images, "write cams"):
        with open(os.path.join(cam_dir, "%08d_cam.txt" % i), "w") as f:
            f.write("extrinsic\n")
            for j in range(4):
                for k in range(4):
                    f.write(str(extrinsic[i][j, k]) + " ")
                f.write("\n")
            f.write("\nintrinsic\n")
            for j in range(3):
                for k in range(3):
                    f.write(str(intrinsic[images[i].camera_id][j, k]) + " ")
                f.write("\n")
            f.write("\n%f %f \n" % (depth_ranges[i][0], depth_ranges[i][1]))

    pair_path = os.path.join(args.output_folder, "pair.txt")
    log(f"Writing pair file to {pair_path}")
    with open(pair_path, "w") as f:
        f.write("%d\n" % len(images))
        for i, sorted_score in enumerate(view_sel):
            f.write("%d\n%d " % (i, len(sorted_score)))
            for image_id, s in sorted_score:
                f.write("%d %f " % (image_id, s))
            f.write("\n")

    log(f"Writing converted images to {renamed_dir}")
    converted = 0
    skipped = 0
    for i in progress_iter(range(num_images), num_images, "write images"):
        target_path = os.path.join(renamed_dir, "%08d.jpg" % i)
        if os.path.exists(target_path) and not args.overwrite:
            skipped += 1
            continue
        if args.convert_format:
            img = cv2.imread(os.path.join(image_dir, images[i].name))
            if img is None:
                raise RuntimeError(f"Failed to read image: {os.path.join(image_dir, images[i].name)}")
            cv2.imwrite(target_path, img)
        else:
            shutil.copyfile(os.path.join(image_dir, images[i].name),
                            target_path)
        converted += 1
    log(f"Image write complete: converted={converted}, skipped_existing={skipped}")
