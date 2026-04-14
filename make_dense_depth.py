#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path("/3dgs_pipe") if Path("/3dgs_pipe").is_dir() else SCRIPT_DIR.parents[1]
DEFAULT_CKPT = SCRIPT_DIR / "checkpoints" / "casdiffmvs_blendmvg.ckpt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create DiffMVS dense depth maps and a fused dense point cloud."
    )
    parser.add_argument("source_folder", help="Source folder containing undistorted/")
    parser.add_argument(
        "--ckpt",
        default=str(DEFAULT_CKPT),
        help=f"DiffMVS checkpoint path (default: {DEFAULT_CKPT})",
    )
    parser.add_argument(
        "--method",
        default="casdiffmvs",
        choices=["casdiffmvs", "diffmvs"],
        help="DiffMVS method variant (default: casdiffmvs)",
    )
    parser.add_argument(
        "--num-view",
        type=int,
        default=5,
        help="Number of views passed to DiffMVS, including the reference view (default: 5)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Process at most this many generated MVS reference views",
    )
    parser.add_argument(
        "--image-stride",
        type=int,
        default=1,
        help="Process every Nth generated MVS reference view (default: 1)",
    )
    parser.add_argument(
        "--pair-window",
        type=int,
        default=0,
        help="Only compare images within this index distance during DiffMVS conversion (default: all pairs)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DiffMVS conversion pair-scoring workers; 0 auto-selects CPU threads - 1",
    )
    parser.add_argument(
        "--num-src-images",
        type=int,
        default=12,
        help="Maximum source images per reference in pair.txt; 0 keeps all candidates (default: 12)",
    )
    parser.add_argument(
        "--min-src-score",
        type=float,
        default=50.0,
        help="Minimum source-view score to keep before fallback (default: 50.0)",
    )
    parser.add_argument(
        "--min-src-images",
        type=int,
        default=5,
        help="Keep at least this many top-ranked source images when available (default: 5)",
    )
    parser.add_argument(
        "--filter-workers",
        type=int,
        default=0,
        help="Parallel workers for DiffMVS depth filtering; 0 auto-selects CPU threads - 1",
    )
    parser.add_argument(
        "--max-points-per-view",
        type=int,
        default=200000,
        help="Maximum fused points contributed by each reference view; 0 disables the cap (default: 200000)",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.01,
        help="Voxel size in meters for fused point cloud downsampling; 0 disables it (default: 0.01)",
    )
    parser.add_argument(
        "--filter-only",
        action="store_true",
        help="Reuse existing DiffMVS depth outputs and only rerun filtering/conversion",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing DiffMVS intermediate outputs before running",
    )
    return parser.parse_args()


def read_pfm(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = handle.readline().decode("utf-8").rstrip()
        if header == "PF":
            color = True
        elif header == "Pf":
            color = False
        else:
            raise RuntimeError(f"Not a PFM file: {path}")

        dim_line = handle.readline().decode("utf-8")
        dim_match = re.match(r"^(\d+)\s(\d+)\s$", dim_line)
        if not dim_match:
            raise RuntimeError(f"Malformed PFM dimensions in {path}")
        width, height = map(int, dim_match.groups())

        scale = float(handle.readline().rstrip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(handle, endian + "f")
        shape = (height, width, 3) if color else (height, width)
        return np.flipud(np.reshape(data, shape))


def write_depth_png(path: Path, depth: np.ndarray) -> None:
    millimeters = np.clip(np.nan_to_num(depth, nan=0.0) * 1000.0, 0, 65535)
    cv2.imwrite(str(path), millimeters.astype(np.uint16))


def write_depth_preview(path: Path, depth: np.ndarray) -> None:
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        cv2.imwrite(str(path), np.zeros((*depth.shape, 3), dtype=np.uint8))
        return
    near = np.percentile(depth[valid], 1)
    far = np.percentile(depth[valid], 99)
    normalized = np.clip((depth - near) / (far - near + 1e-8) * 255, 0, 255)
    preview = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    cv2.imwrite(str(path), preview)


def output_stem(image_id: int, image_name: str) -> str:
    stem = Path(image_name).with_suffix("").as_posix().replace("/", "__")
    return f"{image_id:08d}_{stem}"


def run_command(cmd: list[str], cwd: Path) -> None:
    print("[cmd]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def load_colmap_images(sparse_dir: Path):
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from colmap_input import read_model

    _, images, _ = read_model(str(sparse_dir), ".bin")
    return images


def selected_indices(num_images: int, image_stride: int, max_images: int | None) -> list[int]:
    if image_stride < 1:
        raise RuntimeError(f"Image stride must be >= 1: {image_stride}")
    indices = list(range(num_images))[::image_stride]
    if max_images is not None:
        if max_images < 1:
            raise RuntimeError(f"Max images must be >= 1: {max_images}")
        indices = indices[:max_images]
    return indices


def select_source_views(
    sorted_scores: list[tuple[int, float]],
    max_sources: int,
    min_score: float,
    min_sources: int,
) -> tuple[list[tuple[int, float]], bool]:
    high_score = [item for item in sorted_scores if item[1] >= min_score]
    fallback = sorted_scores[:min_sources]
    fallback_used = len(high_score) < min_sources

    selected: list[tuple[int, float]] = []
    seen = set()
    for image_id, score in fallback + high_score:
        if image_id in seen:
            continue
        selected.append((image_id, score))
        seen.add(image_id)

    selected = sorted(selected, key=lambda item: item[1], reverse=True)
    if max_sources > 0:
        selected = selected[:max_sources]
    return selected, fallback_used


def read_pair_scores(pair_path: Path) -> dict[int, list[tuple[int, float]]]:
    if not pair_path.is_file():
        return {}
    lines = pair_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return {}

    scores: dict[int, list[tuple[int, float]]] = {}
    cursor = 1
    try:
        num_refs = int(lines[0].strip())
        for _ in range(num_refs):
            ref_view = int(lines[cursor].strip())
            cursor += 1
            tokens = lines[cursor].strip().split()
            cursor += 1
            entries = []
            for idx in range(1, len(tokens), 2):
                entries.append((int(tokens[idx]), float(tokens[idx + 1])))
            scores[ref_view] = entries
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"Malformed DiffMVS pair file: {pair_path}") from exc
    return scores


def rewrite_pair_file(
    pair_path: Path,
    selected: list[int],
    max_sources: int,
    min_score: float,
    min_sources: int,
) -> None:
    if len(selected) < 2:
        raise RuntimeError("DiffMVS smoke runs need at least 2 selected images")
    selected_set = set(selected)
    existing_scores = read_pair_scores(pair_path)
    used_synthetic_scores = False
    fallback_refs = 0
    lines = [f"{len(selected)}\n"]
    for ref_view in selected:
        known_candidates = [
            (src_view, score)
            for src_view, score in existing_scores.get(ref_view, [])
            if src_view in selected_set and src_view != ref_view
        ]
        known_ids = {src_view for src_view, _ in known_candidates}
        synthetic_candidates = [
            (idx, 0.0)
            for idx in selected
            if idx != ref_view and idx not in known_ids
        ]
        if synthetic_candidates:
            used_synthetic_scores = True
        src_views, fallback_used = select_source_views(
            known_candidates + synthetic_candidates,
            max_sources,
            min_score,
            min_sources,
        )
        if fallback_used:
            fallback_refs += 1
        lines.append(f"{ref_view}\n")
        tokens = [str(len(src_views))]
        for src_view, score in src_views:
            tokens.extend([str(src_view), f"{score:.6f}"])
        lines.append(" ".join(tokens) + " \n")
    pair_path.write_text("".join(lines), encoding="utf-8")
    source_label = "all" if max_sources <= 0 else str(max_sources)
    print(
        f"[info] Rewrote {pair_path} for selected refs: {sorted(selected_set)}; "
        f"max sources/ref={source_label}; min score={min_score}; min sources={min_sources}; "
        f"fallback refs={fallback_refs}"
    )
    if used_synthetic_scores:
        print("[warn] Some selected subset source scores were unavailable; used score 0.0 fallback entries")


def has_complete_depth_outputs(raw_output_dir: Path, selected: list[int], method: str) -> bool:
    required_dirs = ["depth_est", "cams", "images", "conf0", "conf1"]
    if method == "casdiffmvs":
        required_dirs.append("conf2")

    missing = []
    for index in selected:
        checks = [
            raw_output_dir / "depth_est" / f"{index:08d}.pfm",
            raw_output_dir / "cams" / f"{index:08d}_cam.txt",
            raw_output_dir / "images" / f"{index:08d}.jpg",
        ]
        checks.extend(raw_output_dir / conf_dir / f"{index:08d}.pfm" for conf_dir in required_dirs if conf_dir.startswith("conf"))
        missing.extend(path for path in checks if not path.is_file())

    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:20])
        if len(missing) > 20:
            preview += f"\n  ... {len(missing) - 20} more"
        print(f"[info] Existing DiffMVS depth outputs are incomplete:\n{preview}")
        return False
    return True


def convert_depth_outputs(
    *,
    images,
    selected: list[int],
    raw_output_dir: Path,
    depth_dir: Path,
) -> None:
    depth_est_dir = raw_output_dir / "depth_est"
    if not depth_est_dir.is_dir():
        raise RuntimeError(f"Missing DiffMVS depth output folder: {depth_est_dir}")

    depth_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    converted = 0
    for index in selected:
        image = images[index]
        pfm_path = depth_est_dir / f"{index:08d}.pfm"
        if not pfm_path.is_file():
            missing.append(pfm_path)
            continue

        depth = read_pfm(pfm_path).astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        stem = output_stem(image.id, image.name)
        npy_path = depth_dir / f"{stem}.npy"
        png_path = depth_dir / f"{stem}.png"
        vis_path = depth_dir / f"{stem}_vis.png"
        np.save(npy_path, depth)
        write_depth_png(png_path, depth)
        write_depth_preview(vis_path, depth)
        converted += 1

    if missing:
        rendered = "\n".join(f"  - {path}" for path in missing)
        raise RuntimeError(f"Missing expected DiffMVS depth files:\n{rendered}")
    print(f"[info] Converted {converted} DiffMVS depth maps into {depth_dir}")


def validate_inputs(source_folder: Path, ckpt: Path, num_view: int) -> tuple[Path, Path, Path]:
    if not source_folder.is_dir():
        raise RuntimeError(f"Source folder is not a directory: {source_folder}")
    if not ckpt.is_file():
        raise RuntimeError(
            "DiffMVS checkpoint not found: "
            f"{ckpt}\nPass --ckpt /path/to/casdiffmvs_blendmvg.ckpt or place it at the default path."
        )
    if num_view < 2:
        raise RuntimeError(f"Num view must be >= 2: {num_view}")

    undistorted_dir = source_folder / "undistorted"
    images_dir = undistorted_dir / "images"
    sparse_dir = undistorted_dir / "sparse"
    if not images_dir.is_dir():
        raise RuntimeError(f"Missing undistorted images folder: {images_dir}")
    if not sparse_dir.is_dir():
        raise RuntimeError(f"Missing undistorted sparse folder: {sparse_dir}")
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        if not (sparse_dir / name).is_file():
            raise RuntimeError(f"DiffMVS integration requires binary COLMAP model file: {sparse_dir / name}")
    return undistorted_dir, images_dir, sparse_dir


def main() -> int:
    args = parse_args()
    if args.num_src_images < 0:
        raise RuntimeError(f"Num src images must be >= 0: {args.num_src_images}")
    if args.min_src_score < 0:
        raise RuntimeError(f"Min src score must be >= 0: {args.min_src_score}")
    if args.min_src_images < 0:
        raise RuntimeError(f"Min src images must be >= 0: {args.min_src_images}")
    if args.filter_workers < 0:
        raise RuntimeError(f"Filter workers must be >= 0: {args.filter_workers}")
    if args.max_points_per_view < 0:
        raise RuntimeError(f"Max points per view must be >= 0: {args.max_points_per_view}")
    if args.voxel_size < 0:
        raise RuntimeError(f"Voxel size must be >= 0: {args.voxel_size}")

    source_folder = Path(args.source_folder).expanduser().resolve()
    ckpt = Path(args.ckpt).expanduser().resolve()
    undistorted_dir, _, sparse_dir = validate_inputs(source_folder, ckpt, args.num_view)

    mvs_dir = undistorted_dir / "diffmvs_mvs"
    raw_output_dir = undistorted_dir / "diffmvs_outputs"
    depth_dir = undistorted_dir / "depth"
    fused_path = undistorted_dir / "fused_dense_point_cloud.ply"

    if args.overwrite:
        shutil.rmtree(mvs_dir, ignore_errors=True)
        shutil.rmtree(raw_output_dir, ignore_errors=True)

    images = load_colmap_images(sparse_dir)
    selected = selected_indices(len(images), args.image_stride, args.max_images)
    if not selected:
        raise RuntimeError("No registered COLMAP images selected")

    run_command(
        [
            sys.executable,
            "colmap_input.py",
            f"--input_folder={undistorted_dir}",
            f"--output_folder={mvs_dir}",
            f"--pair_window={args.pair_window}",
            f"--num_workers={args.num_workers}",
            f"--num_src_images={args.num_src_images if args.num_src_images > 0 else -1}",
            f"--min_src_score={args.min_src_score}",
            f"--min_src_images={args.min_src_images}",
            "--convert_format",
            *(["--overwrite"] if args.overwrite else []),
        ],
        SCRIPT_DIR,
    )
    if args.image_stride != 1 or args.max_images is not None:
        rewrite_pair_file(
            mvs_dir / "pair.txt",
            selected,
            args.num_src_images,
            args.min_src_score,
            args.min_src_images,
        )

    effective_num_view = min(args.num_view, len(selected))
    geo_mask_thres = min(2, max(1, effective_num_view - 1))
    outputs_complete = has_complete_depth_outputs(raw_output_dir, selected, args.method)
    if args.filter_only and not outputs_complete:
        raise RuntimeError("--filter-only was requested, but existing DiffMVS depth outputs are incomplete")
    run_depth = not args.filter_only and (args.overwrite or not outputs_complete)
    if not run_depth:
        print("[info] Reusing existing DiffMVS depth outputs; running filtering/conversion only")

    run_command(
        [
            sys.executable,
            "test.py",
            "--dataset=general",
            "--batch_size=1",
            f"--num_view={effective_num_view}",
            f"--method={args.method}",
            *(["--save_depth"] if run_depth else []),
            f"--testpath={mvs_dir}",
            "--numdepth_initial=48",
            "--numdepth=384",
            f"--loadckpt={ckpt}",
            f"--outdir={raw_output_dir}",
            "--scale",
            "0.0",
            "0.125",
            "0.025",
            "--sampling_timesteps",
            "0",
            "1",
            "1",
            "--ddim_eta",
            "0",
            "1",
            "1",
            "--stage_iters",
            "1",
            "3",
            "3",
            "--cost_dim_stage",
            "4",
            "4",
            "4",
            "--CostNum",
            "0",
            "4",
            "4",
            "--hidden_dim",
            "0",
            "32",
            "20",
            "--context_dim",
            "32",
            "32",
            "16",
            "--unet_dim",
            "0",
            "16",
            "8",
            "--min_radius=0.125",
            "--max_radius=8",
            "--geo_pixel_thres=0.125",
            "--geo_depth_thres=0.01",
            f"--geo_mask_thres={geo_mask_thres}",
            f"--filter_workers={args.filter_workers}",
            f"--max_points_per_view={args.max_points_per_view}",
            f"--voxel_size={args.voxel_size}",
            "--photo_thres",
            "0.3",
            "0.5",
            "0.5",
        ],
        SCRIPT_DIR,
    )

    convert_depth_outputs(
        images=images,
        selected=selected,
        raw_output_dir=raw_output_dir,
        depth_dir=depth_dir,
    )

    raw_pcd = raw_output_dir / "pc.ply"
    if not raw_pcd.is_file():
        raise RuntimeError(f"Missing DiffMVS fused point cloud: {raw_pcd}")
    shutil.copyfile(raw_pcd, fused_path)
    print(f"[info] Wrote fused dense point cloud: {fused_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
