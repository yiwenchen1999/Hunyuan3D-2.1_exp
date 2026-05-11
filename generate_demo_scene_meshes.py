import sys
from pathlib import Path
import re
from collections import defaultdict

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "hy3dshape"))
sys.path.insert(0, str(ROOT_DIR / "hy3dpaint"))

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline


VIEW_PATTERN = re.compile(r".*_view_(\d+)\.png$")
ITER_PATTERN = re.compile(r"iter_(\d+)$")


def generate_one_mesh(
    shape_pipeline: Hunyuan3DDiTFlowMatchingPipeline,
    paint_pipeline: Hunyuan3DPaintPipeline,
    image_path: Path,
    output_glb_path: Path,
) -> Path:
    output_glb_path.parent.mkdir(parents=True, exist_ok=True)
    untextured_mesh_path = output_glb_path.with_name(f"{output_glb_path.stem}_untextured.glb")

    mesh_untextured = shape_pipeline(image=str(image_path))[0]
    mesh_untextured.export(str(untextured_mesh_path))

    textured_output = paint_pipeline(
        mesh_path=str(untextured_mesh_path),
        output_mesh_path=str(output_glb_path),
        image_path=str(image_path),
    )
    return Path(textured_output)


def collect_jobs() -> list[tuple[Path, Path]]:
    roots = [
        ROOT_DIR / "relight2views/objaverse",
        ROOT_DIR / "relight2views/polyhaven",
    ]

    # Keep only the latest iter for each (scene, view).
    best_candidates: dict[tuple[Path, int], tuple[int, Path]] = {}

    for root in roots:
        for image_path in root.glob("**/iter_*/*_view_*.png"):
            iter_dir = image_path.parent
            scene_dir = iter_dir.parent
            view_name = image_path.name.lower()

            # Only process context views; explicitly discard target views.
            if not view_name.startswith("context"):
                continue
            if view_name.startswith("target"):
                continue

            view_match = VIEW_PATTERN.match(image_path.name)
            iter_match = ITER_PATTERN.match(iter_dir.name)
            if view_match is None or iter_match is None:
                continue

            view_idx = int(view_match.group(1))
            iter_idx = int(iter_match.group(1))
            key = (scene_dir, view_idx)

            if key not in best_candidates or iter_idx > best_candidates[key][0]:
                best_candidates[key] = (iter_idx, image_path)

    jobs_by_scene: dict[Path, list[tuple[int, Path]]] = defaultdict(list)
    for (scene_dir, view_idx), (_, image_path) in best_candidates.items():
        jobs_by_scene[scene_dir].append((view_idx, image_path))

    jobs: list[tuple[Path, Path]] = []
    for scene_dir in sorted(jobs_by_scene.keys()):
        for view_idx, image_path in sorted(jobs_by_scene[scene_dir], key=lambda x: x[0]):
            output_glb_path = scene_dir / f"mesh_context_{view_idx:02d}.glb"
            jobs.append((image_path, output_glb_path))

    return jobs


def main() -> None:
    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")
    paint_pipeline = Hunyuan3DPaintPipeline(Hunyuan3DPaintConfig(max_num_view=6, resolution=512))

    jobs = collect_jobs()
    print(f"Found {len(jobs)} views to process across objaverse/polyhaven.")

    for image_path, output_glb_path in jobs:
        print(f"Generating {output_glb_path} from {image_path}")
        result_path = generate_one_mesh(shape_pipeline, paint_pipeline, image_path, output_glb_path)
        print(f"Generated: {result_path}")


if __name__ == "__main__":
    main()
