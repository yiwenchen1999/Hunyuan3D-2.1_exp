import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "hy3dshape"))
sys.path.insert(0, str(ROOT_DIR / "hy3dpaint"))

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline


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


def main() -> None:
    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")
    paint_pipeline = Hunyuan3DPaintPipeline(Hunyuan3DPaintConfig(max_num_view=6, resolution=512))

    jobs = [
        (
            ROOT_DIR / "relight2views/demo_scene/modern_arm_chair_01_env_1/iter_00000175/input512_view_00.png",
            ROOT_DIR / "relight2views/demo_scene/modern_arm_chair_01_env_1/mesh_00.glb",
        ),
        (
            ROOT_DIR / "relight2views/demo_scene/modern_arm_chair_01_env_1/iter_00000175/input512_view_01.png",
            ROOT_DIR / "relight2views/demo_scene/modern_arm_chair_01_env_1/mesh_01.glb",
        ),
    ]

    for image_path, output_glb_path in jobs:
        result_path = generate_one_mesh(shape_pipeline, paint_pipeline, image_path, output_glb_path)
        print(f"Generated: {result_path}")


if __name__ == "__main__":
    main()
