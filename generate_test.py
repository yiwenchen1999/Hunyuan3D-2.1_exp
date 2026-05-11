import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

input_image_path = 'relight2views/polyhaven/modern_arm_chair_01_env_1/iter_00000175/input_view_01.jpg'

# let's generate a mesh first
shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1')
mesh_untextured = shape_pipeline(image=input_image_path)[0]
mesh_untextured.export('modern_arm_chair_untextured.glb')

paint_pipeline = Hunyuan3DPaintPipeline(Hunyuan3DPaintConfig(max_num_view=6, resolution=512))
mesh_textured = paint_pipeline(
    mesh_path='modern_arm_chair_untextured.glb',
    output_mesh_path='modern_arm_chair_textured.glb',
    image_path=input_image_path,
)