# ui/gradio_app.py

import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

MODEL_ID = "runwayml/stable-diffusion-v1-5"


def load_pipeline():
    """
    Load SD 1.5 pipeline once at startup.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    pipe = pipe.to(device)
    pipe.safety_checker = None  # baseline demo, 暂不做复杂安全策略

    return pipe, device


# 全局只加载一次，避免每次点击都重新下载模型
pipe, device = load_pipeline()


def generate_image(prompt: str, steps: int = 25, guidance_scale: float = 7.5):
    """
    Gradio 回调函数：
    根据用户输入的 prompt 生成一张 512x512 的图像。
    """
    if not prompt or prompt.strip() == "":
        return None

    generator = torch.Generator(device=device).manual_seed(42)

    extra_kwargs = {}
    if device == "cuda":
        extra_kwargs["torch_dtype"] = torch.float16

    with torch.autocast(device) if device == "cuda" else torch.no_grad():
        image = pipe(
            prompt,
            height=512,
            width=512,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

    return image


def build_demo():
    """
    构建 Gradio 界面：Textbox + Slider + Image 输出。
    """
    prompt_box = gr.Textbox(
        label="Prompt",
        lines=2,
        placeholder="e.g. a cute cat sitting on the moon, digital art",
        value="a cute cat sitting on the moon, digital art",
    )

    steps_slider = gr.Slider(
        minimum=10,
        maximum=50,
        value=25,
        step=1,
        label="Inference steps",
    )

    gs_slider = gr.Slider(
        minimum=1.0,
        maximum=12.0,
        value=7.5,
        step=0.5,
        label="CFG scale",
    )

    demo = gr.Interface(
        fn=generate_image,
        inputs=[prompt_box, steps_slider, gs_slider],
        outputs=gr.Image(type="pil", label="Generated image"),
        title="Controllable Text-to-Image (SD1.5 Baseline)",
        description="Minimal SD1.5 Gradio demo: enter a prompt and generate a 512×512 image.",
    )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    # share=True 方便在 Colab 中通过公网访问；本地可以去掉 share=True
    demo.launch(share=True)

