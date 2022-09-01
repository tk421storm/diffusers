# Inference Examples

## Instructions for ONNX DML Execution on Windows
### Installing the dependencies

Before running the scripts, make sure to install the library's dependencies & this updated diffusers code (Python 3.8 is recommended):

```bash
git clone https://github.com/harishanand95/diffusers.git
cd diffusers && pip install -e .
pip install transformers ftfy scipy
```

## Clone ONNX runtime 

```bash
git clone https://github.com/microsoft/onnxruntime.git
cd onnxruntime 
```

## Build DML 
- The build instruction are from this [link](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html#build)

> 
    Requirements for building the DirectML execution provider:
    - Visual Studio 2017 toolchain (You can install VS 2022 too)
    - The Windows 10 SDK (10.0.18362.0) for Windows 10, version 1903 (or newer)

Open "x64 native Tools Command Prompt for VS 2022" application and run the following in the onnxruntime folder:

```bash
cd C:\Users\username\Desktop\pkgs\onnxruntime
build.bat --build_shared_lib --build_wheel --config Release --use_dml --cmake_generator "Visual Studio 17 2022" --parallel --skip_tests
```
If the above command fails, make sure to delete the build folder inside onnxruntime before retrying with changes.

## Install directml python package
After build, the whl files can be found in the following location `onnxruntime/build/Windows/Release/Release/dist/`.

```bash
pip install onnxruntime/build/Windows/Release/Release/dist/onnxruntime_directml-1.13.0-cp38-cp38-win_amd64.whl
```

## Create ONNX files
This step requires huggingface token. ALl onnx files are created in a folder named onnx in `examples/inference`

```bash
cd diffusers/examples/inference/
python save_onnx.py 
```

## Run using ONNX files
Run the onnx model using DirectML Execution Provider. Please check the last few lines in `dml_onnx.py` to see the examples.
Currently only 512x512 image is supported. 

```bash
cd diffusers/examples/inference/
python dml_onnx.py 
```


### Note: Please raise any issues on this repo, I'll take a look. thanks!


----------------------------------------------------------------------------------------




## Installing the dependencies

Before running the scripts, make sure to install the library's dependencies:

```bash
pip install diffusers transformers ftfy
```

## Image-to-Image text-guided generation with Stable Diffusion

The `image_to_image.py` script implements `StableDiffusionImg2ImgPipeline`. It lets you pass a text prompt and an initial image to condition the generation of new images. This example also showcases how you can write custom diffusion pipelines using `diffusers`!

### How to use it


```python
import torch
from torch import autocast
import requests
from PIL import Image
from io import BytesIO

from image_to_image import StableDiffusionImg2ImgPipeline, preprocess

# load the pipeline
device = "cuda"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16", 
    torch_dtype=torch.float16,
    use_auth_token=True
).to(device)

# let's download an initial image
url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((768, 512))
init_image = preprocess(init_image)

prompt = "A fantasy landscape, trending on artstation"

with autocast("cuda"):
    images = pipe(prompt=prompt, init_image=init_image, strength=0.75, guidance_scale=7.5)["sample"]

images[0].save("fantasy_landscape.png")
```
You can also run this example on colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/patil-suraj/Notebooks/blob/master/image_2_image_using_diffusers.ipynb)

## Tweak prompts reusing seeds and latents

You can generate your own latents to reproduce results, or tweak your prompt on a specific result you liked. [This notebook](stable-diffusion-seeds.ipynb) shows how to do it step by step. You can also run it in Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pcuenca/diffusers-examples/blob/main/notebooks/stable-diffusion-seeds.ipynb).


## In-painting using Stable Diffusion

The `inpainting.py` script implements `StableDiffusionInpaintingPipeline`. This script lets you edit specific parts of an image by providing a mask and text prompt.

### How to use it

```python
import torch
from io import BytesIO

from torch import autocast
import requests
import PIL

from inpainting import StableDiffusionInpaintingPipeline

def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))

device = "cuda"
pipe = StableDiffusionInpaintingPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16", 
    torch_dtype=torch.float16,
    use_auth_token=True
).to(device)

prompt = "a cat sitting on a bench"
with autocast("cuda"):
    images = pipe(prompt=prompt, init_image=init_image, mask_image=mask_image, strength=0.75)["sample"]

images[0].save("cat_on_bench.png")
```

You can also run this example on colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/patil-suraj/Notebooks/blob/master/in_painting_with_stable_diffusion_using_diffusers.ipynb)
