# __vari'art__
Use variational autoencoders to perfom image latent analysis and generate new images.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/hugros-93/variart/actions/workflows/main.yml/badge.svg)](https://github.com/hugros-93/kichtai/actions/workflows/main.yml)

- Set of tools for the preprocessing of videos or sets of images
- Train a VAE using __tensorflow__
- Perform latent space analysis, generate new images and create GIFs

![GIF DrillFR4](https://github.com/hugros-93/variart/blob/init/outputs/gif_DrillFR4.gif)

```python
from variart.preprocessing import ArtVideos

```
`output:`
```
output
```

A complete example is available in `example.py`.

_References:_
- *https://www.tensorflow.org/tutorials/generative/cvae*