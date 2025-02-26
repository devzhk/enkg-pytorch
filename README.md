## Ensemble Kalman Diffusion Guidance - Offical Pytorch Implementation
[Ensemble Kalman Diffusion Guidance: A Derivative-free Method for Inverse Problems](https://arxiv.org/abs/2409.20175)

Hongkai Zheng, Wenda Chu*, Austin Wang*, Nikola Kovachki, Ricardo Baptista, Yisong Yue

Abstract: 
*When solving inverse problems, one increasingly popular approach is to use pre-trained diffusion models as plug-and-play priors. This framework can accommodate different forward models without re-training while preserving the generative capability of diffusion models.  Despite their success in many imaging inverse problems, most existing methods rely on privileged information such as derivative, pseudo-inverse, or full knowledge about the forward model. This reliance poses a substantial limitation that restricts their use in a wide range of problems where such information is unavailable, such as in many scientific applications. We propose Ensemble Kalman Diffusion Guidance (EnKG), a derivative-free approach that can solve inverse problems by only accessing forward model evaluations and a pre-trained diffusion model prior. We study the empirical effectiveness of EnKG across various inverse problems, including scientific settings such as inferring fluid flows and astronomical objects, which are highly non-linear inverse problems that often only permit black-box access to the forward model.*


## Environment requirements
- We recommend Linux with 64-bit Python 3.11.5 and Pytorch 2.2.2. See https://pytorch.org for PyTorch install instructions.
- `torch, accelerate, hydra-core, ehtim, ehtplot, piq, wandb, pillow, lmbd, omegaconf` are the main Python libraries required. Environment file is provided in `env.yml`. 
- We also provide a [Dockerfile](Docker/Dockerfile) under `Docker`. You can use as follows:

```bash
# Build docker image
docker build -t [image tag] --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) .

# Run docker container
docker run --gpus all -it --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v [path to the top of this git repo]:/enkg -v [path to data]:/data [image tag]
```
Breakdown of the `docker run` command:
- `--gpus all -it --rm`: Enable all available GPUs, starts an interactive session, and automatically remove the container upon exit.
- `--ipc=host --ulimit memlock=-1 --ulimit stack=67108864`: Recommended NVIDIA flags to unlock resource constraints.
- `-v [path to the top of this repo]:/enkg -v [path to data]:/data`: Mount the current dir to `/enkg`. Mount the data directory to `/data`.

## Data and pre-trained models
Data and pre-trained models can be found in the Github release page. By default, the data should be placed in `../data` and the pre-trained models should be placed in `checkpoints` directory. You can also specify the data and checkpoint path in the config file.

## Inference

By default, `configs/config.yaml` will be loaded for inference. You can override the config value by
```bash
python3 main.py problem=[inverse problem config name] algorithm=[algorithm config name] pretrain=[pretrained model config name]
```
The structure of the inference config is explained below. 
| Key       | Description                                                                      |
|-----------|----------------------------------------------------------------------------------|
| `problem`   | Name of the inverse problem configuration. (See `configs/problem`)             |
| `algorithm` | Name of the algorithm configuration. (See `configs/algorithm`)                 |
| `pretrain`  | Name of the pre-trained model configuration. (see `configs/pretrain`)          |
| `tf32`      | (bool) Enables TF32 mode for improved performance on Ampere+ GPUs.             |
| `compile`   | (bool) Enable `torch.compile` (recommended for ensemble methods).              |
| `seed`      | (int) Random seed.                                                             |
| `inference` | (bool) If False, skip inference and only run evaluation.                       |
| `exp_name`  | (string) Sets the experiment name for logging and saving results.              |
| `wandb`     | (bool) Enables logging to Weights & Biases (WandB).                            |

We provide sample scripts to run experiments in `scripts`. 
- `scripts/navier-stokes.sh` contains commands to run different algorithms on the inverse problem of the Navier-Stokes equation. (Takes ~2 hours on an A100 GPU as the numerical solver takes time to run) 
- `scripts/ffhq.sh` contains commands to run different algorithms on image restoration tasks. In general, image restoration tasks here are not the best use case for derivative-free methods. For example, EnKG is inefficient when the forward model is much faster than diffusion model evaluation. This serves as a proof-of-concept example. 
- `scripts/blackhole.sh` contains commands to run inference on black hole imaging tasks. (These experiments run efficiently on an A100 GPU.)


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Citation
```bibtex   
@article{zheng2024ensemble,
  title={Ensemble kalman diffusion guidance: A derivative-free method for inverse problems},
  author={Zheng, Hongkai and Chu, Wenda and Wang, Austin and Kovachki, Nikola and Baptista, Ricardo and Yue, Yisong},
  journal={arXiv preprint arXiv:2409.20175},
  year={2024}
}
```

## Acknowledgements
- The pre-trained model weights for FFHQ256 is converted from [DPS's repository](https://github.com/DPS2022/diffusion-posterior-sampling). We thank the authors for releasing their pre-trained model. 
- We thank Ben Prather, Abhishek Joshi, Vedant Dhruv, C.K. Chan, and Charles Gammie for
the synthetic blackhole images [GRMHD Dataset](https://iopscience.iop.org/article/10.3847/1538-4365/ac582e) used here, generated under NSF grant AST 20-34306. 