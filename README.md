## Set up environment
For convenience, [Dockerfile](Docker/Dockerfile) is provided under `Docker`. 
You can use as follows:

```bash
# Build docker image
docker build -t [image tag] --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) .

# Run docker container
docker run --gpus all -it --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v [path to the top of this git repo]:/enkg -v [path to data]:/data [image tag]
```
Breakdown of the `docker run` command:
- `--gpus all -it --rm`: With all GPUs enabled, run an interactive session, and automatically remove the container when it exits.
- `--ipc=host --ulimit memlock=-1 --ulimit stack=67108864`: Flags recommended by NVIDIA. Unlock the resource constraint.
- `-v [path to the top of this repo]:/enkg -v [path to data]:/data`: Mount the current dir to `/enkg`. Mount the data directory to `/data`.

## Pretrain

```bash
accelerate launch --multi_gpu train.py -cn [specify config_name]
```

## Inference


By default, `configs/config.yaml` will be loaded for inference. You can override the config value by
```bash
python3 main.py problem=[inverse problem config name] algorithm=[algorithm config name] pretrain=[pretrained model config name]
```
You can also specify the main config file with `-cn` flag. 
```bash
python3 main.py -cn [specify config_name]
```

The structure of the inference config is explained below. 
| Key       | Description                                                       |
|-----------|----------------------------------------------------------------------------------|
| problem   | string, config name for inverse problem                                          |
| algorithm | string, config name for algorithm                                                |
| pretrain  | string, config name for pretrained model (optional if loading from .pkl)         |
| tf32      | True or False, enable TF32 mode for improved speed on Ampere generation and later|


## Pretrained models

| Problem | Test set | Pretrained diffusion model| Training config |
|---------|-------------|---------------------------| ------------- |
| 2D Navier-Stokes | [navier-stokes-test](https://enkg.s3.us-east-2.amazonaws.com/navier-stokes-test.zip) | [ns-5m.pt](https://enkg.s3.us-east-2.amazonaws.com/ns-5m.pt) | [navier-stokes](configs/pretrain/navier-stokes.yaml) |
|FFHQ256 | [FFHQ256](https://enkg.s3.us-east-2.amazonaws.com/ffhq256-val.zip) | [checkpoint from DPS](https://enkg.s3.us-east-2.amazonaws.com/ffhq256.pt) | [ffhq256](configs/pretrain/ffhq256.yaml)|
|Blackhole | Private | Private | Private |
