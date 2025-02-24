FROM nvcr.io/nvidia/pytorch:24.02-py3

ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

RUN pip install omegaconf wandb lmdb
RUN pip install accelerate
RUN pip install monai piq hydra-core
RUN pip install ehtim

ENV PATH="$PATH:/home/user/.local/bin"