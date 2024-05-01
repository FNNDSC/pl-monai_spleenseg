# Python version can be changed, e.g.
# FROM python:3.8
# FROM ghcr.io/mamba-org/micromamba:1.5.1-focal-cuda-11.3.1
FROM docker.io/python:3.12.1-slim-bookworm

LABEL org.opencontainers.image.authors="FNNDSC <dev@babyMRI.org>" \
      org.opencontainers.image.title="Spleen 3D image segmentation training (MONAI)" \
      org.opencontainers.image.description="A ChRIS based off project MONAI's spleen segmenation notebook (training only)"

ARG SRCDIR=/usr/local/src/pl-monai_spleenseg
WORKDIR ${SRCDIR}

COPY requirements.txt .
# RUN --mount=type=cache,sharing=private,target=/root/.cache/pip pip install -r requirements.txt
RUN pip install scikit-build
RUN pip install -r requirements.txt

COPY . .
ARG extras_require=none
# RUN pip install ".[${extras_require}]" \
#     && cd / && rm -rf ${SRCDIR}
WORKDIR ${SRCDIR}

CMD ["spleenseg"]
