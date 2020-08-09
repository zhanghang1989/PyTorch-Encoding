FROM nvcr.io/nvidia/pytorch:20.06-py3

# Set working directory	# Set working directory
WORKDIR /workspace

#ENV PYTHONPATH "${PYTHONPATH}:/workspace"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-tk python-pip git tmux htop tree

RUN python -m pip install --upgrade pip
#RUN python -m pip install torch==1.4.0
#RUN python -m pip install torchvision==0.5.0
RUN python -m pip install pycocotools==2.0.0

#RUN chmod a+rwx -R /opt/conda/

COPY ./setup.py .
COPY ./encoding ./encoding

ENV FORCE_CUDA="1"
RUN python setup.py develop

COPY ./experiments ./experiments
