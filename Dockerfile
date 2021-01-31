FROM pytorch/pytorch:latest
# WORKDIR /project
RUN apt update

########## opencv ############
RUN apt install -y libsm6 libxext6 libxrender-dev libglib2.0-0 cmake
RUN CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
RUN pip install opencv-python
RUN pip install jupyter,pandas,neptune-client
RUN pip install matplotlib
RUN pip install torchgeometry
RUN pip install torchvision,pytorch-lightning
RUN pip install sacred
RUN apt install -y openssh-server
RUN mkdir ~/.ssh && touch ~/.ssh/authorized_keys && echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDdILBbrDrDK04MFooVnPKV6MYI7+uWIRZ1IJCpyml23k2SZCQLPJQdxSTDAjho/5xmctI+OUA55LI0z5ujRh8ZlytHZ1q2vpLma5eNlzXKLZ1aEquJpoXx+cLFMBfNZrhrqQBCbJlS9EK4+syBVltt5mFwXZtz4PcTV2OBjN886eUb7w/ueoNuzBEw3n5DWqGdFFdffhBOYbYN66o3JBAzfE7rOVeSxUF/Xp/I0X9MZIt47oGL7O1/KdTU8M0/UoNvKAEnWUKyKyJRiVxII0gcNV4AnkUGosRLMqmdgtduq/FWgtriK9IKgQJniiH3BN2B2IDL2/SXCzQKGMdz2cQb" >> ~/.ssh/authorized_keys
RUN chmod o-rwx,g-rwx ~/.ssh/authorized_keys
ENV SHELL /bin/bash
