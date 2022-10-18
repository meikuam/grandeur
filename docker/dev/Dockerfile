# FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04
# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
SHELL ["/bin/bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt install -y --no-install-recommends \
    libsm6 libxext6 libxrender-dev \
    sudo \
    openssh-server \
    vim \
    git \
    curl wget supervisor\
    libsqlite3-dev libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev  llvm libncurses5-dev libncursesw5-dev \
    libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

ARG UID

EXPOSE 22
EXPOSE 8888
EXPOSE 8080
EXPOSE 8081


ENV HOME_DIR /home/user
ENV DATA_DIR /data

RUN useradd -d $HOME_DIR -ms /bin/bash  -u $UID -G sudo user
RUN echo user:user | chpasswd


## python
RUN mkdir /opt/anaconda && \
    chown -R user:user /opt/anaconda &&\
    chmod 770 -R /opt/anaconda
USER user
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O  /opt/anaconda/anaconda.sh &&\
    chmod +x /opt/anaconda/anaconda.sh &&\
    bash /opt/anaconda/anaconda.sh -b -p /opt/anaconda/anaconda


USER root
RUN mkdir /var/run/sshd
RUN sed -i 's/#PermitRootLogin /PermitRootLogin yes #/' /etc/ssh/sshd_config
RUN sed -i 's/#PasswordAuthentication/PasswordAuthentication yes#/' /etc/ssh/sshd_config
#SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV PATH /opt/anaconda/anaconda/bin:/usr/local/cuda/bin:$PATH
RUN echo "export PATH=$PATH" >> /etc/profile
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> /etc/profile
RUN echo "export VISIBLE=now" >> /etc/profile
RUN echo "export NOTVISIBLE=$NOTVISIBLE" >> /etc/profile
RUN echo "export LC_ALL=C.UTF-8" >> /etc/profile
RUN echo "export LANG=C.UTF-8" >> /etc/profile




USER user
RUN eval "$(/opt/anaconda/anaconda/bin/conda shell.bash hook)"
WORKDIR /tmp/
COPY environment.yml /tmp/environment.yml
RUN conda env create -f environment.yml && \
    conda init bash && \
    conda clean -a
RUN echo "conda activate grandeur" >> ~/.bashrc


COPY supervisord /etc/supervisor/conf.d/

WORKDIR $HOME_DIR
RUN mkdir $HOME_DIR/grandeur
USER root
CMD ["supervisord", "-n"]