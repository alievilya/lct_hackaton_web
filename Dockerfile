# start by pulling the python image
#FROM cuda:11.4.2-base-ubuntu20.04
FROM nvidia/cuda:11.6.0-devel-ubuntu20.04

WORKDIR /root

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND=noninteractive

# copy the requirements file into the image
COPY ./requirements.txt /flask_app/requirements.txt
WORKDIR /flask_app


# install the dependencies and packages in the requirements file
#RUN pip install -r requirements.txt

# copy every content from the local file to the image
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install python3.9 -y
RUN apt-get install python3-pip -y
RUN python3.9 -m pip install --upgrade pip
RUN python3.9 -m pip install urllib3
#RUN python3.9 -m pip install torch==1.10.2+cu111 torchvision==0.11.3+cu111 torchaudio==0.10.2 -f https://download.pytorch.org/whl/torch_stable.html
#RUN python3.9 -m pip install torch==1.10.2+cu102 torchvision==0.11.3+cu102 torchaudio==0.10.2+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
RUN python3.9 -m pip install -r requirements.txt

COPY /flask_app /flask_app
# switch working directory

# configure the container to run in an executed manner
#ENTRYPOINT [ "python" ]

CMD ["python3.9", "-m", "flask_app"]

#  docker build -t lct_otso_flask .
#  docker run --gpus=all --restart=always --name lct_hack_otso -ti -p 5004:5005 lct_otso_flask
#--runtime=gpu