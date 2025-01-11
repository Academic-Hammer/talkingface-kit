FROM python:3.10
ADD . /code
WORKDIR /code
ENV PATH=$PATH:/code
ENV PYTHONPATH=$PATH:/code
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y --fix-missing
RUN apt-get install libgl1 --fix-missing
RUN pip install opencv-python-headless
RUN pip install -r requirements.txt
CMD ["python","test.py","--mode","niqe","--path=/code/input.png"]