FROM public.ecr.aws/lambda/python:3.8

# A librosa dependency uses numba to just-in-time complie some code
# and this needs to be saved to a cache dir
ENV NUMBA_CACHE_DIR=/tmp

RUN pip install librosa
RUN pip install numpy
RUN pip install s3fs
RUN pip install boto3
RUN pip install requests
RUN pip install tensorflow
RUN pip install tensorflow-hub
RUN pip install gitpython
RUN pip install pandas
RUN pip install matplotlib
RUN pip install scikit-learn

# Librosa dependency
RUN yum install libsndfile -y

COPY *.py   ./
COPY * ./

ADD phoneme_convolutional_softmax phoneme_convolutional_softmax
ADD speech_embedding_1 speech_embedding_1
ADD anomaly_voice_convolutional_softmax anomaly_voice_convolutional_softmax

EXPOSE 9000
EXPOSE 8888

CMD ["app.handler"]
