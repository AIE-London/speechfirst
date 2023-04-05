# SpeechFirst Inference

This area of our repository contains the tools necessary to perform inference against our models. You must have completed the data collection and modelling activities first - we don't provide pretrained weights.

### Dependencies

Install any python dependencies as per the requirements.txt and Dockerfile.

#### Model Weights

In order to process the data and train the phoneme model, we require two pretrained model's weights to be downloaded.

##### Speech Embedding

Download the Speech Embedding model from TFHub (https://tfhub.dev/google/speech_embedding/1)

The "saved_model.pb" should be placed under the `inference/speech_embedding_1` directory and the variables files in a subdirectory of this named "variables".

### Trained Models

#### Consonants Classifier

Save the weights and ensure you place them under the `inference/anomaly_voice_convolutional_softmax` directory and the variables files in a subdirectory of this named "variables".

#### Phoneme Model

Save the weights and ensure you place them under the `inference/phoneme_convolutional_softmax` directory and the variables files in a subdirectory of this named "variables".

You should be ready for inference.

## Running and Invocation

The inference server is setup to be executed on an AWS lambda via docker.

### Deployment and Inference

#### Build docker image
Simply build using docker - by executing a docker build in the inference directory.

```
docker build -t $ECR_REPOSITORY .
```

#### Uploading Docker image to ECR 
Now we need to deploy this image to ECR in AWS by pushing it to our registry.

```
docker tag $ECR_REPOSITORY:latest $ECR_REGISTRY/$ECR_REPOSITORY:latest
docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest
```

#### Setting up a lambda
You need to setup a python-based lambda for asynchronous invocation.

Ensure you wire the lambda to two Amazon SQS destinations.

1. One queue should be an "on success" queue
2. Second queue should be an "on failure" queue

Ensure you note these queue names and update the relevant information in `inference/test-inference/test.py`

#### Deploying to an existing lambda 
Given an existing lambda in AWS - let's update that function/lambda to use our latest image.

With the AWS CLI:

```
aws lambda update-function-code --function-name $LAMBDA_NAME --image-uri $ECR_REGISTRY/$ECR_REPOSITORY:latest"
```

#### Invoking the lambda for inference

Pre-requisites:
1. Record your audio to be analysed as a WAV file
2. Upload to AWS S3 - in a location the lambda has read access
3. Install requirements.txt for `test-inference` directory
4. Populate `inference/test-inference/test.py` with the correct success and failure queue names from the above steps
5. Populate a test-case.csv from the `test-case-example.csv` template

Finally, execute the test.py file in test-inference using python.

If all goes well, you should see entries enter the success queue and the tests will pass.
