aws ecr get-login-password --region eu-west-2 | docker login --username AWS --password-stdin 708553369289.dkr.ecr.eu-west-2.amazonaws.com
docker build -t speech-first-ai-docker-base-testing .
docker tag speech-first-ai-docker-base-testing:latest 708553369289.dkr.ecr.eu-west-2.amazonaws.com/speech-first-ai-docker-base-testing:latest
docker push 708553369289.dkr.ecr.eu-west-2.amazonaws.com/speech-first-ai-docker-base-testing:latest

