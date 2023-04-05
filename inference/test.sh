docker run -p 9000:8080 speech-first-ai-docker-base
curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{}'
