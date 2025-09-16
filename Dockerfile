# Dockerfile (for SageMaker inference)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# install system deps
RUN apt-get update && apt-get install -y git build-essential

# install requirements
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# copy model server code
WORKDIR /opt/program
COPY ./model_server /opt/program

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

# port for inference
EXPOSE 8080

ENTRYPOINT ["python", "server.py"]
