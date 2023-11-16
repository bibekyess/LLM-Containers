FROM nvidia/cuda:11.8.0-base-ubuntu20.04

RUN set -xe \
    && apt-get update \
    && apt-get install -y curl python3-pip \
    && pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# Add non-root user
RUN groupadd -r user && useradd -r -g user user

# Download the model into user home directory, using the user
WORKDIR /home/user
RUN chown -R user /home/user/
USER user
RUN python3 -c "from transformers import AutoTokenizer, AutoModelForCausalLM; tokenizer = AutoTokenizer.from_pretrained('EleutherAI/polyglot-ko-12.8b'); model = AutoModelForCausalLM.from_pretrained('EleutherAI/polyglot-ko-12.8b', device_map='auto')"

COPY app /home/user/app
WORKDIR /home/user/app

# Expose port
EXPOSE 5000

ENV CUDA_VISIBLE_DEVICES=0,1

ENTRYPOINT [ "python3", "app.py" ]
