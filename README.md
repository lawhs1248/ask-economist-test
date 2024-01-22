# Generative Business Intelligence

## Overview
POC to demonstrate integration of LLM to ask questions on database without touching any datasets by ChatGPT

## Requirements
Python 3.5.2+

## Usage
To run the server, please execute the following from the root directory:

```
pip3 install -r requirements.txt
streamlit run main.py
```


## Running with Docker

To run the server on a Docker container, please execute the following from the root directory:

```bash
# building the image
docker build -t gen_bi .

# starting up a container
docker run -p 8501:8501 --env OPENAI_TOKEN="" --env OPENAI_ENDPOINT="" gen_bi 
```


```bash
#For pure benchmarking purpose
export OPENAI_ENDPOINT=""
export OPENAI_ENDPOINT=""
python benchmark.py
```

## TODO

- Support for custom database URI through environment variables
- Better environment variable support for OPENAI
- Script to auto-generate schema.py from database connection