# Agent Environments - Webshop

## Setup

``` sh
conda env create -n agentenv-webshop -f environment.yml
conda activate agentenv-webshop
bash ./setup.sh
pip install "numpy>=1.19.0,<1.25.0"
pip install "spacy>=3.5.0,<3.6.0"
pip install "pydantic>=1.10.0,<2.0.0"
python -m spacy download en_core_web_sm
```

## Launch

``` sh
webshop --host 0.0.0.0 --port 36001
```
