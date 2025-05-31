#!/bin/bash

apt-get install zip unzip
wget http://mattmahoney.net/dc/enwik9.zip
unzip enwik9.zip
rm enwik9.zip
pip install tiktoken
python parse.py
pip install wandb
