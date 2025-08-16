#!/bin/bash
source classifier/bin/activate || (python3 -m venv classifier && source classifier/bin/activate)
python3 -m pip install -r requirements.txt
