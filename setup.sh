#!/bin/bash
source classifier/bin/activate || (python3 -m venv classifier && source classifier/bin/activate)
python3 -m pip install --upgrade -r requirements.txt
