#!/bin/bash
PYTHONPATH=/opt/python:$PYTHONPATH \
    exec python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
