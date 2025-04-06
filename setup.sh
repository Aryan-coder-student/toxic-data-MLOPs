#!/bin/bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
dvc init
dvc remote add -d myremote https://dagshub.com/<username>/toxic-comment-classifier.dvc
dvc remote modify myremote auth basic
dvc remote modify myremote user <dagshub-username>
dvc remote modify myremote password <dagshub-token>