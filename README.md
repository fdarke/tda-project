# tda-project

### How to start
1) Run this project
2) Setup virtual environment through following steps (run the following commands in Terminal)
python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
pip install --upgrade ipykernel
python -m ipykernel install --user --name MyEnv --display-name "TDA Env"
deactivate && exit
3) Now you have the virtual environment installed. Activate it again with source .venv/bin/activate
