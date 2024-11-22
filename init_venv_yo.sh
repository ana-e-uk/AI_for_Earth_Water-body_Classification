rm -rf ai_venv_yo

python3.11 -m venv ai_venv_yo

source ai_venv_yo/bin/activate

pip install --upgrade pip

pip install --upgrade setuptools

pip install wheel

pip install -r requirements.txt