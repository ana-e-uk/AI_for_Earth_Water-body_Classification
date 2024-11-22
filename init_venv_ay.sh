rm -rf ai_venv_ay

python3.11 -m venv ai_venv_ay

source ai_venv_ay/bin/activate

pip install --upgrade pip

pip install --upgrade setuptools

pip install wheel

pip install -r requirements.txt