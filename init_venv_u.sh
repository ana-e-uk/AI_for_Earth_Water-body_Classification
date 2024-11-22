rm -rf ai_venv_u

python3.11 -m venv ai_venv_u

source ai_venv_u/bin/activate

pip install --upgrade pip

pip install --upgrade setuptools

pip install wheel

pip install -r requirements.txt