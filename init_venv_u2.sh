rm -rf ai_venv_u2

python3.11 -m venv ai_venv_u2

source ai_venv_u2/bin/activate

pip install --upgrade pip

pip install --upgrade setuptools

pip install wheel

pip install -r requirements.txt

pip install seaborn