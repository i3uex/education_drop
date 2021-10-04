pip install -r requirements.txt
dvc remote modify storage --local gdrive_service_account_json_file_path "$(pwd)"/education-drop-d884963a3b3b.json
dvc fetch
dvc pull
PYTHONPATH="$(pwd)"/Code
export PYTHONPATH
python init.py -rd "$(pwd)"
dvc repro