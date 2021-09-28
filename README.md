# Education drop

For run this project you must go to the root directory and execute the next commands:

1. dvc remote modify storage --local gdrive_service_account_json_file_path $(pwd)/education-drop-d884963a3b3b.json
2. dvc fetch
3. dvc pull
4. pip install -r requirements.txt
5. export PYTHONPATH=$(pwd)/Code
6. python init.py -rd $(pwd)
7. dvc repro