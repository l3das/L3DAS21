import requests
import argparse
import os
from tqdm import tqdm

'''
Download our pre-trained baseline models for task 1 and task2, separately.
Command line arguments define which task to download and where to put the checkpoint file.
'''


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024) as bar:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    bar.update(CHUNK_SIZE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--task', type=int)
    args = parser.parse_args()

    if args.task == 1:
        file_id = '1gZbyVTXvzPKe2j_08XGSUkdDwU6pE8nK'
    elif args.task == 2:
        file_id = '1gZcTnXlxjYEKgLRMiTNy7JAGltLe_aj4'

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    output_path = os.path.join(args.output_path, 'checkpoint')

    download_file_from_google_drive(file_id, output_path)

    print ('Pre-trained model for Task ' + str(args.task) + ' successfully downloaded')
