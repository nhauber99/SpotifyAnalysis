import base64

import time

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


def request_access_token():
    with open('spotify_api_key.txt', 'r') as f:
        client_id = f.readline().strip()
        client_secret = f.readline().strip()
    auth_url = 'https://accounts.spotify.com/api/token'
    auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    headers = {'Authorization': f'Basic {auth_header}'}
    data = {'grant_type': 'client_credentials'}
    response = requests.post(auth_url, headers=headers, data=data)
    response_data = response.json()
    return response_data['access_token']


def request_track_metadata(data, access_token, file_path):
    track_ids = data["track_id"].unique()

    metadata_list = []
    endpoint = 'https://api.spotify.com/v1/tracks'
    headers = {'Authorization': f'Bearer {access_token}'}

    for i in tqdm(range(0, len(track_ids), 50), 'Requesting Track Data'):
        params = {'ids': ','.join(track_ids[i:i + 50])}
        response = requests.get(endpoint, headers=headers, params=params)

        if response.status_code == 200:
            metadata_list.extend(response.json()['tracks'])
        else:
            print(f"Error: {response.status_code}, {response.text}")

        time.sleep(0.3)  # limit requests

    metadata = pd.json_normalize(metadata_list)

    def extract_artist_ids(row):
        artist_ids = [artist['id'] for artist in row[:5]] if isinstance(row, list) else [None] * 5
        return artist_ids + [None] * (5 - len(artist_ids))

    metadata[['artist0', 'artist1', 'artist2', 'artist3', 'artist4']] = metadata['artists'].apply(extract_artist_ids).tolist()
    metadata = metadata[['id', 'name', 'duration_ms', 'popularity', 'is_local', 'album.id', 'artist0', 'artist1', 'artist2', 'artist3', 'artist4']]
    metadata.to_csv(file_path, index=False)


def request_album_metadata(track_data, access_token, file_path):
    album_data = pd.read_json("Data/album_metadata2.json")
    album_ids = pd.unique(track_data['album.id'])
    album_ids = np.setdiff1d(album_ids, np.array(album_data['id']))
    metadata_list = []
    endpoint = 'https://api.spotify.com/v1/albums'
    headers = {'Authorization': f'Bearer {access_token}'}

    for i in tqdm(range(0, len(album_ids), 20), 'Requesting Album Data'):
        params = {'ids': ','.join(album_ids[i:i + 20])}
        response = requests.get(endpoint, headers=headers, params=params)

        if response.status_code == 200:
            metadata_list.extend(response.json()['albums'])
        else:
            print(f"Error: {response.status_code}, {response.text}")

        time.sleep(0.3)  # limit requests

    metadata = pd.json_normalize(metadata_list)
    metadata = metadata[['id', 'name', 'popularity', 'release_date', 'genres']]
    metadata = pd.concat([metadata, album_data])
    metadata.to_json(file_path)


def request_artist_metadata(track_data, access_token, file_path):
    artist_uris = pd.unique(track_data[['artist0', 'artist1', 'artist2', 'artist3', 'artist4']].stack())
    metadata_list = []
    endpoint = 'https://api.spotify.com/v1/artists'
    headers = {'Authorization': f'Bearer {access_token}'}

    for i in tqdm(range(0, len(artist_uris), 50), 'Requesting Artist Data'):
        params = {'ids': ','.join(artist_uris[i:i + 50])}
        response = requests.get(endpoint, headers=headers, params=params)

        if response.status_code == 200:
            metadata_list.extend(response.json()['artists'])
        else:
            print(f"Error: {response.status_code}, {response.text}")

        time.sleep(0.3)  # limit requests

    metadata = pd.json_normalize(metadata_list)
    metadata = metadata[['id', 'name', 'popularity', 'genres']]
    metadata.to_json(file_path)
