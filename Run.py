import os

import pandas as pd

data_dir = "Data"
out_dir = "Results"
data_filename = "audio_data.csv"


def process_data():
    data = []
    for file in os.listdir(data_dir):
        if file.endswith(".json") and "Audio" in file:
            data.append(pd.read_json(os.path.join(data_dir, file)))
    data = pd.concat(data, ignore_index=True)
    data = data[['ts', 'ms_played', 'master_metadata_track_name', 'master_metadata_album_artist_name']]
    data['hours_played'] = data['ms_played'] / 1000 / 3600
    data = data.rename(columns={'master_metadata_track_name': 'track', 'master_metadata_album_artist_name': 'artist'})
    data.drop(columns=['ms_played'])
    data.to_csv(os.path.join(data_dir, data_filename))
    return data


def read_data():
    data = pd.read_csv(os.path.join(data_dir, data_filename))
    data['ts'] = pd.to_datetime(data['ts'])
    data['plays'] = 1
    return data


def save_artists(data, min_hours_played=1):
    x = data.groupby(["artist"])[["hours_played", "plays"]].sum().sort_values(by="hours_played", ascending=False)
    x[x['hours_played'] > min_hours_played].to_csv(os.path.join(out_dir, "artists.csv"))


def save_tracks(data, min_hours_played=0.1):
    x = data.groupby(["artist", "track"])[["hours_played", "plays"]].sum().sort_values(by="hours_played", ascending=False)
    x[x['hours_played'] > min_hours_played].to_csv(os.path.join(out_dir, "tracks.csv"))


def save_annoying_tracks(data, min_plays=20):
    x = data.groupby(["artist", "track"])[["hours_played", "plays"]].sum()
    x = x[x["plays"] > min_plays]
    x["seconds_per_play"] = x["hours_played"] / x["plays"] * 3600
    x = x.sort_values(by="seconds_per_play", ascending=True)
    x.to_csv(os.path.join(out_dir, "annoying_tracks.csv"))


def save_least_skipped(data, min_plays=10):
    x = data.groupby(["artist", "track"])[["hours_played", "plays"]].agg({
        'hours_played': ['sum', 'max'],
        'plays': 'sum'
    })
    x = x[x["plays"]["sum"] > min_plays]
    x["assumed_song_length_minute"] = x["hours_played"]["max"] * 60
    x["rel_avg_listen"] = x["hours_played"]["sum"] / x["plays"]["sum"] / x["hours_played"]["max"]
    x = x.sort_values(by="rel_avg_listen", ascending=False)
    x.to_csv(os.path.join(out_dir, "least_skipped_tracks.csv"))


def main():
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(os.path.join(data_dir, data_filename)):
        process_data()
    data = read_data()
    print(f'total hours: {data["hours_played"].sum():.0f}')
    print(f'plays: {data["plays"].sum()}')
    save_artists(data)
    save_tracks(data)
    save_annoying_tracks(data)
    save_least_skipped(data)


if __name__ == "__main__":
    main()
