import os

import pandas as pd

from MetadataRequests import request_access_token, request_track_metadata, request_artist_metadata, request_album_metadata

enable_metadata = False  # still experimental and I'm waiting on my api rate limit to end until continuing this
data_dir = "Data"
out_dir = "Results"
data_filename = "audio_data.csv"
track_data_filename = "track_metadata.csv"
artist_data_filename = "artist_metadata.json"
album_data_filename = "album_metadata.json"


def process_data():
    data = []
    for file in os.listdir(data_dir):
        if file.startswith("Streaming_History_Audio_") and file.endswith(".json"):
            data.append(pd.read_json(os.path.join(data_dir, file)))
    data = pd.concat(data, ignore_index=True)
    data = data[['ts', 'ms_played', 'master_metadata_track_name', 'master_metadata_album_artist_name', 'spotify_track_uri', 'reason_start', 'reason_end']]
    data = data.rename(columns={'master_metadata_track_name': 'track', 'master_metadata_album_artist_name': 'artist', 'spotify_track_uri': 'uri'})
    data = data[data['uri'].apply(lambda x: isinstance(x, str) and x.startswith('spotify:track'))]
    data.to_csv(os.path.join(data_dir, data_filename))
    return data


def read_data():
    data = pd.read_csv(os.path.join(data_dir, data_filename))
    data['track_id'] = data['uri'].str[14:]
    data['ts'] = pd.to_datetime(data['ts'])
    data['hours_played'] = data['ms_played'] / 1000 / 3600
    data['sec_played'] = data['ms_played'] / 1000
    data['plays'] = 1
    data.sort_values(by='ts', ascending=True, inplace=True)
    return data


def join_data(data, track_data, artist_data, album_data):
    data = pd.merge(data, track_data, how='left', left_on='track_id', right_on='id')
    data.rename(columns={'name': 'track.name', 'popularity': 'track.popularity'}, inplace=True)
    data = pd.merge(data, artist_data, how='left', left_on='artist0', right_on='id')
    data.rename(columns={'name': 'artist.name', 'popularity': 'artist.popularity', 'genres': 'artist.genres'}, inplace=True)
    data = pd.merge(data, album_data, how='left', left_on='album.id', right_on='id')
    data.rename(columns={'name': 'album.name', 'popularity': 'album.popularity', 'genres': 'album.genres'}, inplace=True)
    data.drop(columns=['id_x', 'id_y', 'id', 'artist0', 'artist1', 'artist2', 'artist3', 'artist4'], inplace=True)
    data.loc[data["duration_ms"] < 1000 * 60, "duration_ms"] = 1000 * 60 * 4
    data["duration_ms"].clip(1000 * 60, 1000 * 3600, inplace=True)
    return data


def save_artists(data, min_hours_played=1):
    x = data.groupby(["artist"])[["hours_played", "plays"]].sum().sort_values(by="hours_played", ascending=False)
    x[x['hours_played'] > min_hours_played].to_csv(os.path.join(out_dir, "artists.csv"))


def save_tracks_by_time(data, min_hours_played=0.1):
    x = data.groupby(["artist", "track", "uri"])[["hours_played", "plays"]].sum().sort_values(by="hours_played", ascending=False)
    x[x['hours_played'] > min_hours_played].to_csv(os.path.join(out_dir, "tracks_by_time.csv"))


def save_tracks_by_plays(data, min_playes=20, min_seconds_per_play=30):
    x = data[data["hours_played"] > min_seconds_per_play / 3600]
    x = x.groupby(["artist", "track", "uri"])[["hours_played", "plays"]].sum().sort_values(by="plays", ascending=False)
    x[x['plays'] > min_playes].to_csv(os.path.join(out_dir, "tracks_by_plays.csv"))


def save_least_skipped(data, min_plays=10):
    x = data.groupby(["artist", "track", "uri"])[["hours_played", "plays"]].agg({
        'hours_played': ['sum', 'max'],
        'plays': 'sum'
    })
    x = x[x["plays"]["sum"] > min_plays]
    x["assumed_song_length_minute"] = x["hours_played"]["max"] * 60
    x["rel_avg_listen"] = x["hours_played"]["sum"] / x["plays"]["sum"] / x["hours_played"]["max"]
    x.sort_values(by="rel_avg_listen", ascending=False, inplace=True)
    x.to_csv(os.path.join(out_dir, "least_skipped_tracks.csv"))


def post_process(data):
    data['rel_time'] = (data["ms_played"] / data["duration_ms"]).clip(0, 1)
    data['rel_time_3m'] = data["sec_played"] / 180

    trackdone_start_mask = data['reason_start'] == 'trackdone'
    group = (~trackdone_start_mask).cumsum()
    data['consecutive_trackdone'] = trackdone_start_mask.groupby(group).cumsum()

    trackdone_mask = data['reason_end'] == 'trackdone'
    fwdbtn_mask = data['reason_end'] == 'fwdbtn'
    clickrow_mask = data['reason_start'] == 'clickrow'
    backbtn_mask = data['reason_start'] == 'backbtn'

    data['score'] = 0.2
    data['score'] += data['rel_time'] - 0.5
    data['score'] += 0.1 * (data['rel_time_3m'] - 0.5)
    data.loc[trackdone_mask, 'score'] /= 0.5 * (data.loc[trackdone_mask, 'consecutive_trackdone'] + 4).pow(0.5)
    data.loc[fwdbtn_mask, 'score'] -= 0.3 * data.loc[fwdbtn_mask, 'consecutive_trackdone'].clip(0, 5)
    data.loc[clickrow_mask, 'score'] += 0.4 * data.loc[clickrow_mask, 'rel_time']
    data.loc[backbtn_mask, 'score'] += 0.4 * data.loc[backbtn_mask, 'rel_time']

    return data


def save_by_score(data, min_plays=5):
    x = data[['artist', 'track', 'uri', 'hours_played', 'plays', 'score']]
    x = x.groupby(["artist", "track", "uri"])[["hours_played", "plays", "score"]].sum().sort_values(by="score", ascending=False)
    x[x['plays'] > min_plays].to_csv(os.path.join(out_dir, "tracks_by_score.csv"))


def save_by_mean_score(data, min_plays=5):
    x = data[['artist', 'track', 'uri', 'hours_played', 'plays', 'score', 'duration_ms']]
    x = x.groupby(["artist", "track", "uri", "duration_ms"])[["hours_played", "plays", "score"]].agg({
        'hours_played': 'sum',
        'plays': 'sum',
        'score': 'mean'
    })
    x.sort_values(by="score", ascending=False, inplace=True)
    x[x['plays'] > min_plays].to_csv(os.path.join(out_dir, "tracks_by_mean_score.csv"))


def main():
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(os.path.join(data_dir, data_filename)):
        process_data()
    data = read_data()
    print(f'total hours: {data["hours_played"].sum():.0f}')
    print(f'plays: {data["plays"].sum()}')
    print(f'unique tracks: {len(data["uri"].unique())}')

    # basic statistics
    save_artists(data)
    save_tracks_by_time(data)
    save_tracks_by_plays(data)
    save_least_skipped(data)

    # metadata
    if enable_metadata:
        track_data_path = os.path.join(data_dir, track_data_filename)
        album_data_path = os.path.join(data_dir, album_data_filename)
        artist_data_path = os.path.join(data_dir, artist_data_filename)
        if not os.path.exists(track_data_path) or not os.path.exists(album_data_path) or not os.path.exists(artist_data_path):
            access_token = request_access_token()
            if not os.path.exists(track_data_path):
                request_track_metadata(data, access_token, track_data_path)
            track_data = pd.read_csv(track_data_path)
            if not os.path.exists(album_data_path):
                request_album_metadata(track_data, access_token, album_data_path)
            if not os.path.exists(artist_data_path):
                request_artist_metadata(track_data, access_token, artist_data_path)
        else:
            track_data = pd.read_csv(track_data_path)
        album_data = pd.read_json(album_data_path)
        artist_data = pd.read_json(artist_data_path)

        data = join_data(data, track_data, artist_data, album_data)
    else:
        data["duration_ms"] = data.groupby('uri')['ms_played'].transform('max').clip(1000 * 60 * 2, 1000 * 60 * 5)
    post_process(data)
    save_by_score(data)
    save_by_mean_score(data)


if __name__ == "__main__":
    main()
