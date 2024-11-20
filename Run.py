import os

import pandas as pd

from MetadataRequests import request_access_token, request_track_metadata, request_artist_metadata, request_album_metadata

enable_metadata = False  # still experimental and I'm waiting on my api rate limit to end until continuing this
use_cache = True
data_dir = 'Data'
out_dir = 'Results'
data_filename = 'audio_data.csv'
track_data_filename = 'track_metadata.csv'
artist_data_filename = 'artist_metadata.json'
album_data_filename = 'album_metadata.json'


def process_data():
    # initial processing to aggregate all files and write the relevant information to another file
    # this significantly speeds up the script when it's run multiple times (after changing some settings for example)
    data = []
    for file in os.listdir(data_dir):
        if file.startswith('Streaming_History_Audio_') and file.endswith('.json'):
            data.append(pd.read_json(os.path.join(data_dir, file)))
    data = pd.concat(data, ignore_index=True)
    data = data[['ts', 'ms_played', 'master_metadata_track_name', 'master_metadata_album_artist_name', 'spotify_track_uri', 'reason_start', 'reason_end']]
    data = data.rename(columns={'master_metadata_track_name': 'track', 'master_metadata_album_artist_name': 'artist', 'spotify_track_uri': 'uri'})
    data = data[data['uri'].apply(lambda x: isinstance(x, str) and x.startswith('spotify:track'))]
    data.to_csv(os.path.join(data_dir, data_filename))
    return data


def read_data():
    # read cached data file
    data = pd.read_csv(os.path.join(data_dir, data_filename))
    data['track_id'] = data['uri'].str[14:]
    data['ts'] = pd.to_datetime(data['ts'])
    data['hours_played'] = data['ms_played'] / 1000 / 3600
    data['sec_played'] = data['ms_played'] / 1000
    data['plays'] = 1
    data.sort_values(by='ts', ascending=True, inplace=True)

    # if track and artist match -> replace all with the most recent uri
    data['uri'] = data.groupby(['track', 'artist'])['uri'].transform('last')

    return data


def join_data(data, track_data, artist_data, album_data):
    # aggregate listening history and the gathered metadata
    data = pd.merge(data, track_data, how='left', left_on='track_id', right_on='id')
    data.rename(columns={'name': 'track.name', 'popularity': 'track.popularity'}, inplace=True)
    data = pd.merge(data, artist_data, how='left', left_on='artist0', right_on='id')
    data.rename(columns={'name': 'artist.name', 'popularity': 'artist.popularity', 'genres': 'artist.genres'}, inplace=True)
    data = pd.merge(data, album_data, how='left', left_on='album.id', right_on='id')
    data.rename(columns={'name': 'album.name', 'popularity': 'album.popularity', 'genres': 'album.genres'}, inplace=True)
    data.drop(columns=['id_x', 'id_y', 'id', 'artist0', 'artist1', 'artist2', 'artist3', 'artist4'], inplace=True)
    data.loc[data['duration_ms'] < 1000 * 60, 'duration_ms'] = 1000 * 60 * 4
    data['duration_ms'].clip(1000 * 60, 1000 * 3600, inplace=True)
    return data


def save_artists(data, min_hours_played=1):
    # top artists by listening time
    x = data.groupby(['artist'])[['hours_played', 'plays']].sum().sort_values(by='hours_played', ascending=False)
    x[x['hours_played'] >= min_hours_played].to_csv(os.path.join(out_dir, 'artists.csv'))


def save_tracks_by_time(data, min_hours_played=0.1):
    x = data.groupby(['artist', 'track', 'uri'])[['hours_played', 'plays']].sum().sort_values(by='hours_played', ascending=False)
    x[x['hours_played'] >= min_hours_played].to_csv(os.path.join(out_dir, 'tracks_by_time.csv'))


def save_tracks_by_plays(data, min_playes=20, min_seconds_per_play=30):
    x = data[data['hours_played'] > min_seconds_per_play / 3600]
    x = x.groupby(['artist', 'track', 'uri'])[['hours_played', 'plays']].sum().sort_values(by='plays', ascending=False)
    x[x['plays'] >= min_playes].to_csv(os.path.join(out_dir, 'tracks_by_plays.csv'))


def save_tracks_by_single_day_time(data):
    daily_played = data.groupby(['artist', 'track', 'uri', data['ts'].dt.date])['hours_played'].sum().reset_index()
    max_daily_played = daily_played.groupby(['artist', 'track', 'uri'])['hours_played'].max().reset_index()
    x = max_daily_played.sort_values(by='hours_played', ascending=False)
    x.to_csv(os.path.join(out_dir, 'tracks_by_single_day_time.csv'))


def post_process(data):
    data['rel_time'] = (data['ms_played'] / data['duration_ms']).clip(0, 1.5)
    data['rel_time_3m'] = data['sec_played'] / 180

    trackdone_start_mask = data['reason_start'] == 'trackdone'
    group = (~trackdone_start_mask).cumsum()
    data['consecutive_trackdone'] = trackdone_start_mask.groupby(group).cumsum()

    trackdone_mask = data['reason_end'] == 'trackdone'
    fwdbtn_mask = data['reason_end'] == 'fwdbtn'
    clickrow_mask = data['reason_start'] == 'clickrow'
    backbtn_start_mask = data['reason_start'] == 'backbtn'
    backbtn_end_mask = data['reason_end'] == 'backbtn'
    skipped_mask = data['rel_time'] < 0.1

    data['score_weight'] = 1.0

    # base score for each play
    data['score'] = 0.2
    # bonus for longer listening / penalty for skipped songs
    data['score'] += data['rel_time'] - 0.5
    # add some additional score for long listening times
    data['score'] += 0.1 * (data['rel_time_3m'] - 0.5)
    # if listened to a long time without skipping, penalize the first song that is skipped
    data.loc[fwdbtn_mask, 'score'] -= 0.3 * data.loc[fwdbtn_mask, 'consecutive_trackdone'].clip(0, 5)
    # bonus for songs actively clicked on
    data.loc[clickrow_mask, 'score'] += 0.4 * data.loc[clickrow_mask, 'rel_time']
    # bonus for songs reached via the back button
    data.loc[backbtn_start_mask, 'score'] += 0.4 * data.loc[backbtn_start_mask, 'rel_time']

    # reduce impact of score if listened for a long time without skipping (maybe afk/less inclined to skip)
    data.loc[trackdone_mask, 'score_weight'] /= 0.5 * (data.loc[trackdone_mask, 'consecutive_trackdone'] + 4).pow(0.5)
    # skipped via backbutton doesn't tell much
    data.loc[backbtn_end_mask & skipped_mask, 'score_weight'] *= 0.3

    return data


def save_by_score(data, min_plays=5):
    x = data[['artist', 'track', 'uri', 'hours_played', 'plays', 'score', 'score_weight']].copy()
    x['score'] *= x['score_weight']
    x = x.groupby(['artist', 'track', 'uri'])[['hours_played', 'plays', 'score']].sum().sort_values(by='score', ascending=False)
    x[x['plays'] >= min_plays].to_csv(os.path.join(out_dir, 'tracks_by_score.csv'))


def save_by_mean_score(data, min_plays=5):
    x = data[['artist', 'track', 'uri', 'hours_played', 'plays', 'score', 'score_weight', 'duration_ms']].copy()
    x['score'] *= x['score_weight']
    x = x.groupby(['artist', 'track', 'uri', 'duration_ms'])[['hours_played', 'plays', 'score', 'score_weight']].sum()
    x['score'] /= x['score_weight']
    x.drop(columns='score_weight', inplace=True)
    x.sort_values(by='score', ascending=False, inplace=True)
    x[x['plays'] >= min_plays].to_csv(os.path.join(out_dir, 'tracks_by_mean_score.csv'))


def main():
    # delete the audio_data.csv file if changing the input files, otherwise the cache will be used!
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not use_cache or not os.path.exists(os.path.join(data_dir, data_filename)):
        process_data()
    data = read_data()
    print(f'total hours: {data["hours_played"].sum():.0f}')
    print(f'plays: {data["plays"].sum()}')
    print(f'unique tracks: {len(data["uri"].unique())}')

    # basic statistics
    save_artists(data)
    save_tracks_by_time(data)
    save_tracks_by_plays(data)
    save_tracks_by_single_day_time(data)

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
        data['duration_ms'] = data.groupby('uri')['ms_played'].transform('max').clip(1000 * 60 * 2, 1000 * 60 * 5)
    post_process(data)

    # advanced statistics making use of the calculated score / metadata
    save_by_score(data)
    save_by_mean_score(data)


if __name__ == '__main__':
    main()
