"""This module contains functions for plotting audio embeddings on a 2D/3D graph.

 It also acts as a data cleaning/relabelling tool"""
import base64

import boto3
import re
import os
from dash.dependencies import Input, Output
from jupyter_dash import JupyterDash
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import umap
import numpy as np
from flask_caching import Cache

from ah_consonants_ah import AhConsonantsAhDataset
from dataset import DatasetStage
from utils import S3_BUCKET

try:
    os.remove('./plotly_cache', recursive=True)
except:
    pass

cache = Cache(config={
    # try 'filesystem' if you don't want to setup redis
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': './plotly_cache'
})

DEFAULT_N_NEIGHBORS = 8
DEFAULT_N_COMPONENTS = 3
DEFAULT_MIN_DIST = 0.2
DEFAULT_MIN_GAP = 0


@cache.memoize(timeout=3600)
def calc_umap(df, embeddings_col='X', n_components=DEFAULT_N_COMPONENTS, n_neighbors=DEFAULT_N_NEIGHBORS, min_dist=DEFAULT_MIN_DIST):
    u = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric='cosine')
    X = u.fit_transform(list(df[embeddings_col]))
    if n_components == 3:
        df[['umap_x', 'umap_y', 'umap_z']] = X
    elif n_components == 2:
        df[['umap_x', 'umap_y']] = X
        try:
            df = df.drop('umap_z', axis=1)
        except:
            pass
    elif n_components == 1:
        df['umap_x'] = X
        try:
            df = df.drop('umap_y', axis=1)
            df = df.drop('umap_z', axis=1)
        except:
            pass
    else:
        raise ValueError('Invalid number of n components')
    return df


# @cache.memoize(timeout=3600)
def create_figure(data, color_col='label', width=None, height=None):
    hover_template = """Sound: %{customdata[0]}<br>
Session ID: %{customdata[1]}<br>
Filename: %{customdata[2]}
"""

    # Use UMAP to determine X and Y in the 2D space

    custom_data_columns = ['local_filepath', 'label', 'group', 'filename']
    if 'clf' in data.columns:
        custom_data_columns.append('clf')
        hover_template += '<br>Classification: %{customdata[' + str(len(custom_data_columns) - 1) + ']}'
    if 'gap' in data.columns:
        custom_data_columns.append('gap')
        hover_template += '<br>Gap: %{customdata[' + str(len(custom_data_columns) - 1) + ']}'

    fig = go.Figure()
    fig.update_layout(
        autosize=False,
        width=width,
        height=height
    )
    color_keys = [k for k in list(data[color_col].unique()) if k != 'negative']
    color_map = dict(zip(color_keys, px.colors.qualitative.Plotly))
    if 'negative' in list(data[color_col].unique()):
        color_map['negative'] = '#333333'

    # We do marker style based on if prediction is correct

    for is_correct in ([True] + ([False] if 'clf' in data.columns else [])):
        if 'clf' in data.columns:
            is_correct_data = data[(data['clf'] == data['label']) == is_correct]
        else:
            # If predictions are not provided it is always the same marker
            is_correct_data = data

        # We do marker color based on color column
        for color_val in data[color_col].unique():
            color_data = is_correct_data[is_correct_data[color_col] == color_val]
            if 'umap_z' not in data.columns:
                fig.add_trace(go.Scatter(x=color_data['umap_x'], y=color_data['umap_y'], name=color_val + ('' if is_correct else ' (wrong prediction)'), mode='markers',
                                         marker=dict(symbol=('circle' if is_correct else 'x'), color=list([color_map[x] for x in color_data[color_col]])),
                                         customdata=color_data[custom_data_columns]))
            else:
                fig.add_trace(go.Scatter3d(x=color_data['umap_x'], y=color_data['umap_y'], z=color_data['umap_z'], name=color_val + ('' if is_correct else ' (wrong prediction)'), mode='markers',
                                           marker=dict(symbol=('circle' if is_correct else 'x'), color=list([color_map[x] for x in color_data[color_col]]), size=3, sizemode='diameter'),
                                           customdata=color_data[custom_data_columns]))

    fig.update_traces(hovertemplate=hover_template, showlegend=True)

    return fig


def plot_embeddings(data,
                    embeddings_col='X',
                    color_col='label',
                    s3_bucket=S3_BUCKET,
                    dataset_stages=(DatasetStage.RAW, DatasetStage.TRIMMED),
                    playback_dataset_stage=DatasetStage.RAW,
                    filename_pattern='ah-{sound_name}ah-{sample_number}.wav',
                    dry=False,
                    width=None,
                    height=None,
                    port=8050,
                    mode='inline'):
    """

    Args:
        data: DataFrame with label, group, filename, [clf (optional), gap (optional)] columns.
        embeddings_col: Column name for the embedding column.
        dataset_stages: Stages for the datasets that will be affected by rename/remove operations.
        playback_dataset_stage: Dataset stage to use for sound playback (on mouse click).
        color_col: Column to use for coloring.
        s3_bucket: S3 Bucket.
        filename_pattern: Pattern for filenames with {sound_name} and {sample_number} being template variables.
        width: Figure width.
        height: Figure height.
        mode: Plotly Dash mode ('inline', 'jupyterlab', etc)
        dry: True, if no S3 renaming/deleting should be performed

    Returns:

    """
    s3 = boto3.client('s3')
    app = JupyterDash('__main__')
    cache.init_app(app.server)
    print(f'Dry run {"ENABLED" if dry else "DISABLED"}')

    data = data.sort_values(color_col)

    data[embeddings_col] = data[embeddings_col].apply(lambda x: x.flatten())
    data = calc_umap(data, embeddings_col=embeddings_col)
    fig = create_figure(data, width=width, height=height, color_col=color_col)

    layout_elements = [
        dcc.Loading(id="loading-1", children=[dcc.Graph(id="scatter-plot", figure=fig)], type="default"),
        html.Audio(
            id='audio',
            src='',
            controls=True,
            autoPlay=True
        ),
        html.Div(dcc.Input(id='label', type='text')),
        html.Div(id='sound-info-1', children='Click on a data point'),
        html.Div(id='sound-info-2', children=''),
        html.Button('Re-label', id='relabel'),
        html.Button('Remove', id='remove'),
        html.Div(id='container', children='Click on a datapoint first to choose a new label'),
        html.Div(id='container2', children=''),
        html.Div(children='n-neighbours'),
        html.Div(dcc.Slider(
            id='n-neighbors-slider',
            min=2,
            max=50,
            step=1,
            value=DEFAULT_N_NEIGHBORS,
            marks={i: str(i) for i in [2, 5, 10, 15, 20, 30, 40, 50]},
        )),
        html.Div(children='n-components'),
        html.Div(dcc.Slider(
            id='n-components-slider',
            min=1,
            max=3,
            step=1,
            value=DEFAULT_N_COMPONENTS,
            marks={
                1: '1',
                2: '2',
                3: '3'
            },
        )),
        html.Div(children='min dist'),
        html.Div(dcc.Slider(
            id='min-dist-slider',
            min=0,
            max=1,
            step=0.01,
            value=DEFAULT_MIN_DIST,
            marks={i: str(i) for i in [0, 0.2, 0.4, 0.6, 0.8, 1]},
        )),
        html.Div(children='Minimum gap between label and prediction'),
        html.Div(dcc.Slider(
            id='min-gap-slider',
            min=0,
            max=1,
            step=0.01,
            value=DEFAULT_MIN_GAP,
            marks={i: str(i) for i in [0, 0.2, 0.4, 0.6, 0.8, 1]},
        )),
    ]

    app.layout = html.Div(layout_elements)

    @app.callback(
        dash.dependencies.Output('scatter-plot', 'figure'),
        [dash.dependencies.Input('n-components-slider', 'value'),
         dash.dependencies.Input('n-neighbors-slider', 'value'),
         dash.dependencies.Input('min-dist-slider', 'value'),
         dash.dependencies.Input('min-gap-slider', 'value')])
    def generate_graph(n_components, n_neighbors, min_dist, min_gap):
        data_umaped = calc_umap(df=data, n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist)
        if 'gap' in data.columns:
            data_filtered = data_umaped[data_umaped['gap'] >= min_gap]
        else:
            data_filtered = data_umaped

        return create_figure(data_filtered, width=width, height=height, color_col=color_col)

    @app.callback(
        Output('container', 'children'),
        [Input('relabel', 'n_clicks')],
        [State('label', 'value'),
         State('scatter-plot', "clickData")])
    def relabel(n_clicks, value, clickData):
        if clickData:
            old_local_filepath, old_label, group, old_filename = clickData['points'][0]['customdata'][:4]
            if old_label == value:
                return "Type in a new label in the text box"
            for dataset_stage in dataset_stages:
                s3_dataset_path = AhConsonantsAhDataset.get_path(stage=dataset_stage, local=False)
                local_dataset_path = AhConsonantsAhDataset.get_path(stage=dataset_stage, local=True)
                old_local_filepath = AhConsonantsAhDataset.get_filepath(old_filename, group=group, stage=dataset_stage, local=True)
                old_s3_filepath = AhConsonantsAhDataset.get_filepath(old_filename, group=group, stage=dataset_stage, local=False)

                # S3
                # Look for a gap in sample numbering, or if such does not exist take the next available number
                list_response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=os.path.join(s3_dataset_path, group))
                files_in_session = [os.path.basename(item["Key"]) for item in list_response['Contents']]
                regex = filename_pattern.replace('{sound_name}', value).replace('{sample_number}', r'(\d+)')
                taken_numbers = [int(re.search(regex, file).group(1)) for file in files_in_session if re.findall(regex, file)]
                available_number = (min(set(range(1, max(taken_numbers + [0]) + 2)).difference(taken_numbers)))
                new_filename = filename_pattern.replace('{sound_name}', value).replace('{sample_number}', str(available_number))

                new_local_filepath = os.path.join(local_dataset_path, group, new_filename)
                new_s3_filepath = os.path.join(s3_dataset_path, group, new_filename)
                print('Rename:', f's3://{s3_bucket}/{old_s3_filepath} -> s3://{s3_bucket}/{new_s3_filepath}')
                print('Rename:', old_local_filepath, '->', new_local_filepath)
                if not dry:
                    s3.copy_object(CopySource={'Bucket': s3_bucket, 'Key': old_s3_filepath}, Bucket=s3_bucket, Key=new_s3_filepath)
                    s3.delete_object(Bucket=s3_bucket, Key=old_s3_filepath)
                    try:
                        os.remove(new_local_filepath)  # overwritting just in case local has something there already
                    except:
                        pass
                    os.rename(old_local_filepath, new_local_filepath)
            if not dry:
                return f'Relabelled "{group}/{old_filename}" -> "{group}/{new_filename}"'
            else:
                return f'(DRY): Relabelled "{group}/{old_filename}" -> "{group}/{new_filename}"'

    @app.callback(
        Output('container2', 'children'),
        [Input('remove', 'n_clicks')],
        [State('scatter-plot', "clickData")]
    )
    def remove(n_clicks, click_data):
        if click_data:
            local_filepath, label, group, filename = click_data['points'][0]['customdata'][:4]
            for dataset_stage in dataset_stages:
                # Removing across all dataset stages
                local_filepath = AhConsonantsAhDataset.get_filepath(filename, group=group, stage=dataset_stage, local=True)
                s3_filepath = AhConsonantsAhDataset.get_filepath(filename, group=group, stage=dataset_stage, local=False)
                print('Remove:', f's3://{s3_bucket}/{s3_filepath}')
                print('Remove:', local_filepath)
                if not dry:
                    s3.delete_object(Bucket=s3_bucket, Key=s3_filepath)
                    os.remove(local_filepath)
            if not dry:
                return f'Removed "{group}/{filename}'
            else:
                return f'(DRY): Removed {group}/{filename}'

    @app.callback(
        [Output("audio", "src"),
         Output("sound-info-1", "children"),
         Output("sound-info-2", "children"),
         Output("label", "value")],
        [Input("scatter-plot", "clickData")])
    def play_sound(click_data):
        if click_data is None:
            return ['', '', '', '']

        if 'clf' in data.columns:
            local_filepath, label, group, filename, clf = click_data['points'][0]['customdata'][:5]
        else:
            local_filepath, label, group, filename = click_data['points'][0]['customdata'][:4]
            clf = None

        try:
            if not os.path.exists(local_filepath):
                local_filepath = AhConsonantsAhDataset.save_locally(filename, s3_bucket=s3_bucket, stage=playback_dataset_stage, group=group)
        except:
            print('S3 file cannot be retrieved')
            return ['', '', '', '']
        with open(local_filepath, 'rb') as f:
            audio_bytes = f.read()
        fmt = os.path.splitext(local_filepath)[-1][1:]
        return [
            f'data:audio/{fmt};base64,{base64.b64encode(audio_bytes).decode()}',
            f'Path: {local_filepath}',
            f'Prediction: {clf}',
            label
        ]

    return app.run_server(mode=mode, debug=True, dev_tools_ui=True,  # debug=True,
                          dev_tools_hot_reload=True, threaded=True, port=port)
