import yaml
from botocore.exceptions import ClientError
from fastapi import FastAPI, UploadFile, File, Form
import os
import ffmpeg
import boto3
import logging

app = FastAPI()

with open('config.yml') as f:
    config = yaml.load(f, yaml.SafeLoader)


def upload_to_s3(filepath, s3_dir, bucket, object_name=None):
    """Upload a file to an S3 bucket

    Args:
        filepath: Path to file that will be upload
        s3_dir: Directory in S3 to upload to
        bucket: Bucket to upload to
        object_name: S3 object name. If not specified then file_name is used
    Returns
        True if file was uploaded, else False
    """
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(filepath)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(filepath, bucket, os.path.join(s3_dir, object_name))
    except ClientError as e:
        logging.error(e)
        return False
    return True


UNSAFE_API_FEATURES_ENABLED = False

if UNSAFE_API_FEATURES_ENABLED is True:
    @app.get('/s3rm/{session_id}/{filename}')
    async def remove_from_s3(session_id: str, filename: str):
        s3_client = boto3.client('s3')
        try:
            s3_path = f"{config.get('s3_upload_dir')}/{config.get('default_dataset')}/{session_id}/{filename}"
            print(s3_path)
            response = s3_client.delete_object(Bucket=config.get('s3_bucket'), Key=s3_path)
        except ClientError as e:
            logging.error(e)
            return False
        return response


    @app.get('/s3rn/{session_id}/{filename}/{new_filename}')
    async def rename_on_s3(session_id: str, filename: str, new_filename: str):
        try:
            s3_path = f"{config.get('s3_upload_dir')}/{config.get('default_dataset')}/{session_id}/{filename}"
            new_s3_path = f"{config.get('s3_upload_dir')}/{config.get('default_dataset')}/{session_id}/{new_filename}"
            print(s3_path)
            s3_resource = boto3.resource('s3')
            response = s3_resource.Object(config.get('s3_bucket'), new_s3_path).copy_from(CopySource=s3_path)
            # response = s3_client.move_object(Bucket=config.get('s3_bucket'), Key=s3_path)
        except ClientError as e:
            logging.error(e)
            return False
        return response


@app.post("/save")
async def save_audio(
        name: str = Form(...), dataset: str = Form(...), session_id: str = Form(...),
        audio: UploadFile = File(...)):
    print(name, dataset, session_id)

    # Define save directories and make they exist
    save_directory = os.path.join(config.get('recordings_dir'), dataset, session_id)
    os.makedirs(os.path.join(save_directory, 'webm'), exist_ok=True)
    os.makedirs(os.path.join(save_directory, 'wav'), exist_ok=True)

    # Specify WEBM input
    webm_filepath = os.path.join(save_directory, 'webm', name + '.webm')
    with open(webm_filepath, 'wb') as f:
        f.write(audio.file.read())
        f.close()
    stream = ffmpeg.input(webm_filepath)

    # Specify WAV output
    wav_filepath = os.path.join(save_directory, 'wav', name + '.wav')
    stream = ffmpeg.output(stream, wav_filepath)

    # Convert
    ffmpeg.run(stream, overwrite_output=True)

    # Upload to S3
    if bool(config.get('s3_upload_enabled')):
        upload_to_s3(wav_filepath, os.path.join(config.get('s3_upload_dir'), dataset, session_id), bucket=config.get('s3_bucket'))

    return {
        'success': True,
        'webm_path': webm_filepath,
        'wav_path': wav_filepath
    }


@app.get("/")
async def test():
    print("hello")
    return "hello"
