"""This code pings Lambda with test files and calculates a metric based on the returned scores"""

import json
from dataclasses import dataclass
import pandas as pd
import boto3
import time

from queue_wrapper import get_queue
from message_wrapper import receive_messages, delete_message

LOSS_THRESHOLD = 0.1  # 10%


@dataclass
class TestCase:
    uid: str
    file: str
    type: str
    target: str
    notes: str = ''
    expected_score_min: float = 0
    expected_score_max: float = 1

    @property
    def payload(self):
        return {
            'uid': self.uid, 'wavFile':self.file,
            'exerciseDetail': {'type': self.type,'target': self.target}
        }

    def run(self):
        return invoke_lambda_function('<your-lambda-name>', self.payload)


def validate_score(row: pd.Series)->bool:
    """Check returned result against labelled range and return a bool"""
    score_min, score_max = row['expected_score_min'], row['expected_score_max']
    score = row['result']

    return score_min <= score <= score_max


def validate_loss(row: pd.Series)->float:
    """Check returned result against labelled range and return a loss metric"""
    
    score_min, score_max = row['expected_score_min'], row['expected_score_max']
    score = row['result']
    # calc absolute distance of predicted score from nearest labelled boundary
    # if within labelled range, then return 0
    loss = abs(score-score_min) if score < score_min else abs(score-score_max) if score > score_max else 0
    
    return loss


def run_tests(test_file: str = 'test-cases.csv'):

    df = pd.read_csv(test_file)
    if 'test_case' not in df.columns:
        df['test_case'] = df.apply(lambda row: TestCase(**row), axis=1)

    return df

def invoke_lambda_function(function_name:str, payload:dict={})->str:
    """Invoke Lambda asynchronously with a payload"""

    payload_str = json.dumps(payload)
    payload_BytesArr = bytes(payload_str, encoding='utf8')

    client = boto3.client('lambda')

    # async invocation of Lambda func
    response = client.invoke(
        FunctionName=function_name,
        InvocationType="Event",
        Payload=payload_BytesArr
    )

    response_payload = response['Payload']

    return response


def calculate_mean_loss():
    """
    Ping Lambda func with all samples asynchronously then get the results and calculate a loss metric
    The responses are found in two SQS queues, one for success and one for failure
    
    Note: if a request fails, it will retry again after 1 minute and - if it fails again - retry after 2 minutes
    """
    
    test_cases_df = run_tests()
    
    uid_list = []
    score_dict = {
        'success': {},
        'fail': {}
    }

    for x in test_cases_df['test_case']:
        r = x.run()
        uid_list.append(x.uid)

    sqs_success = get_queue('my-queue-success')
    sqs_failure = get_queue('my-queue-failure')

    while uid_list:
        print('IDs remaining:', len(uid_list))

        # check success SQS queue
        msgs_success = receive_messages(sqs_success, max_number=min(10, len(uid_list)), wait_time=20)
        for msg in msgs_success:
            body = json.loads(msg.body)
            msg_uid = body['requestPayload']['uid']
            if msg_uid in uid_list:
                idx = uid_list.index(msg_uid)
                uid_list.pop(idx)
                score = body['responsePayload']['feedback']['score']
                score_dict['success'][msg_uid] = score
            delete_message(msg)

        # check failure SQS queue
        msgs_failure = receive_messages(sqs_failure, max_number=10, wait_time=0)
        for msg in msgs_failure:
            body = json.loads(msg.body)
            msg_uid = body['requestPayload']['uid']
            if msg_uid in uid_list:
                idx = uid_list.index(msg_uid)
                uid_list.pop(idx)
                score_dict['fail'][msg_uid] = score
            delete_message(msg)

    result = pd.Series(score_dict['success'], name='result')
    test_cases_df = test_cases_df.merge(result, how='left', left_on='uid', right_index=True)

    test_cases_df['is_valid'] = test_cases_df.apply(validate_score, axis=1)
    test_cases_df['loss'] = test_cases_df.apply(validate_loss, axis=1)

    mean_scores = test_cases_df[['type', 'is_valid']].groupby(by='type').mean()
    mean_loss = test_cases_df.loss.mean()

    return mean_loss, mean_scores


tick = time.time()
mean_loss, mean_scores = calculate_mean_loss()
tock = time.time()

wall_time = tock - tick
print('Elapsed time', wall_time)

print(mean_scores)
mean_score = mean_scores.is_valid.mean()
print(f'{mean_score:.0%} of tests passed.')
print(f'Loss: {mean_loss:.0%}')

if mean_loss < LOSS_THRESHOLD:
    print('Loss passed the test')
else:
    raise Exception(f'Loss was greater than {LOSS_THRESHOLD}')
