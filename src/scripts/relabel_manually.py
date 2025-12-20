import argparse
import random
import pandas as pd
from src.util import PathHelper

parser = argparse.ArgumentParser(description='A script that allows to relabel some data.')
parser.add_argument(
    '--sample_n',
    type=int,
    default=None,
    help='The number of manually relabeled messages in the test sample.'
)
args = parser.parse_args()

pd.set_option('display.max_colwidth', None)

df_base = pd.read_csv(PathHelper.data.raw.data_set)
df_ask = pd.read_csv(PathHelper.data.raw.get_path('AskReddit_comments.csv'))
df_task = pd.read_csv(PathHelper.data.raw.get_path('TrueAskReddit_comments.csv'))
df_casual = pd.read_csv(PathHelper.data.raw.get_path('CasualConversation_comments.csv'))

df_list = [df_base, df_ask, df_task, df_casual]
output_file = PathHelper.data.raw.get_path('test_set.csv')
if output_file.exists():
    df_output = pd.read_csv(output_file)
else:
    df_output = pd.DataFrame(columns=['text', 'class'])

mapper = {'s': 'suicide', 'n': 'non-suicide'}
while df_output.shape[0] < args.sample_n:
    if df_output[df_output['class'] == 'suicide'].shape[0] < args.sample_n / 2:
        current_df = df_base
        indexes = current_df[current_df['class'] == 'suicide'].index
    else:
        current_df = random.choice(df_list)
        indexes = current_df[current_df['class'] == 'non-suicide'].index

    idx = random.choice(list(indexes))
    chosen_row = current_df.loc[idx]
    print('\n'*2)
    print('-'*50)
    print(chosen_row['text'].strip())
    print('-'*50)
    print(chosen_row['class'])
    print('\n')
    answer = None
    while not answer in ['s', 'n', 'u']:
        answer = input("Is it suicidal, non suicidal or unknown? (s/n/u): ").lower()

    if answer =='u':
        continue

    current_df.loc[idx, 'class'] = mapper[answer]
    df_output.loc[len(df_output)] = current_df.loc[idx]
    current_df.drop(index=idx, inplace=True)
    print(df_output['class'].value_counts())

df_output.to_csv(output_file, index=False)
answer = None
while not answer in ['y', 'n']:
    answer = input("Do you want to remove relabeled rows from original files? (y/n): ").lower()

if answer == 'y':
    df_base.to_csv(PathHelper.data.raw.data_set, index=False)
    df_ask.to_csv(PathHelper.data.raw.get_path('AskReddit_comments.csv'), index=False)
    df_task.to_csv(PathHelper.data.raw.get_path('TrueAskReddit_comments.csv'), index=False)
    df_casual.to_csv(PathHelper.data.raw.get_path('CasualConversation_comments.csv'), index=False)
