import numpy as np
import pandas as pd
import pickle as pk
import os

# parse data to end at affect
EXPERT_FEATURES = ['avg_attemptCount', 'avg_bottomHint', 'avg_correct', 'avg_frIsHelpRequest', 'avg_frPast5HelpRequest', 'avg_frPast5WrongCount', 'avg_frPast8HelpRequest', 'avg_frPast8WrongCount', 'avg_frWorkingInSchool', 'avg_hint', 'avg_hintCount', 'avg_hintTotal', 'avg_original', 'avg_past8BottomOut', 'avg_scaffold', 'avg_stlHintUsed', 'avg_timeSinceSkill', 'avg_timeTaken', 'avg_totalFrAttempted', 'avg_totalFrPastWrongCount', 'avg_totalFrPercentPastWrong', 'avg_totalFrSkillOpportunities', 'avg_totalFrTimeOnSkill', 'max_attemptCount', 'max_bottomHint', 'max_correct', 'max_frIsHelpRequest', 'max_frPast5HelpRequest', 'max_frPast5WrongCount', 'max_frPast8HelpRequest', 'max_frPast8WrongCount', 'max_frWorkingInSchool', 'max_hint', 'max_hintCount', 'max_hintTotal', 'max_original', 'max_past8BottomOut', 'max_scaffold', 'max_stlHintUsed', 'max_timeSinceSkill', 'max_timeTaken', 'max_totalFrAttempted', 'max_totalFrPastWrongCount', 'max_totalFrPercentPastWrong', 'max_totalFrSkillOpportunities', 'max_totalFrTimeOnSkill', 'min_attemptCount', 'min_bottomHint', 'min_correct', 'min_frIsHelpRequest', 'min_frPast5HelpRequest', 'min_frPast5WrongCount', 'min_frPast8HelpRequest', 'min_frPast8WrongCount', 'min_frWorkingInSchool', 'min_hint', 'min_hintCount', 'min_hintTotal', 'min_original', 'min_past8BottomOut', 'min_scaffold', 'min_stlHintUsed', 'min_timeSinceSkill', 'min_timeTaken', 'min_totalFrAttempted', 'min_totalFrPastWrongCount', 'min_totalFrPercentPastWrong', 'min_totalFrSkillOpportunities', 'min_totalFrTimeOnSkill', 'sum_attemptCount', 'sum_bottomHint', 'sum_correct', 'sum_frIsHelpRequest', 'sum_frPast5HelpRequest', 'sum_frPast5WrongCount', 'sum_frPast8HelpRequest', 'sum_frPast8WrongCount', 'sum_frWorkingInSchool', 'sum_hint', 'sum_hintCount', 'sum_hintTotal', 'sum_original', 'sum_past8BottomOut', 'sum_scaffold', 'sum_stlHintUsed', 'sum_timeSinceSkill', 'sum_timeTaken', 'sum_totalFrAttempted', 'sum_totalFrPastWrongCount', 'sum_totalFrPercentPastWrong', 'sum_totalFrSkillOpportunities', 'sum_totalFrTimeOnSkill']
TARGET_FEATURES = ['confusion', 'concentration', 'boredom', 'frustration']


def apply_clip_sequence_df(df: pd.DataFrame, time_col: str, group_col: str, threshold: float, sort_columns: bool = True) -> pd.DataFrame:
    """
    Add a `clip_sequence` column to df following the same rules as your CSV-based function. These affect detectors
    have been found to perform better when applied to shorter sequences, so this column is meant to break up
    student clip sequences into smaller segments where there are longer gaps between clips
    """
    df = df.copy()

    # make sure the time column is numeric
    df[time_col] = df[time_col].astype(float)

    # compute clip_sequence in a single pass
    clip_seq = []
    prev_time = None
    prev_group = None
    prev_seq = None

    for _, row in df.iterrows():
        cur_time = row[time_col]
        cur_group = row[group_col]

        if prev_group is None:
            # very first data row
            seq = 0
        else:
            # same group AND prev_time - cur_time <= 15
            if (cur_group == prev_group) and ((prev_time - cur_time) <= threshold):
                seq = prev_seq
            else:
                seq = prev_seq + 1

        clip_seq.append(seq)
        prev_time, prev_group, prev_seq = cur_time, cur_group, seq

    df['clip_sequence'] = clip_seq

    if sort_columns:
        df = df.reindex(sorted(df.columns), axis=1)

    return df

padded_expert_length = []
padded_expert_input = []
padded_expert_target = []
padded_expert_weight = []
padded_expert_sid = []


def __parse_data(df, student_column='user_id'):
    present_targets = [c for c in TARGET_FEATURES if c in df.columns]
    if present_targets:
        if (df[present_targets].sum(axis=1) == 1).sum() > 0:
            keepers = df[present_targets[0]].notna()
            df = df[keepers]
            return {
                'length': len(df),
                'input': df[EXPERT_FEATURES].values.reshape(-1, len(EXPERT_FEATURES)),
                'target': df[present_targets].values.reshape(-1, len(present_targets)),
                'weight': df[present_targets].sum(axis=1).values.reshape(-1, 1),
                'sid': df[student_column].iloc[0]
            }
        else:
            return None
    else:
        return {
            'length': len(df),
            'input': df[EXPERT_FEATURES].values.reshape(-1, len(EXPERT_FEATURES)),
            'target': None,
            'weight': None,
            'sid': df[student_column].iloc[0]
        }


def parse_data(df, student_column='user_id'):
    present_targets = [c for c in TARGET_FEATURES if c in df.columns]
    grouped = df.groupby([student_column, 'clip_sequence'])
    results = [
        __parse_data(group, student_column)
        for _, group in grouped
    ]

    # Filter out None results
    results = [r for r in results if r is not None]

    # Aggregate lists for each key
    parsed = {
        'length': [r['length'] for r in results],
        'input': [r['input'] for r in results],
        'target': None if not present_targets else [r['target'] for r in results],
        'weight': None if not present_targets else [r['weight'] for r in results],
        'sid': [r['sid'] for r in results],
    }

    return parsed


def pad_data(a, max_length):
    pad = np.zeros((max_length - a.shape[0], a.shape[1]))
    return np.concatenate([pad, a])


if __name__ == "__main__":
    data = pd.read_csv('resources/aligned_affect_observations.csv')
    data = data.sort_values(['user_id', 'clip_id']).reset_index()
    data[TARGET_FEATURES] = data[TARGET_FEATURES].fillna(0)
    data = apply_clip_sequence_df(data, time_col='sum_timeTaken', group_col='user_id', threshold=300, sort_columns=False)
    data = data[data[TARGET_FEATURES].sum(axis=1) < 2]
    data = data.sort_values(['user_id', 'clip_sequence', 'clip_id']).reset_index()

    parsed_data = parse_data(data, 'user_id')

    max_expert_length = max(parsed_data['length'])
    padded_expert_input = [pad_data(i, max_expert_length) for i in parsed_data['input']]
    padded_expert_target = [pad_data(i, max_expert_length) for i in parsed_data['target']]
    padded_expert_weight = [pad_data(i, max_expert_length) for i in parsed_data['weight']]

    if not os.path.exists('data'):
        os.makedirs('data')

    pk.dump(padded_expert_input, open('data/padded_expert_input.pkl', 'wb'))
    pk.dump(padded_expert_target, open('data/padded_expert_target.pkl', 'wb'))
    pk.dump(padded_expert_weight, open('data/padded_expert_weight.pkl', 'wb'))
    pk.dump(parsed_data['sid'], open('data/padded_expert_sid.pkl', 'wb'))