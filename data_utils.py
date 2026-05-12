import pandas as pd
import numpy as np
import os
import glob
import cv2

def load_data(data_path : str) -> List:
    """
    Loads given gaze data file into a list of tuples.

    Inputs:
    - data_path: Path to .csv file. Assumed that the file has headers 't', 'sx', 'sy', 'valid', 'aoi'

    Outputs:
    - List of tuples with format (t, x, y)
    """
    df = pd.read_csv(data_path)
    # Filter data to only get coordinates from valid and onscreen gaze points
    df = df[(df['valid'] == 1) & (df['aoi'] != 'offscreen')]
    df = df[['t', 'sx', 'sy']]
    return list(df.itertuples(index=False, name=None))


def load_data_df(data_path : str) -> List:
    """
    Loads given gaze data file into a pandas data frame.

    Inputs:
    - data_path: Path to .csv file. Assumed that the file has headers 't', 'sx', 'sy', 'valid', 'aoi'

    Outputs:
    - List of tuples with format (t, x, y)
    """
    df = pd.read_csv(data_path)
    # Filter data to only get coordinates from valid and onscreen gaze points
    df = df[(df['valid'] == 1) & (df['aoi'] != 'offscreen')]
    df = df[['t', 'sx', 'sy']]
    return df


def load_characterized_sets(ch_mapping_path: str, nn_log: str, data_folder: str, stimuli_file: str) -> dict:
    """
    Returns dictionary of data file paths for ASD, NT, ATP groups split by age (>18 months, <18 months)
    """
    # load csvs (mapping, log for static_nn_mean)
    mapping = pd.read_csv(ch_mapping_path)
    log = pd.read_csv(nn_log)
    # Filter logs to include only entries for desired stimuli
    log = log[log['sourcefile'].str.contains(stimuli_file, case=False, na=False)]
    # Filter logs to include only entries with static_nn_mean <= 105
    log['static_nn_mean'] = pd.to_numeric(log['static_nn_mean'], errors='coerce')
    log = log[log['static_nn_mean'] <= 105]
    # Logs df now contains list of participants with valid data for a specific stimulus
    # Filter logs df into 3 groups depending on mapping diagnosis outcome
    merged = pd.merge(log, mapping, left_on='filebase', right_on='edf', how='inner')

    # For each group, go through data folder and read in files for each valid entry
    res = {
        'young': {
            'ASD': [],
            'NT': [],
            'ATP': []
        },
        'old': {
            'ASD': [],
            'NT': [],
            'ATP': []
        }
    }

    for file in os.listdir(data_folder):
        patient_id = file[2:10]
        if stimuli_file.lower() in file.lower() and patient_id in merged['edf'].values:
            full_path = os.path.join(data_folder, file)
            outcome = mapping[mapping['edf'] == patient_id]

            for index, row in outcome.iterrows():
                if row['et_age_corrected_months'] < 18:
                    res['young'][row['dx']].append(full_path)
                else:
                    res['old'][row['dx']].append(full_path)

    return res


def normalize_data(files: list) -> list:
    """
    Given a list of eye data files, normalizes based on how many files are from
    one individual. Returns list of lists of tuples of form (t, x, y, w) where w is
    the weight that should be given to that particular data point.
    """
    weights = {}
    res = []
    for f in files:
        patient_id = f[2:10]
        data = load_data(f)
        w_col = []
        if not patient_id in weights:
            count = sum(patient_id in s for s in files)
            weights[patient_id] = 1 / count
        
        w_col = [weights[patient_id]] * len(data)
        data = [row + (col_val,) for row, col_val in zip(data, w_col)]
        if len(data) > 0:
            res.append(data)

    return res

def get_proportion_in_mask(data_files, mask_file):
    """
    Given a list of individual's eye tracking data and a mask representing the heads
    for each frame, returns the proportion of individuals looking at a head in each
    frame
    """
    mask_df = pd.read_csv(mask_file)
    masks = []
    for file in data_files:
        file_df = load_data_df(file)
        masks.append(get_head_mask(file_df, mask_df))
    
    masks_np = np.array(masks)
    return (np.sum(masks_np, axis=0) / np.shape[0]).tolist()


def get_head_mask(data_df, mask_df, fps=60.0):
    ms_per_frame = 1000 / fps
    # convert frames to ms.
    data_df['frame'] = (data_df['t'] / ms_per_frame).astype(int)

    result = []
    for index, row in mask_df.iterrows():
        cur_frame = row['frame']
        head_l = (row['left_head_x'], row['left_head_y'], row['left_head_radius'] * row['dilation'])
        head_r = (row['left_head_x'], row['left_head_y'], row['right_head_radius'] * row['dilation'])

        cur_data = data_df.loc[data_df['frame'] == cur_frame]
        num_in_mask = 0
        for index, cur_row in cur_data.iterrows():
            if in_mask((cur_row['sx'], cur_row['sy']), head_l) or in_mask((cur_row['sx'], cur_row['sy']), head_r):
                num_in_mask = num_in_mask + 1

        if num_in_mask >= len(cur_data) / 2:
            result.append(1)
        else:
            result.append(0)

    return result


def in_mask(point, head_mask):
    x, y = point
    hx, hy, r = head_mask

    return ((x - hx) ** 2 + (y - hy) ** 2) <= (r ** 2)


def get_middle_frame(video_path : str) -> np.ndarray:
    """
    Returns middle frame of video in an ndarray of dimension W x H x 3 representing
    the frame in RGB.

    Input: video_path: path to video
    Output: ndarray representation of frame at middle of video
    """
    cap = cv2.VideoCapture(video_path)
    tot_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    middle_frame_index = tot_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
    print(f"video dims: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} x {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    ret, frame = cap.read()

    cap.release()
    if not ret or frame is None:
        print(f"Error: Could not read frame at index {middle_frame_index}.")
        return None
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame