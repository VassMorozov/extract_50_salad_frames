import pandas as pd
import numpy as np
import os
import cv2
import math
from PIL import Image

class extract_frames:
    '''
    This class extracts frames from the 50 Salads action recognition videos. These videos are 30 FPS.
    The names of all videos should be stored in a CSV file with their associated index
    For each video, boundaries of each action should be stored in a file with name "{video_index}-activityAnnotation.txt"
    '''

    def __init__(self, videos, ts_file_path: str, act_annot_loc: str, vids_file_path: str, fps: int, save_loc: str):
        self.videos = videos
        self.act_annot_loc = act_annot_loc
        self.vids_file_path = vids_file_path
        self.save_loc = save_loc
        self.ts_file_path = ts_file_path
        self.fps = fps

    def output_frames(self):
        # loop through all videos
        for index, row in self.videos.iterrows():
            print(f"Working with {row['video_names']}")
            # get annotations and time stamps for said video
            annot = pd.read_csv(f"{self.act_annot_loc}\\{row['index']}-activityAnnotation.txt", header=None,
                                delimiter=" ",
                                names=["start", "end", "action"])
            annot = annot.loc[
                    annot["action"].apply(lambda x: any(substring in x for substring in ["_post", "_prep", "_core"])),
                    :]

            ts = pd.read_csv(f"{self.ts_file_path}\\timestamps-{row['index']}.txt", header=None, delimiter=" ",
                             names=["timestamp", "meh"])

            # label each frame
            for ts_index, ts_row in ts.iterrows():

                for annot_index, annot_row in annot.iterrows():
                    if ts_row["timestamp"] >= annot_row["start"] and ts_row["timestamp"] < annot_row["end"]:
                        ts.loc[ts["timestamp"] == ts_row["timestamp"], "action"] = annot_row["action"]
                        break

            # reset index to reference index later
            ts = ts.loc[ts["action"].isna() == False, :]
            ts.reset_index(inplace=True)

            # downsample to specified fps
            # only works with factors of 30
            ts = ts.iloc[::int(math.ceil(30 / self.fps)), :]
            print(f"Labels for {row['video_names']} assigned. Extracting {ts.shape} frames.")

            print(f"Begin processing of {row['video_names']}")
            cap = cv2.VideoCapture(f"{self.vids_file_path}\\{row['video_names']}")
            cap.open(f"{self.vids_file_path}\\{row['video_names']}")
            # Define the codec and create VideoWriter object
            counter = 0

            while (cap.isOpened()):
                ret, frame = cap.read()

                # if a frame is returned process it
                if frame is not False and ret:
                    # if a label exists for frame, process it
                    if ts.loc[ts["index"] == counter, :].shape[0] > 0:
                        pil_img = Image.fromarray(frame)
                        im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        Image.fromarray(im_rgb).save(f"{self.save_loc}\\{row['index']}\\frame_{counter}.jpg")

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    counter += 1
                else:
                    print("Video Processing Complete")
                    break