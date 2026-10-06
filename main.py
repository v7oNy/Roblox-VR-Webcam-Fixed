import os
import sys
import json
import time
import threading

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__)), "config.json")


def load_config():
    defaults = {
        "username": "MyRobloxUsername",
        "authorization": "CHANGE-ME",
        "upload_url": "https://www.example.com/upload_pose/{username}",
        "camera_index": 0,
        "cooldown": 0.2
    }
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                defaults.update(loaded)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(defaults, f, indent=4)
        except OSError:
            pass
    return defaults


config = load_config()

print("Loading extensions...")

# Roblox username of player
username = str(config.get("username", "MyRobloxUsername"))
# AUTHORIZATION token (games will ask you for this)
authorization = str(config.get("authorization", "CHANGE-ME"))
# SERVER OWNERS: move {username} to wherever the player's username should be inserted into the URL.
# PLAYERS: this value should be given from the roblox game.
upload_url = str(config.get("upload_url", "https://www.example.com/upload_pose/{username}")).replace("{username}", username)

# how long in seconds between each pose update
cooldown = float(config.get("cooldown", 0.2))
camera_index = int(config.get("camera_index", 0))

# are we using the typing input feature (True/False case-sensitive)?
typing_input = False

# -----FOR TYPING FEATURE WITHOUT SERVER-----
start_typing_delay = 2
press_delay = 0.002
key_map = {
    "q": "0", "e": "1", "r": "2", "t": "3", "y": "4", "u": "5",
    "p": "6", "f": "7", "g": "8", "h": "9", "j": ',', "k": '[',
    "l": ']', "z": "-", "v": "."
}

starting_char = 'x'
ending_char = 'c'

# -----DO NOT EDIT-----
uploading = True

import requests as rq
import mediapipe as mp
import cv2
import numpy as np
import pydirectinput
import calcs

pydirectinput.PAUSE = press_delay

roblox_rotations = {}
roblox_landmarks = {}
sent_data = []
sequence_num = 1

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def replace_all(string: str):
    for i in key_map:
        v = str(key_map[i])
        string = string.replace(v, i)
    return string


def assign_angles(vector1: str, vector2: str, repeat_side: bool = False):
    global roblox_rotations
    v1_coords = roblox_landmarks[vector1]
    v2_coords = roblox_landmarks[vector2]

    orient = calcs.look_at(
        np.array([v1_coords["x"], v1_coords["y"], v1_coords["z"]]),
        np.array([v2_coords["x"], v2_coords["y"], v2_coords["z"]])
    )
    roll, pitch, yaw = calcs.angles(orient)

    roblox_rotations[vector1] = {
        "x": roll,
        "y": pitch,
        "z": yaw,
        "visibility": v2_coords["visibility"]
    }

    if repeat_side:
        assign_angles(
            vector1.replace("left", "right"),
            vector2.replace("left", "right")
        )


def keep_uploading():
    global sequence_num
    last_sequence_num = 1
    requests_this_minute = 0
    last_request_count_time = time.time()
    last_request_sent = last_request_count_time
    request_headers = {"authorization": authorization}

    while True:
        if sequence_num == 0:
            break
        elif sequence_num > last_sequence_num:
            if time.time() - last_request_sent >= cooldown:
                if typing_input:
                    pydirectinput.press(starting_char)
                    raw_json = json.dumps(sent_data)
                    string_data = replace_all(raw_json.replace(" ", ""))
                    pydirectinput.write(string_data)
                    pydirectinput.press(ending_char)
                else:
                    try:
                        r = rq.post(
                            url=upload_url,
                            json=sent_data,
                            headers=request_headers,
                            timeout=10
                        )
                    except rq.RequestException as e:
                        print(e)
                        continue

                    requests_this_minute += 1
                    last_request_sent = time.time()

                    if r.status_code == 200:
                        last_sequence_num = sequence_num
                    else:
                        print(f"Status code: {r.status_code}")
                        print(f"Response body: {r.text}")

            if time.time() - last_request_count_time >= 60:
                print(f"Requests this minute: {requests_this_minute}")
                last_request_count_time = time.time()
                requests_this_minute = 0
        else:
            time.sleep(0.01)


def main():
    global roblox_landmarks
    global roblox_rotations
    global sequence_num
    global sent_data

    print("Opening the webcam...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Could not open camera index {camera_index}. Change camera_index in config.json and restart.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Initializing pose detection...")

    with mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as pose:
        print('View the new "Webcam Pose" window to make sure your webcam can see your pose clearly.')

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            imageToDisplay = cv2.flip(image, 1)
            imageToProcess = cv2.cvtColor(imageToDisplay, cv2.COLOR_BGR2RGB)

            imageToProcess.flags.writeable = False
            results = pose.process(imageToProcess)

            mp_drawing.draw_landmarks(
                imageToDisplay,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            world_landmarks = results.pose_world_landmarks

            if world_landmarks:
                roblox_landmarks = {
                    "left_shoulder": world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER],
                    "right_shoulder": world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                    "left_elbow": world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW],
                    "right_elbow": world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW],
                    "left_wrist": world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST],
                    "right_wrist": world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST],
                    "left_hip": world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP],
                    "right_hip": world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP],
                    "left_knee": world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE],
                    "right_knee": world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE],
                    "left_ankle": world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE],
                    "right_ankle": world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE],
                    "nose": world_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                }

                for body_part in roblox_landmarks:
                    body_part_value = roblox_landmarks[body_part]
                    roblox_landmarks[body_part] = {
                        "x": body_part_value.x,
                        "y": body_part_value.y,
                        "z": body_part_value.z,
                        "visibility": body_part_value.visibility,
                    }

                roblox_rotations = {}

                assign_angles("left_shoulder", "left_elbow", repeat_side=True)
                assign_angles("left_elbow", "left_wrist", repeat_side=True)
                assign_angles("left_hip", "left_knee", repeat_side=True)
                assign_angles("left_knee", "left_ankle", repeat_side=True)

                lf_sh = roblox_landmarks["left_shoulder"]
                rt_sh = roblox_landmarks["right_shoulder"]
                nose = roblox_landmarks["nose"]

                shoulder_midpoint = {
                    "x": (lf_sh["x"] + rt_sh["x"]) / 2,
                    "y": (lf_sh["y"] + rt_sh["y"]) / 2,
                    "z": (lf_sh["z"] + rt_sh["z"]) / 2
                }

                neck_matrix = calcs.look_at(
                    np.array([shoulder_midpoint["x"], shoulder_midpoint["y"], shoulder_midpoint["z"]]),
                    np.array([nose["x"], nose["y"], nose["z"]])
                )
                nk_x, nk_y, nk_z = calcs.angles(neck_matrix)

                roblox_rotations["neck"] = {
                    "x": nk_x,
                    "y": nk_y,
                    "z": nk_z,
                    "visibility": (lf_sh["visibility"] + rt_sh["visibility"]) / 2
                }

                lf_hip = roblox_landmarks["left_hip"]
                rt_hip = roblox_landmarks["right_hip"]

                waist_midpoint = {
                    "x": (lf_hip["x"] + rt_hip["x"]) / 2,
                    "y": (lf_hip["y"] + rt_hip["y"]) / 2,
                    "z": (lf_hip["z"] + rt_hip["z"]) / 2
                }

                waist_matrix = calcs.look_at(
                    np.array([waist_midpoint["x"], waist_midpoint["y"], waist_midpoint["z"]]),
                    np.array([shoulder_midpoint["x"], shoulder_midpoint["y"], shoulder_midpoint["z"]])
                )
                waist_x, waist_y, waist_z = calcs.angles(waist_matrix)

                roblox_rotations["waist"] = {
                    "x": waist_x,
                    "y": waist_y,
                    "z": waist_z,
                    "visibility": (lf_hip["visibility"] + rt_hip["visibility"]) / 2
                }

                landmarks_order = [
                    "neck", "left_shoulder", "right_shoulder",
                    "left_elbow", "right_elbow", "waist",
                    "left_hip", "right_hip", "left_knee", "right_knee"
                ]

                sent_data = []

                for ln_name in landmarks_order:
                    ln = roblox_rotations[ln_name]
                    ln_array = [
                        round(ln["x"], 2),
                        round(ln["y"], 2),
                        round(ln["z"], 2),
                        round(ln["visibility"], 2)
                    ]
                    sent_data.append(ln_array)

                if uploading:
                    sequence_num += 1

            cv2.imshow("Webcam Pose", imageToDisplay)

            if cv2.waitKey(1) == 27:
                sequence_num = 0
                cap.release()
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    if typing_input:
        input(
            "Press Enter in the console to begin the program. "
            + f"The script will begin typing on the screen in {start_typing_delay} seconds, "
            + "so make sure Roblox is in focus."
        )

    time.sleep(start_typing_delay)
    upload_process = threading.Thread(target=keep_uploading)
    upload_process.start()
    main()
    upload_process.join()
