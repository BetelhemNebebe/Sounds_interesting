import os
import sys

import librosa
import numpy as np

import torch
import torchvision as tv

from PIL import Image, ImageDraw
import cv2
from moviepy.editor import VideoFileClip, AudioFileClip
import moviepy.video.io.ImageSequenceClip

sys.path.append(os.path.abspath(f'{os.getcwd()}/..'))

from model import AudioCLIP
from utils.transforms import ToTensor1D

from NeVA import NeVAWrapper

MODEL_FILENAME = 'AudioCLIP-Full-Training.pt'
SAMPLE_RATE = 44100
IMAGE_SIZE = 224
SPLITTING_SIZE = 1
VIDEO_CLIP_SIZE = 5
BATCH_SIZE = 2

# Load Model
aclp = AudioCLIP(pretrained=f'assets\{MODEL_FILENAME}')

image_size = 224
lr = 0.1
optimization_steps = 20

scanpath_length = 5
foveation_sigma = 0.2
blur_filter_size = 41
forgetting = 0.7
blur_sigma = 10


def cosine_sim(x, y):
    """computes the cosine similarity of image and audio embedding"""
    val = torch.nn.functional.cosine_similarity(x, y, 1)
    return -val + 1

criterion = cosine_sim

def target_function(x, y):
    return y

def extract_frames(video_path):
    """Takes a single video and returns middle frame of each second and the list of all frames"""
    basename = os.path.basename(video_path)
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)

    success = True
    frames_list = []
    i = 0
    
    #extract frames
    while success:
        success, frames = video.read()
        if success and i < VIDEO_CLIP_SIZE*fps: # forexample taking 125 frames: 5 second clip * 25 fps
            frames_list.append(frames)
            i = i + 1

        else:
            break

    split_size = int(fps*SPLITTING_SIZE)
    splitted_frame_list = []
    idx = 0

    # split frames to one second frames and take the middle frame of each second
    for i in range(0, VIDEO_CLIP_SIZE):
        # split frames to a list of 1 sec (split_size) frames
        frame_split = frames_list[idx:idx+split_size] 
        frame_split = np.array(frame_split)

        # take the middle frame
        middle_element = int(len(frame_split)/2)
        middle_frame = frame_split[middle_element]
        
        splitted_frame_list.append(middle_frame)
        cv2.imwrite(f'images_batch/{basename}_{i}.jpg', middle_frame)
        idx = idx+split_size

    print("Video is extracted!")

    return splitted_frame_list, frames_list

def extract_audio(video_path):
    """Takes the path to a single video and returns the extracted audio"""
    basename = os.path.basename(video_path)

    # read video and crop to a desired size and take the audio
    video = VideoFileClip(video_path)
    video_clip = video.subclip(0, VIDEO_CLIP_SIZE)
    audio_clip = video_clip.audio
    splitted_audio = []
    
    # center audio around a single second for each second of the video clip
    for i in range(VIDEO_CLIP_SIZE):
        mid = VIDEO_CLIP_SIZE // 2
        min = i - mid
        max = i + mid

        # padd the audio by centering it around each second
        if min < 0:
            audio_clip_split = audio_clip.subclip(0, max)
            audio_clip_split = audio_clip_split.set_start(abs(min))
            audio_clip_split = audio_clip_split.set_duration(VIDEO_CLIP_SIZE)
        if max > VIDEO_CLIP_SIZE:
            audio_clip_split = audio_clip.subclip(min, VIDEO_CLIP_SIZE)
            audio_clip_split = audio_clip_split.set_duration(VIDEO_CLIP_SIZE)
        else:
            audio_clip_split = audio_clip
        
        audio_clip_split.write_audiofile('temp_audio.wav')
        video_clip.reader.close()
        video_clip.audio.reader.close_proc()

        track, _ = librosa.load('temp_audio.wav', sr=SAMPLE_RATE, dtype=np.float32)
        splitted_audio.append(track)
        audio_clip_split.write_audiofile(f'audios_batch/{basename}_{i}.wav')

    print("Audio is extracted!")

    return splitted_audio, audio_clip

def get_file_list(dir_path):
    """Takes directory path and returns list of paths to each video"""
    file_list = []

    # Iterate through directories
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            file_list.append(os.path.join(dir_path, path))

    print("Video files are read!")
    
    return file_list

def apply_audioclip(audio, images):
    """Takes audio and images and obtains their audioclip embeddings"""
    audio_transforms = ToTensor1D()

    image_transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Resize(IMAGE_SIZE, interpolation=Image.BICUBIC),
        tv.transforms.CenterCrop(IMAGE_SIZE),
        #tv.transforms.Normalize(IMAGE_MEAN, IMAGE_STD)
    ])

    # Input preparation
    audio = torch.stack([audio_transforms(track.reshape(1, -1)) for track in audio]) #[batch x channels x duration]
    images = torch.stack([image_transforms(image) for image in images]) #[batch x channels x height x width]

    # Obtaining embedding
    ((audio_features, _, _), _), _ = aclp(audio=audio)
    ((_, image_features, _), _), _ = aclp(image=images)

    print("Audio and Image features are computed!")

    # Normalization of embeddings
    audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)
    image_features = image_features / torch.linalg.norm(image_features, dim=-1, keepdim=True)

    return audio_features, image_features, images


# Create NeVA Model
NeVA_model = NeVAWrapper(downstream_model=aclp,
                    criterion=criterion,
                    target_function=target_function,
                    image_size=IMAGE_SIZE,
                    foveation_sigma=foveation_sigma,
                    blur_filter_size=blur_filter_size,
                    blur_sigma=blur_sigma,
                    forgetting=forgetting,
                    foveation_aggregation=1,
                    device='cpu') #cuda

scanpaths = []
loss_history = []
batch_size = BATCH_SIZE

video_path = ".\\video_data" # path to video data folder
file_list = get_file_list(video_path)

# write the scan paths and loss into a text file
with open('scan_paths_batch.txt', 'w') as file:
    # iterate over a batch of files from a list of all files
    for i in range(0, len(file_list), batch_size):
        images_per_video_batch = []
        audio_per_video_batch = []
        curr_audio_batch = []
        file_paths = []
        frames_list_batch = []

        # iterate through a batch and extract frames and audio
        for f in file_list[i:i+batch_size]:
            frames, frames_list = extract_frames(f) # gets middle frame of every sec
            audio, curr_audio = extract_audio(f) # gets list of audio of every sec

            images_per_video_batch.append(frames)
            audio_per_video_batch.append(audio)
            curr_audio_batch.append(curr_audio)

            basename = os.path.basename(f)
            file_paths.append(basename)  
            frames_list_batch.extend(frames_list)         


        sublist_scanpath = []
        sublist_loss_history = []

        i = 0      
        idx = 0
        # iterate over extracted videos from a batch of extracted videos
        for sublist_image, sublist_audio in zip(images_per_video_batch, audio_per_video_batch):
            item_scanpath = []
            item_loss_history = []
            file.write(file_paths[i] + '\n')

            video_masked = []
            count = 0 
            # iterate over frames and audio clips of a single video
            for item_image, item_audio in zip(sublist_image, sublist_audio):
                video_split = frames_list_batch[idx:idx+25] #taking frames of a single second, video fps = 25
                # compute embedding
                item_audio = np.concatenate(([item_audio], [item_audio]), axis=0)
                item_image = np.concatenate(([item_image], [item_image]), axis=0)
                image_features, audio_features, image_stack = apply_audioclip(item_audio, item_image)

                filename = file_paths[i][:-4] + "_" + str(count)

                # compute a scan path for each frame
                current_scanpaths, current_loss_history, scan_path = NeVA_model.run_optimization(image_stack[0], audio_features[0], scanpath_length, optimization_steps, lr, filename)
                print("current scanpath: ", current_scanpaths)
                item_scanpath.append(current_scanpaths)
                item_loss_history.append(current_loss_history)

                # draw fixation points and arrow to show saccades in video
                # iterate through frames of 1 second
                for x in video_split:
                    prev_coord = None
                    # iterate through fixations
                    for coord in current_scanpaths: 
                        img = Image.fromarray(x.astype('uint8'), 'RGB')
                        center = (img.width // 2, img.height // 2)
                        dot_coord = (int((coord[0] + 1) * center[0]), int((coord[1] + 1) * center[1]))

                        video_mask = ImageDraw.Draw(img)
                        video_mask.ellipse((dot_coord[0]-3, dot_coord[1]-3, dot_coord[0]+3, dot_coord[1]+3), fill=(0, 255, 0))
                        
                        # calculate coordinates of the previous point and draw an arrow to the current point
                        if prev_coord is not None:
                            prev_dot_coord = (int((prev_coord[0] + 1) * prev_center[0]), int((prev_coord[1] + 1) * prev_center[1]))
                            img = cv2.arrowedLine(np.array(img), prev_dot_coord, dot_coord, (255, 0, 0), 2)
                            
                        prev_coord = coord
                        prev_center = center
                        x = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

                    video_masked.append(x)
                
                # write current scan path and loss history of into file
                file.write("Current scan paths: " + '\n')
                file.write(str(current_scanpaths) + '\n')
                file.write("Current loss history: " + '\n')
                file.write(str(current_loss_history) + '\n')

                count = count + 1
                idx = idx+25
                
            # convert frames to video and save the video
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(video_masked, fps=25)
            clip = clip.set_audio(curr_audio_batch[i])

            output_video_name = "output_videos/output_" + file_paths[i][:-4] + ".mp4"
            clip.write_videofile(output_video_name)

            sublist_scanpath.append(item_scanpath)
            sublist_loss_history.append(item_loss_history)
            i += 1

        scanpaths.extend(sublist_scanpath)
        loss_history.extend(sublist_loss_history)


file.close()