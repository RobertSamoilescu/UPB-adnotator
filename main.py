import cv2
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import pickle as pkl


parser = argparse.ArgumentParser()
parser.add_argument("--src_dir", type=str, help="source directory with videos",
	default="../upb_dataset")
parser.add_argument("--dst_dir", type=str, help="destination directory with videos",
	default="./processed_videos")
parser.add_argument("--signs_dir", type=str, help="directory with signs",
	default="./signs")
args = parser.parse_args()


def parse_video(video_path: str, signs: list):
	cap = cv2.VideoCapture(video_path)
	frames = []
	directions = []
	concat_frames = []

	# read first frame
	ret, frame = cap.read()
	frames.append(frame)
	directions.append(None)
	concat_frames.append(None)
	
	frame_idx = 0
	sign_idx = 0
	fig = plt.figure()


	while True:
		# display current frame unde 
		f = cv2.resize(frames[frame_idx], (512, 256))
		s = cv2.resize(signs[sign_idx], (256, 256))
		disp_img = np.concatenate([f, s], axis=1)
		cv2.imshow("Img", disp_img)
		key = cv2.waitKey(0)
		
		if key == ord('q'):
			break
		elif key == ord('a'):
			frame_idx -= 1
			frame_idx = np.clip(frame_idx, 0, len(frames) - 1)
			sign_idx = directions[frame_idx]

			# additional check
			if sign_idx is None:
				sign_idx = 0

		elif key == ord('d'):
			frame_idx += 1
			frame_idx = np.clip(frame_idx, 0, len(frames) - 1)
			sign_idx = directions[frame_idx]
			
			# additional check here
			if sign_idx is None:
				sign_idx = 0

		elif key == ord('k'):
			sign_idx -= 1
			sign_idx = np.clip(sign_idx, 0, len(signs) - 1)
		elif key == ord('l'): 
			sign_idx += 1
			sign_idx = np.clip(sign_idx, 0, len(signs) - 1)
		elif key == 32: # Space key 
			directions[frame_idx] = sign_idx
			concat_frames[frame_idx] = disp_img

			# going back in time
			if frame_idx < len(frames) - 1:
				frame_idx += 1
				sign_idx = 0
			else:
				# read next image
				ret, frame = cap.read()
				if not ret:
					print("Video has finished")
				else:
					frames.append(frame)
					directions.append(None)
					concat_frames.append(None)
					frame_idx += 1
					sign_idx = 0

		# log current state
		print("Video:", video_path)
		print("Current Index:", frame_idx)
		print("Indices:\t", list(np.arange(max(0, frame_idx - 3), min(frame_idx + 2, len(frames)))))
		print("Directions:\t", directions[max(0, frame_idx - 3):min(frame_idx + 2, len(frames))])
		print("----------------------\n\n")


	# save images and directions
	video_name = video_path.split("/")[-1][:-4]
	dir_path = os.path.join(args.dst_dir, video_name)
	if not os.path.exists(dir_path):
		os.makedirs(dir_path)

	for i, img in enumerate(list(concat_frames)):
		path = os.path.join(dir_path, str(i).zfill(5) + ".png")
		cv2.imwrite(path, img)

	path = os.path.join(args.dst_dir, "directions_" + video_name[:-4] + ".pkl")
	with open(path, "wb") as fout:
		pkl.dump(directions, fout)


if __name__ == "__main__":
	# create destination directory
	if not os.path.exists(args.dst_dir):
		os.makedirs(args.dst_dir)

	# read processed videos
	processed_videos = os.listdir(args.dst_dir)
	processed_videos = [v for v in processed_videos if not v.endswith(".pkl")]

	# all videos
	all_videos = os.listdir(args.src_dir)
	all_videos = [v[:-4] for v in all_videos if v.endswith(".mov")]

	# remaining videos
	remaining_videos = list(set(all_videos).difference(processed_videos))
	print("All videos: %d" % (len(all_videos)))
	print("Processed videos: %d" % (len(processed_videos)))
	print("Remaining videos: %d" % (len(remaining_videos)))
	print("-----------------------------------------------")

	# read signs
	signs_path = os.listdir(args.signs_dir)
	signs_path = [os.path.join(args.signs_dir, s) for s in signs_path]
	signs_path = sorted(signs_path)
	signs = [cv2.imread(s) for s in signs_path]
	del signs[-1]


	# process each vidoe
	for v in remaining_videos:
		path = os.path.join(args.src_dir, v + ".mov")
		parse_video(path, signs)