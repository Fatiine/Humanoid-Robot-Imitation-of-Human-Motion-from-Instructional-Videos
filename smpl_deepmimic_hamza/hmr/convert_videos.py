
import time
import cv2
import os

def video_to_frames(input_loc, output_loc):
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video..\n")
    # Start converting the video

    i = 0
    freq = 2


    success,image = cap.read()
    count = 0
    while success:
        if count%freq == 0:
            cv2.imwrite(output_loc + "/%#05d.jpg" % (i + 1), image)     # save frame as JPEG file 
            i = i + 1 
        success,image = cap.read()
        print('Read a new frame: ', success)
        count += 1


    # while cap.isOpened():
    #     # Extract the frame
    #     ret, frame = cap.read()
    #     if count%freq == 0:
    #         # Write the results back to output location.
    #         cv2.imwrite(output_loc + "/%#05d.jpg" % (i + 1), frame)
    #         i = i + 1
    #     count = count + 1
    #     # If there are no more frames left
    #     if (count > (video_length-1)):
    #         # Log the time again
    #         time_end = time.time()
    #         # Release the feed
    #         cap.release()
    #         # Print stats
    #         print ("Done extracting frames.\n%d frames extracted" % count)
    #         print ("It took %d seconds forconversion." % (time_end-time_start))
    #         break

if __name__ == "__main__":
    name = "videos/MoveGravity.m4v"
    video_to_frames(name, "videos/MoveGravity/")