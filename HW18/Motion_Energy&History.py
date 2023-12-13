import cv2
import numpy as np

def get_frames(video_path:str):
    capture = cv2.VideoCapture(video_path)
    num_of_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    frames = []
    for _ in range(int(num_of_frames) - 1):
        _, frame = capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frames.append(frame)
    return len(frames), frames

def process(video_path:str):
    num_of_frames, frames = get_frames(video_path)
    frames_difference = []
    Motion_Energy_Graph = []
    Motion_History_Graph = []
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    width, height = np.array(frames[0]).shape
    video_for_Motion_Energy_Graph = cv2.VideoWriter(
        "Motion_Energy_Graph_for_"+video_path[:-4]+".avi",
        fourcc,
        30,
        (width, height),
        isColor=False)
    video_for_Motion_History_Graph = cv2.VideoWriter(
        "Motion_History_Graph_for_"+video_path[:-4]+".avi",
        fourcc,
        30,
        (width, height),
        isColor=False)
    for i in range(num_of_frames - 1):
        _, frame_difference = cv2.threshold(cv2.absdiff(frames[i], frames[i+1]), 50, 255, cv2.THRESH_BINARY)
        # print(frame_difference)
        if i == 0:
            frame_for_motion_energy_graph = np.array(frame_difference)
            frame_for_motion_history_graph = np.array(frame_difference / 255 * num_of_frames)
        else:
            _, frame_for_motion_energy_graph = cv2.threshold(frame_difference | Motion_Energy_Graph[-1], 0.5, 255, cv2.THRESH_BINARY)
            frame_for_motion_energy_graph = np.array(frame_for_motion_energy_graph)

            frame_for_motion_history_graph = np.array(frame_difference)
            print(frame_for_motion_history_graph.shape)
            for x in range(width):
                for y in range(height):
                    if frame_difference[x][y]:
                        frame_for_motion_history_graph[x][y] = num_of_frames
                    elif Motion_History_Graph[-1][x][y]:
                        frame_for_motion_history_graph[x][y] = Motion_History_Graph[-1][x][y] - 1

        print("1:", np.shape(Motion_Energy_Graph))
        print("2:", np.shape(Motion_History_Graph))
        print("3:", np.shape(frame_for_motion_energy_graph))
        print("4:", np.shape(frame_for_motion_history_graph))
        Motion_Energy_Graph.append(frame_for_motion_energy_graph)
        Motion_History_Graph.append(frame_for_motion_history_graph)
        frames_difference.append(frame_difference)
        video_for_Motion_Energy_Graph.write(frame_for_motion_energy_graph)
        video_for_Motion_History_Graph.write(frame_for_motion_history_graph)
        cv2.imwrite("Pic/"+video_path[:-4]+"_Energy_"+str(i)+".png", frame_for_motion_energy_graph)
        cv2.imwrite("Pic/"+video_path[:-4]+"_History_"+str(i)+".png", frame_for_motion_history_graph)
        #cv2.imshow("Motion Energy Graph", frame_for_motion_history_graph)
        #cv2.waitKey()

    video_for_Motion_Energy_Graph.release()
    video_for_Motion_History_Graph.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process("Stand_Up.mp4")
    process("Sit_Down.mp4")