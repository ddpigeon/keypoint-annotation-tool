import tkinter as tk
from tkinter import filedialog
import cv2
import os
from PIL import Image, ImageTk
from keypoint_annotation import KeypointEditor
from ultralytics import YOLO

model = YOLO(r"weights/last.pt")


class VideoGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Video GUI")
        self.geometry("400x300")
        self.configure(bg="#f0f0f0")  # Set background color
        self.canvas = tk.Canvas(self, width=256, height=256)
        self.canvas.pack()

        # Create labels and buttons with improved styling
        self.video_label = tk.Label(
            self, text="Select a video file:", font=("Arial", 12), bg="#f0f0f0"
        )
        self.video_label.pack(pady=10)
        self.video_button = tk.Button(
            self, text="Browse...", command=self.select_video, font=("Arial", 12)
        )
        self.video_button.pack(pady=5)

        self.folder_label = tk.Label(
            self, text="Frames folder:", font=("Arial", 12), bg="#f0f0f0"
        )
        self.folder_label.pack(pady=10)
        self.folder_button = tk.Button(
            self, text="Browse...", command=self.select_folder, font=("Arial", 12)
        )
        self.folder_button.pack(pady=5)

        self.save_button = tk.Button(
            self, text="Save Frames", command=self.save_frames, font=("Arial", 12)
        )
        self.save_button.pack(pady=10)
        
        
        

        # Initialize variables
        self.video_path = ""
        self.fps = 1
        self.folder_path = None
        self.video_player = cv2.VideoCapture(self.video_path)
        self.video_width = int(self.video_player.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.video_player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.config(width=self.video_width, height=self.video_height)

        # self.show_video()

    def select_video(self):
        # Open a file dialog for selecting a video file
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        self.video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.video_label.config(text=f"Selected video file: {self.video_path}")

    def select_folder(self):
        # Open a folder dialog for selecting the frames folder
        self.folder_path = filedialog.askdirectory()
        self.folder_label.config(text=f"Frames folder: {self.folder_path}")


    # def show_video(self):
    #     ret, frame = self.video_player.read()

    #     if ret:
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))  # Make photo an instance variable
    #         self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
    #         self.canvas.image = self.photo  # Keep a reference to the image to prevent garbage collection

    #         self.after(10, self.show_video)  # Schedule the next frame update
    #     else:
    #         self.video_player.release()
    #         self.canvas.delete("all")

    def save_frames(self):
        # Get the number of frames per second from the entry
        # self.fps = int(self.fps_entry.get())

        # Open the video file
        video = cv2.VideoCapture(self.video_path)

        # Check if the frames folder exists
        if not os.path.exists(self.folder_path):
            # Create the frames folder if it doesn't exist
            os.makedirs(self.folder_path)

        # Get the total number of frames in the video
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        j=1015
        # Loop through the frames and save each one
        for i in range( 0 ,total_frames,3 ):
            # Move the video to the next frame
            video.grab()
            
            # Read the current frame
            ret, frame = video.read()
            
            # Check if the frame was read successfully
            if ret:
                # Save the frame to the frames folder
                frame = cv2.resize(frame, (256, 256))
                frame_name = self.video_name
                frame_path = os.path.join(self.folder_path, f"{frame_name}_{j}.jpg")
                cv2.imwrite(frame_path, frame)
                j=j+1
                # Print a status message
                print(f"Saved frame {i} to {frame_path}")
              

            # Break out of the loop if the user presses the "Save Frames" button again
            if not self.save_button.winfo_exists():
                break

        # Release the video file
        video.release()
    

if __name__ == "__main__":
    app = VideoGUI()
    app.mainloop()