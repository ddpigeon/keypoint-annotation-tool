import os
import cv2
import tkinter as tk
from tkinter import Canvas, Button, Label, Toplevel, Frame
import math
from ultralytics import YOLO


colors = [(0,0,0),(255,255,255),(128,128,128),
          (139,69,19),(244,164,96),(0,100,0),
          (0,0,139),(0,128,0), (0,0,255),
          (0,255,0) ,(30,144,255),  (139,0,0),
          (128,0,128),(255,0,0),(255,20,147),
          (255,127,80) , (255,192,203) , (173,255,47),
          (135,206,250),(0,128,128),(255,215,0),
          (216,191,216), (255, 0, 215), (0, 215, 255),
          (250, 135, 100), (100, 15, 200), (13, 220, 100)

          ]

parts = {}
possible_connections = {}


# Add new model keypoint names and connections like this

# 17 keypoint model
parts["17"] = [
    "Nose", "Left eye", "Right eye", "Left ear", "Right ear",
    "left shoulder", "right shoulder", "left elbow", "right elbow", "left wrist", "right wrist", 
    "left hip", "right hip", "left knee", "right knee", "left ankle", "right ankle"
]

possible_connections["17"] = [
        (0, 1), (0, 2), (1, 3), (2, 4), # Face
        (5, 6), (11, 12), (5, 11), (6, 12),  # Torso
        (5, 7), (7, 9), # Left arm
        (6, 8), (8, 10), # Right arm
        (11, 13), (13, 15), # Left leg
        (12, 14), (14, 16) # Right leg
]


# 22 keypoint model
parts["22"] = [
    "left eye", "right eye", "nose", "left ear", "right ear",
    "left shoulder", "right shoulder", "left elbow", "right elbow",
    "left wrist", "right wrist", "left hip",
    "right hip", "left knee", "right knee", "left ankle", "right ankle",
    "left finger", "right finger", "back", "left toe", "right toe"
]

possible_connections["22"] = [
    (0, 1), (1, 2), (0, 2),  # Left eye, right eye, nose
    (0, 3), (1, 4),                # Left ear, right ear
    (5, 7), (7 ,9), (9,17)  ,         # Left Arm
    (6, 8), (8, 10), (10 ,18) ,          # Right Arm
    (11, 13), (13, 15), (15 , 20) ,      # left Leg 
    (12, 14), (14, 16), (16 , 21) ,      # Left knee, right knee
    (11 , 19) , (12 , 19) , (5 , 19) , (6, 19)
]

# 26 Keypoint model
parts["26"] = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "left shoulder", "right shoulder", "left elbow", "right elbow", "left wrist", "right wrist", 
    "left hip", "right hip", "left knee", "right knee", "left ankle", "right ankle",
    "Head", "Neck", "Hip",
    "Left big toe", "Right big toe", "Left small toe", "Right small toe",
    "Left Heel", "Right Heel"
]

possible_connections["26"] = [
        (0, 1), (0, 2), (1, 3), (2, 4), # Face
        (17, 18), (18, 5), (18, 6), (18, 19), (19, 11), (19, 12),  # Torso
        (5, 7), (7, 9), # Left arm
        (6, 8), (8, 10), # Right arm
        (11, 13), (13, 15), # Left leg
        (12, 14), (14, 16), # Right leg
        (15, 24), (24, 20), (24, 22), # Left foot
        (16, 25), (25, 21), (25, 23) # Right foot
]

# 27 Keypoint model
parts["27"] = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "left shoulder", "right shoulder", "left elbow", "right elbow", "left wrist", "right wrist", 
    "left hip", "right hip", "left knee", "right knee", "left ankle", "right ankle",
    "Head", "Neck", "Hip",
    "Left big toe", "Right big toe", "Left small toe", "Right small toe",
    "Left Heel", "Right Heel",
    "Middle back"
]

possible_connections["27"] = [
        (0, 1), (0, 2), (1, 3), (2, 4), (17, 18), # Face
        (26, 18), (18, 5), (18, 6), (26, 19), (19, 11), (19, 12),  # Torso
        (5, 7), (7, 9), # Left arm
        (6, 8), (8, 10), # Right arm
        (11, 13), (13, 15), # Left leg
        (12, 14), (14, 16), # Right leg
        (15, 24), (24, 20), (24, 22), # Left foot
        (16, 25), (25, 21), (25, 23) # Right foot
]

class KeypointEditor:
    def __init__(self, root,folder_path, model_type):
        # initial_points = initial_points[0]
        self.root = root
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_size = 256
        self.image_ac_size = None
        self.is_initial = True  # This is true if initial points predicted by model are not changed, otherwise becomes false
        # self.image_path = image_path

        self.bbox = [] # bounding box coordinates on canvas
        self.bbox_n = [None] * 4 # Normalized yolo format bbox coords
        self.selected_bbox = None


        self.keypoint_count = int(model_type)
        self.initial_points = []
        self.final_points = []  # Convert to lists
        self.keypoint_conf = [float(1)] * self.keypoint_count
        self.current_conf = float(1)
        self.selected_point = None


        self.current_image_index = 0
        self.labels_folder = ""
        self.save_path = ""
        self.scale_factor = 2.5  # Scale factor for both image and points
        # image_path = os.path.join(self.folder_path, self.image_files[self.current_image_index])

        # self.label = Label(root, text="File Name: {}".format(os.path.basename(image_path)))
        # self.label.pack()
        self.label = Label( root ,text="")
        self.label.pack()
        self.message_label = Label(root, text="")
        self.messagesave_label = Label(root , text="j")
        self.size_label = Label(root, text="")
        self.size_label.pack()

        self.message_label.pack()
        self.messagesave_label.pack()
        self.canvas = Canvas(root)
        self.canvas.pack()
        self.connections = possible_connections[model_type]
        self.load_image()
        self.draw_connections(self.connections)
        self.draw_keypoints()
        self.draw_bbox()
        self.check_file()

        self.save_button = Button(root, text="Save", command=self.save_coordinates)
        self.save_button.pack()
        self.toggle_conf_button = Button(root, text="Toggle confidence", command=self.toggle_confidence)
        self.toggle_conf_button.pack()
        self.next10_button = Button(root, text="10 >>", command=self.next_10)
        self.next10_button.pack(side=tk.RIGHT)
        self.prev10_button = Button(root, text="<< 10", command=self.prev_10)
        self.prev10_button.pack(side=tk.LEFT)
        self.next_button = Button(root, text="Next", command=self.next_image)
        self.prev_button = Button(root, text="Previous", command=self.prev_image)
        self.delete_button = Button(root, text="Delete", command=self.delete_current_image)

        self.pack_buttons()

        self.canvas.bind("<Button-1>", self.select_point)
        self.canvas.bind("<B1-Motion>", self.move_selected_point)

    def pack_buttons(self):
        self.next_button.pack(side=tk.RIGHT)
        self.prev_button.pack(side=tk.LEFT)
        self.delete_button.pack()

    def load_image(self):
        image_path1 = os.path.join(self.folder_path, self.image_files[self.current_image_index])
        self.label.config(text="File Name: {}".format(os.path.basename(image_path1)))
        image = cv2.imread(image_path1)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        results = model(image)
        
        # if results and results[0].keypoints is not None:
        if self.is_initial: 
            self.initial_points = results[0].keypoints.xy.tolist()
            self.keypoint_conf = [float(1)] * self.keypoint_count

            self.bbox = results[0].boxes.xyxy.tolist()[0]
            self.bboxn = results[0].boxes.xywhn.tolist()[0]
            self.bbox = [i * self.scale_factor for i in self.bbox]

            initial_points = self.initial_points[0]
            self.final_points = [list(point) for point in initial_points]
            if self.keypoint_count == 27:
                self.final_points.append([float(self.image_size/2), float(self.image_size/2)])

            self.is_initial = False

        self.convert_bbox()
        print([i // self.scale_factor for i in self.bbox])
        print(self.bbox_n)
        #self.image = cv2.imread(image_path1)
        self.image = image
        image_size = self.image.shape
        self.size_label.config(text=f"Image_ac_size {image_size}")
        # self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image = cv2.resize(self.image, None, fx=self.scale_factor, fy=self.scale_factor)  # Scale up the image
        self.photo = tk.PhotoImage(data=cv2.imencode('.png', self.image)[1].tobytes())
        self.canvas.config(width=self.image.shape[1], height=self.image.shape[0])  # Set canvas size to match image size
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def draw_keypoints(self):
        for i, (x, y) in enumerate(self.final_points):
            x_scaled, y_scaled = x * self.scale_factor, y * self.scale_factor  # Scale up the points
            color = colors[i % len(colors)]
            outline_color = "#%02x%02x%02x" % color
            if self.keypoint_conf[i] < 0.5:
                outline_color = "black"
            self.canvas.create_oval(x_scaled - 5, y_scaled - 5, x_scaled + 5, y_scaled + 5,
                                    fill="#%02x%02x%02x" % color, outline=outline_color, width=2)
            

    def draw_bbox(self):
        self.canvas.create_rectangle(self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], outline="red", width=3)
        self.canvas.create_oval(self.bbox[0]-5, self.bbox[1]-5, self.bbox[0]+5, self.bbox[1]+5,fill="red", outline="red")
        self.canvas.create_oval(self.bbox[2]-5, self.bbox[3]-5, self.bbox[2]+5, self.bbox[3]+5,fill="red", outline="red")


    def convert_bbox(self):
        tmp_bbox = [i // self.scale_factor for i in self.bbox]
        self.bbox_n[0] = (tmp_bbox[0] + tmp_bbox[2]) / 2
        self.bbox_n[1] = (tmp_bbox[1] + tmp_bbox[3]) / 2
        self.bbox_n[2] = tmp_bbox[2] - tmp_bbox[0]
        self.bbox_n[3] = tmp_bbox[3] - tmp_bbox[1]
            

    def draw_connections(self, connections):
        for connection in connections:
            i, j = connection
            x1, y1 = self.final_points[i]
            x2, y2 = self.final_points[j]
            x1_scaled, y1_scaled = x1 * self.scale_factor, y1 * self.scale_factor
            x2_scaled, y2_scaled = x2 * self.scale_factor, y2 * self.scale_factor
            self.canvas.create_line(x1_scaled, y1_scaled, x2_scaled, y2_scaled, fill="white", width=3)

    def select_point(self, event):
        self.selected_point = None

        x, y = event.x // self.scale_factor, event.y // self.scale_factor  # Scale down the mouse click coordinates
        for i, (px, py) in enumerate(self.final_points):
            distance = math.sqrt((px - x)**2 + (py - y)**2)
            if distance < 5:
                self.selected_point = i
                self.selected_bbox = None
                #break

        if self.selected_point == None:
            for i in range(0, 2):
                px = self.bbox[2 * i] // self.scale_factor
                py = self.bbox[2 * i + 1] // self.scale_factor
                distance = math.sqrt((px - x) ** 2 + (py - y) ** 2)
                if distance < 5:
                    self.selected_bbox = i
                    break


    def move_selected_point(self, event):
        if self.selected_point is not None:
            self.selected_bbox = None
            x, y = event.x // self.scale_factor, event.y // self.scale_factor  # Scale down the mouse move coordinates
            self.final_points[self.selected_point] = [int(x), int(y)]  # Convert to integers and update as list
            self.keypoint_conf[self.selected_point] = self.current_conf
            self.redraw_keypoints()

        if self.selected_bbox is not None:
            self.selected_point = None
            x, y = event.x, event.y
            self.bbox[2 * self.selected_bbox] = x
            self.bbox[2 * self.selected_bbox + 1] = y
            self.redraw_keypoints()


    def redraw_keypoints(self):
        self.canvas.delete("all")
        self.load_image()
        self.redraw_connections()
        self.draw_keypoints()
        self.draw_bbox()

    def redraw_connections(self):
        self.canvas.delete("connections")
        self.draw_connections(self.connections)
    

    def save_coordinates(self):
        # Ensure all points in final_points are integers
        self.final_points = [[int(px), int(py)] for px, py in self.final_points]

        # Normalize the points to the original image size, add confidence scores
        normalized_points = []
        for i, p in enumerate(self.final_points):
            normalized_points.append((p[0] / self.image_size, p[1] / self.image_size, self.keypoint_conf[i]))

        # Create a formatted string for saving
        save_string = "0 " + " ".join([str(i / self.image_size) for i in self.bbox_n]) + " " + " ".join([f"{x} {y} {c}" for x, y, c in normalized_points])

        # Create a 'labels' folder if it doesn't exist
        image_path = os.path.join(self.folder_path, self.image_files[self.current_image_index])

        labels_folder = os.path.join(os.path.dirname(image_path), "labels")
        os.makedirs(labels_folder, exist_ok=True)

        # Save to a .txt file inside the 'labels' folder
        save_filename = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        save_path = os.path.join(labels_folder, save_filename)
        
          # Update the message label
        with open(save_path, "w") as file:
            file.write(save_string)

        # Update the message label
        self.message_label.config(text=f"Saved coordinates to {save_path}")

    def next_image(self):
        # print("Next button pressed")
        self.current_image_index = (self.current_image_index + 1) % len(self.image_files)
        self.is_initial = True
        self.load_image()
        self.draw_connections(self.connections)
        self.draw_keypoints()
        self.check_file()
        self.message_label.config(text="")


    def next_10(self):
        self.current_image_index = (self.current_image_index + 10) % len(self.image_files)
        self.is_initial = True
        self.load_image()
        self.draw_connections(self.connections)
        self.draw_keypoints()
        self.check_file()
        self.message_label.config(text="")
        

    def prev_image(self):
        self.current_image_index = (self.current_image_index - 1) % len(self.image_files)
        if self.current_image_index < 0:
            self.current_image_index = len(self.image_files) - 1
        self.is_initial = True
        self.load_image()
        self.draw_connections(self.connections)
        self.draw_keypoints()
        self.message_label.config(text="")
        self.check_file()


    def prev_10(self):
        self.current_image_index = (self.current_image_index - 10) % len(self.image_files)
        self.is_initial = True
        self.load_image()
        self.draw_connections(self.connections)
        self.draw_keypoints()
        self.check_file()
        self.message_label.config(text="")

    def check_file(self):
        image_path = os.path.join(self.folder_path, self.image_files[self.current_image_index])

        labels_folder = os.path.join(os.path.dirname(image_path), "labels")
        os.makedirs(labels_folder, exist_ok=True)
        save_filename = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        save_path = os.path.join(labels_folder, save_filename)
        txt_files = os.listdir(labels_folder)
        # jpg_files = os.listdir(self.folder_path)
        jpg_file = self.image_files[self.current_image_index]
        if jpg_file[:-4] not in [f.split('.')[0] for f in txt_files if f.endswith('.txt')]:
            self.messagesave_label.config(text = f"coordinates not there {jpg_file}")
        else :
            self.messagesave_label.config(text = f"coordinates saved {jpg_file}" )


    def delete_current_image(self):
        file_path_d = os.path.join(self.folder_path, self.image_files[self.current_image_index])
        if self.current_image_index < len(self.image_files):
            os.remove(os.path.join(self.folder_path, self.image_files[self.current_image_index]))
            self.image_files.pop(self.current_image_index)

        if self.current_image_index >= len(self.image_files):
            self.current_image_index = 0

        self.load_image()
        self.draw_connections(self.connections)
        self.draw_keypoints()
        self.message_label.config(text=f" {file_path_d} file deleted")


    def toggle_confidence(self):
        # Change confidence scores of keypoints
        # Use 0.05 for keypoints which are hidden/outside the image
        # keypoints with 0.05 confidence will have a black border
         
        if self.current_conf == 1:
            self.current_conf = 0.05
        else:
            self.current_conf = float(1)

    # @property
    # def image_files(self):
    #     return [f for f in os.listdir(self.folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def display_parts_colors(model_type):
    part_colors_window = Toplevel()
    part_colors_window.title("Parts and Colors")

    for i in range(len(parts[model_type])):
        color = colors[i % len(colors)]
        part_label = Label(part_colors_window, text=parts[model_type][i], fg="#%02x%02x%02x" % color)
        part_label.grid(row=i, column=0, sticky="w")

        color_label = Label(part_colors_window, text="#%02x%02x%02x" % color)
        color_label.grid(row=i, column=1, sticky="e")


def main():
    folder_path = "trial2" #Change this to wherever images are
    model_type = "27"   #Model type, can be 17, 22 or 26 for now - Number of keypoints
                        # 27 is 26 model, with an added 27th back keypoint in annotations which has to always be fixed manually
    global model 

    if model_type == "17":
        model = YOLO("yolov8s-pose.pt")

    if model_type == "22":
        model = YOLO("weights/best.pt")

    if model_type == "26":
        model = YOLO("weights/best26.pt")

    if model_type == "27":
        model = YOLO("weights/best26.pt")


    # image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # for image_file in image_files:
    #     image_path = os.path.join(folder_path, image_file)
    #     image = cv2.resize(cv2.imread(image_path), (256, 256))
    #     results = model(image)
        
    #     # if results and results[0].keypoints is not None:
    #     initial_points = results[0].keypoints.xy.tolist()
        # else:
            # print("Error: results is empty or keypoints is not defined)
    root = tk.Tk()
    root.title("Keypoint Editor")
    editor = KeypointEditor(root, folder_path, model_type)
    parts_button = Button(root, text="Display Parts and Colors", command=display_parts_colors(model_type))
        # delete_btn = Button(root, text="delete all" , command=redraw_keypoints )
    parts_button.pack()
    root.mainloop()

if __name__ == "__main__":
    main()
