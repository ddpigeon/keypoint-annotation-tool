import os
import cv2
import tkinter as tk
from tkinter import Canvas, Button, Label, Toplevel, Frame
import math
from ultralytics import YOLO

colors = [
    (0, 0, 0),       # Black (nose)
    (255, 0, 0),     # Red (left eye)
    (0, 0, 255),     # Blue (right eye)
    (255, 0, 0),     # Red (left ear)
    (0, 0, 255),     # Blue (right ear)
    (255, 128, 0),   # Orange (left shoulder)
    (0, 128, 255),   # Sky blue (right shoulder)
    (255, 128, 0),   # Orange (left elbow)
    (0, 128, 255),   # Sky blue (right elbow)
    (255, 128, 0),   # Orange (left wrist)
    (0, 128, 255),   # Sky blue (right wrist)
    (255, 255, 0),   # Yellow (left hip)
    (0, 255, 255),   # Cyan (right hip)
    (255, 255, 0),   # Yellow (left knee)
    (0, 255, 255),   # Cyan (right knee)
    (255, 255, 0),   # Yellow (left ankle)
    (0, 255, 255),   # Cyan (right ankle)
    (128, 0, 128),   # Purple (head)
    (128, 128, 128), # Gray (neck)
    (0, 255, 0),     # Green (mid back)
    (0, 255, 0),     # Green (lower back)
    (0, 255, 0),     # Green (upper back)
    (255, 128, 0),   # Orange (left palm end)
    (0, 128, 255),   # Sky blue (right palm end)
    (255, 255, 0),   # Yellow (left foot end)
    (0, 255, 255)    # Cyan (right foot end)
]

parts= [    
    "nose", "left eye", "right eye", "left ear", "right ear", # 0, 1, 2, 3, 4
    "left shoulder", "right shoulder", "left elbow", "right elbow", "left wrist", "right wrist", # 5, 6, 7, 8, 9, 10
    "left hip", "right hip", "left knee", "right knee", "left ankle", "right ankle", # 11, 12, 13, 14, 15, 16
    "head", "neck", # 17, 18
    "mid back", "lower back", "upper back", # 19, 20, 21
    "left palm end","right palm end","left foot end","right foot end" # 22, 23, 24, 25
]

possible_connections= [
        (0, 1), (0, 2), (1, 3), (2, 4), # Face
        (5, 7), (7, 9), # Left arm
        (6, 8), (8, 10), # Right arm
        (11, 13), (13, 15), # Left leg
        (12, 14), (14, 16), # Right leg
        (17, 18), (18, 21), (19, 21), (19, 20), # Back
        (20, 11), (20, 12), # Waist
        (15,24), (16,25), # ankle to feet end
        (9,22), (10,23) # wrist to palm end
]

def display_parts_colors(model_type):
    part_colors_window = Toplevel()
    part_colors_window.title("Parts and Colors")
    for i in range(len(parts)):
        color = colors[i % len(colors)]
        part_label = Label(part_colors_window, text=parts[i], fg="#%02x%02x%02x" % color)
        part_label.grid(row=i, column=0, sticky="w")

        color_label = Label(part_colors_window, text="#%02x%02x%02x" % color)
        color_label.grid(row=i, column=1, sticky="e")

# Function to adjust keypoints to be within image boundaries
def adjust_keypoints(keypoints,image, margin=5):
    img_width, img_height,_ = image.shape
    adjusted_keypoints = []
    for (x, y) in keypoints:
        if x < margin:
            x = margin
        elif x > img_width - margin:
            x = img_width - margin
        if y < margin:
            y = margin
        elif y > img_height - margin:
            y = img_height - margin
        adjusted_keypoints.append([x, y])
    return adjusted_keypoints

# Correct and adjust keypoints
def correct_keypoints(full_keypoints, image):
    if not full_keypoints: 
        print("No keypoints found!!")
        return None
    # print("here you go: ", full_keypoints)
    full_keypoints[21][0] = (full_keypoints[5][0] + full_keypoints[6][0]) / 2
    full_keypoints[21][1] = (full_keypoints[5][1] + full_keypoints[6][1]) / 2

    full_keypoints[20][0] = full_keypoints[19][0]
    full_keypoints[20][1] = full_keypoints[19][1]

    full_keypoints[19][0] = (full_keypoints[20][0] + full_keypoints[21][0]) / 2
    full_keypoints[19][1] = (full_keypoints[20][1] + full_keypoints[21][1]) / 2

    full_keypoints[22][0] = full_keypoints[9][0]
    full_keypoints[22][1] = full_keypoints[9][1] + 10

    full_keypoints[23][0] = full_keypoints[10][0]
    full_keypoints[23][1] = full_keypoints[10][1] + 10

    full_keypoints[24][0] = full_keypoints[15][0] + 10
    full_keypoints[24][1] = full_keypoints[15][1] + 5

    full_keypoints[25][0] = full_keypoints[16][0] + 10
    full_keypoints[25][1] = full_keypoints[16][1] + 5
    full_keypoints = adjust_keypoints(full_keypoints,image)
    return full_keypoints

class KeypointEditor:
    def __init__(self, root,folder_path, model_type):
        # initial_points = initial_points[0]
        self.root = root
        self.counter = 0
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_size = 256
        self.root.geometry("800x900")
        self.image_ac_size = None
        self.is_initial = True  # This is true if initial points predicted by model are not changed, otherwise becomes false
        # self.image_path = image_path

        self.bbox = [] # bounding box coordinates on canvas
        self.bbox_n = [None] * 4 # Normalized yolo format bbox coords
        self.selected_bbox = None
        self.is_annotation_loaded = False

        self.keypoint_count = int(model_type)
        self.initial_points = []
        self.final_points = []  # Convert to lists
        self.keypoint_conf = [float(1)] * self.keypoint_count
        self.current_conf = float(1)
        self.selected_point = None
        self.keypoint_radius = 3

        self.current_image_index = 0
        self.labels_folder = ""
        self.save_path = ""
        self.scale_factor = 2.5

        self.label = Label( root ,text="")
        self.label.pack()
        self.message_label = Label(root, text="")
        self.messagesave_label = Label(root , text="j")
        self.size_label = Label(root, text="")
        self.size_label.pack()
        self.confidence_label = Label(root, text="Confidence: High (1.0)")
        self.confidence_label.pack()

        self.message_label.pack()
        self.messagesave_label.pack()
        self.canvas = Canvas(root)
        self.canvas.pack()
        self.connections = possible_connections
        self.load_image()
        self.draw_connections(self.connections)
        self.draw_keypoints()
        self.draw_bbox()
        self.check_file()

        self.save_button = Button(root, text="Save(D)", command=self.save_coordinates)
        self.save_button.pack()
        self.toggle_conf_button = Button(root, text="Toggle confidence", command=self.toggle_confidence)
        self.toggle_conf_button.pack()
        self.next_button = Button(root, text="Next(R)", command=self.next_image)
        self.prev_button = Button(root, text="Previous(L)", command=self.prev_image)
        self.delete_button = Button(root, text="Delete (U)", command=self.delete_current_image)
        self.pack_buttons()
        self.canvas.bind("<Button-1>", self.select_point)
        self.canvas.bind("<B1-Motion>", self.move_selected_point)
        self.root.bind("<Control_L>", self.toggle_confidence)
        self.root.bind("<Key>", self.handle_key_press)
        

        

    def handle_key_press(self, event):
        if event.keysym == 'Left':
            self.prev_image()
        elif event.keysym == 'Right':
            self.next_image()
        elif event.keysym == 'Down':
            self.save_coordinates()
        elif event.keysym == 'Up':
            self.delete_current_image()
        elif event.keysym == 'space':
            self.toggle_confidence()

    def pack_buttons(self):
        self.next_button.pack(side=tk.RIGHT)
        self.prev_button.pack(side=tk.LEFT)
        self.delete_button.pack()

    def load_annotation(self, image_path):
        labels_folder = os.path.join(os.path.dirname(image_path), "labels")
        annotation_file = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        annotation_path = os.path.join(labels_folder, annotation_file)

        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                annotation = f.read().strip().split()
            print("Annotation loaded: ", annotation_path)
            self.is_annotation_loaded=True
            self.bbox_n = [float(x) for x in annotation[1:5]]
            x_center, y_center, width, height = self.bbox_n[0],self.bbox_n[1], self.bbox_n[2], self.bbox_n[3]
            self.bbox_n = [i * self.image_size for i in self.bbox_n]
            self.reverse_convert_bbox_n_to_bbox_during_annotation_loading()
            # self.bbox = [i * self.scale_factor for i in self.bbox]
            # print("fun-call end")
            # Parse keypoints
            keypoints = [float(x) for x in annotation[5:]]
            self.final_points = []
            # print("@@@@@@@@@@@@@@@@")
            self.keypoint_conf = []
            self.initial_points = []
            for i in range(0, len(keypoints), 3):
                x = keypoints[i] * self.image_size
                y = keypoints[i+1] * self.image_size
                conf = keypoints[i+2]
                self.final_points.append([x, y])
                self.initial_points.append([x,y])
                self.keypoint_conf.append(conf)
            self.is_initial = False
            # print("loaded annotation: ", " ".join(annotation))
            return True
        return False

    def load_image(self):
        # print("A")
        image_path = os.path.join(self.folder_path, self.image_files[self.current_image_index])
        prev_image_path = os.path.join(self.folder_path, self.image_files[(self.current_image_index-1)%len(self.image_files)])
        # print("initial path: ", image_path)
        self.label.config(text="File Name: {}".format(os.path.basename(image_path)))
        image = cv2.imread(image_path)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        self.is_annotation_loaded = False
        # if results and results[0].keypoints is not None:
        if self.is_initial: 
            try:
                if not self.load_annotation(image_path):
                # if not self.load_annotation(prev_image_path):
                    results = model(image)
                    self.initial_points = results[0].keypoints.xy.tolist()
                    self.initial_points = correct_keypoints(self.initial_points[0],image)
                    self.keypoint_conf = [float(1)] * self.keypoint_count
                    self.bbox = results[0].boxes.xyxy.tolist()[0]
                    self.bbox_n = results[0].boxes.xywhn.tolist()[0]
                    self.bbox = [i * self.scale_factor for i in self.bbox]
                    self.final_points = [list(point1) for point1 in self.initial_points]
                    # print("models bounding box pred: ", self.final_points)
                    self.is_initial = False
                    print("model ran!!!!")
                else: 
                    print("model did not run!! ")
            except Exception as e:
                # print("finall _points in exception: ", self.final_points)
                print(f"Error: {e}")
                return None
            
        # annotation_loaded = self.load_annotation(image_path)
        # if annotation_loaded:
        #     print("annotation loaded successfully!")
        if not self.is_annotation_loaded: self.convert_bbox()
        self.image = image
        image_size = self.image.shape
        self.size_label.config(text=f"Image_ac_size {image_size}")
        self.image = cv2.resize(self.image, None, fx=self.scale_factor, fy=self.scale_factor)  # Scale up the image
        self.photo = tk.PhotoImage(data=cv2.imencode('.png', self.image)[1].tobytes())
        self.canvas.config(width=self.image.shape[1], height=self.image.shape[0])  # Set canvas size to match image size
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def save_coordinates(self):
        # print("I")
        if not self.final_points:
            print("No points to save in image: ", self.image,"!!")
            return
        # print("save coordinates: ",self.bbox, self.bbox_n)
        self.final_points = [[px, py] for px, py in self.final_points]
        normalized_points = []
        # self.convert_bbox()
        for i, p in enumerate(self.final_points): normalized_points.append((p[0] / self.image_size, p[1] / self.image_size, self.keypoint_conf[i]))
        save_string = "0 " + " ".join([str(i / self.image_size) for i in self.bbox_n]) + " " + " ".join([f"{x} {y} {c}" for x, y, c in normalized_points]) # Create a formatted string for saving
        image_path = os.path.join(self.folder_path, self.image_files[self.current_image_index])
        labels_folder = os.path.join(os.path.dirname(image_path), "labels")
        os.makedirs(labels_folder, exist_ok=True)
        save_filename = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        save_path = os.path.join(labels_folder, save_filename)
        # print("to be saved annotation: ", save_string)
        with open(save_path, "w") as file: file.write(save_string)
        self.message_label.config(text=f"Saved coordinates to {save_path}")

    def draw_keypoints(self):
        # print("B")
        for i, (x, y) in enumerate(self.final_points):
            x_scaled, y_scaled = x * self.scale_factor, y * self.scale_factor  # Scale up the points
            color = colors[i % len(colors)]
            outline_color = "#%02x%02x%02x" % color
            if self.keypoint_conf[i] < 0.5:
                outline_color = "black"
            self.canvas.create_oval(x_scaled - self.keypoint_radius, y_scaled - self.keypoint_radius, x_scaled + self.keypoint_radius, y_scaled + self.keypoint_radius,
                                    fill="#%02x%02x%02x" % color, outline=outline_color, width=2)
            
    def draw_bbox(self):
        # print("C")
        self.canvas.create_rectangle(self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], outline="red", width=3)
        self.canvas.create_oval(self.bbox[0]-5, self.bbox[1]-5, self.bbox[0]+5, self.bbox[1]+5,fill="red", outline="red")
        self.canvas.create_oval(self.bbox[2]-5, self.bbox[3]-5, self.bbox[2]+5, self.bbox[3]+5,fill="red", outline="red")

    def convert_bbox(self):
        # print("D")
        tmp_bbox = [i / self.scale_factor for i in self.bbox]
        self.bbox_n[0] = (tmp_bbox[0] + tmp_bbox[2]) / 2
        self.bbox_n[1] = (tmp_bbox[1] + tmp_bbox[3]) / 2
        self.bbox_n[2] = tmp_bbox[2] - tmp_bbox[0]
        self.bbox_n[3] = tmp_bbox[3] - tmp_bbox[1]
    
    def reverse_convert_bbox_n_to_bbox_during_annotation_loading(self):
        
        # print("D")
        self.bbox = [0]*4
        if self.bbox_n!=None:
            self.bbox[0] = (self.bbox_n[0] - (self.bbox_n[2] / 2.0)) * self.scale_factor
            self.bbox[1] = (self.bbox_n[1] - (self.bbox_n[3] / 2.0)) * self.scale_factor
            self.bbox[2] = (self.bbox_n[0] + (self.bbox_n[2] / 2.0)) * self.scale_factor
            self.bbox[3] = (self.bbox_n[1] + (self.bbox_n[3] / 2.0)) * self.scale_factor


    def draw_connections(self, connections):
        # print("E")
        for connection in connections:
            i, j = connection
            x1, y1 = self.final_points[i]
            x2, y2 = self.final_points[j]
            x1_scaled, y1_scaled = x1 * self.scale_factor, y1 * self.scale_factor
            x2_scaled, y2_scaled = x2 * self.scale_factor, y2 * self.scale_factor
            self.canvas.create_line(x1_scaled, y1_scaled, x2_scaled, y2_scaled, fill="white", width=3)

    def select_point(self, event):
        # print("F")
        self.selected_point = None
        x, y = event.x / self.scale_factor, event.y / self.scale_factor  # Scale down the mouse click coordinates
        for i, (px, py) in enumerate(self.final_points):
            distance = math.sqrt((px - x)**2 + (py - y)**2)
            if distance < 5:
                self.selected_point = i
                self.selected_bbox = None
                #break
        if self.selected_point == None:
            for i in range(0, 2):
                px = self.bbox[2 * i] / self.scale_factor
                py = self.bbox[2 * i + 1] / self.scale_factor
                distance = math.sqrt((px - x) ** 2 + (py - y) ** 2)
                if distance < 5:
                    self.selected_bbox = i
                    break

    def move_selected_point(self, event):
        # print("G")
        if self.selected_point is not None:
            self.selected_bbox = None
            x, y = event.x / self.scale_factor, event.y / self.scale_factor  # Scale down the mouse move coordinates
            self.final_points[self.selected_point] = [x, y]  # Convert to integers and update as list
            self.keypoint_conf[self.selected_point] = self.current_conf
            self.redraw_keypoints()
        if self.selected_bbox is not None:
            self.selected_point = None
            x, y = event.x, event.y
            self.bbox[2 * self.selected_bbox] = x
            self.bbox[2 * self.selected_bbox + 1] = y                                           
            self.redraw_keypoints()

    def redraw_keypoints(self):
        # print("H")
        self.canvas.delete("all")
        self.load_image()
        self.redraw_connections()
        self.draw_keypoints()
        self.draw_bbox()

    def redraw_connections(self):
        # print("H")
        self.canvas.delete("connections")
        self.draw_connections(self.connections)
    
    

    def next_image(self):
        # print("J")
        # print("Next button pressed")
        self.counter+=1
        print("images done: ", self.counter)
        self.current_image_index = (self.current_image_index + 1) % len(self.image_files)
        self.is_initial = True
        self.is_annotation_loaded = False
        self.load_image()
        if self.final_points: 
            self.draw_connections(self.connections)
            self.draw_keypoints()
        self.check_file()
        self.message_label.config(text="")

    def prev_image(self):
        # print("H")
        self.current_image_index = (self.current_image_index - 1) % len(self.image_files)
        if self.current_image_index < 0: self.current_image_index = len(self.image_files) - 1
        self.is_initial = True
        self.is_annotation_loaded = False
        self.load_image()
        self.draw_connections(self.connections)
        self.draw_keypoints()
        self.message_label.config(text="")
        self.check_file()

    def check_file(self):
        # print("I")
        image_path = os.path.join(self.folder_path, self.image_files[self.current_image_index])
        labels_folder = os.path.join(os.path.dirname(image_path), "labels")
        os.makedirs(labels_folder, exist_ok=True)
        save_filename = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        save_path = os.path.join(labels_folder, save_filename)
        txt_files = os.listdir(labels_folder)
        # jpg_files = os.listdir(self.folder_path)
        jpg_file = self.image_files[self.current_image_index]
        if jpg_file[:-4] not in [f.split('.')[0] for f in txt_files if f.endswith('.txt')]: self.messagesave_label.config(text = f"coordinates not there {jpg_file}")
        else : self.messagesave_label.config(text = f"coordinates saved {jpg_file}" )
        # self.next_image()

    def delete_current_image(self):
        # print("J")
        file_path_d = os.path.join(self.folder_path, self.image_files[self.current_image_index])
        print("--- deleting path: ",file_path_d)
        if self.current_image_index < len(self.image_files):
            os.remove(os.path.join(self.folder_path, self.image_files[self.current_image_index]))
            self.image_files.pop(self.current_image_index)
        if self.current_image_index >= len(self.image_files): self.current_image_index = 0
        self.load_image()
        self.draw_connections(self.connections)
        self.draw_keypoints()
        self.message_label.config(text=f" {file_path_d} file deleted")

    def toggle_confidence(self):         
        self.current_conf = 0.05 if self.current_conf == 1.0 else 1.0
        confidence_text = f"{'Low' if self.current_conf == 0.05 else 'High'} ({self.current_conf})"
        self.confidence_label.config(text=f"Confidence: {confidence_text}")

def main():    
    folder_path = "train2017" #Change this to wherever images are
    model_type = "26"   
    global model 
    model = YOLO("weights/best26.pt")
    global another_model
    another_model = YOLO("yolov8s-pose.pt")
    global fast_review
    fast_review = False
    root = tk.Tk()
    root.title("Keypoint Editor")
    editor = KeypointEditor(root, folder_path, model_type)
    parts_button = Button(root, text="Display Parts and Colors", command=display_parts_colors(model_type))
    parts_button.pack()
    root.mainloop()

if __name__ == "__main__":
    main()
