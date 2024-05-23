import os
import tkinter as tk
from PIL import Image, ImageTk

class ImageViewer(tk.Tk):
    def __init__(self, image_dir):
        super().__init__()
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.num_images = len(self.image_files)
        self.image_index = 0
        self.title("Image Viewer")
        self.geometry("800x600")
        
        # self.canvas = tk.Canvas()
        # self.canvas.pack()
        
        self.create_widgets()

    def create_widgets(self):
        self.image_label = tk.Label(self)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        self.display_image()

        self.prev_button = tk.Button(self, text="Prev", command=self.prev_image)
        self.prev_button.pack(side=tk.LEFT)

        self.next_button = tk.Button(self, text="Next", command=self.next_image)
        self.next_button.pack(side=tk.RIGHT)

        self.delete_button = tk.Button(self, text="Delete", command=self.delete_image)
        self.delete_button.pack(side=tk.BOTTOM)

    def display_image(self):
        image_path = os.path.join(self.image_dir, self.image_files[self.image_index])
        image = Image.open(image_path)
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.label = tk.Label( text="File Name: {}".format(os.path.basename(image_path)))
        self.label.pack()
        self.image_label.image = photo

    def prev_image(self):
        self.image_index = (self.image_index - 1) % self.num_images
        self.display_image()

    def next_image(self):
        self.image_index = (self.image_index + 1) % self.num_images
        self.display_image()

    def delete_image(self):
        image_path = os.path.join(self.image_dir, self.image_files[self.image_index])
        os.remove(image_path)
        self.image_files.pop(self.image_index)
        self.num_images = len(self.image_files)
        if self.num_images == 0:
            self.image_label.config(text="No more images")
        elif self.image_index >= self.num_images:
            self.image_index = self.num_images - 1
        self.display_image()

if __name__ == "__main__":
    image_dir = "trial2"
    app = ImageViewer(image_dir)
    app.mainloop()
