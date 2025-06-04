import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import pyttsx3
from tkinterdnd2 import TkinterDnD, DND_FILES

# Suppress TensorFlow warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Import TensorFlow components
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

class VisionDropApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VisionDrop - Image Recognition")
        self.root.geometry("600x700")
        self.root.configure(bg="#1e1e1e")
        
        # Create a canvas with scrollbar for the entire window
        self.main_canvas = tk.Canvas(root, bg="#1e1e1e", highlightthickness=0)
        self.scrollbar = tk.Scrollbar(root, orient="vertical", command=self.main_canvas.yview)
        self.scrollable_frame = tk.Frame(self.main_canvas, bg="#1e1e1e")
        
        # Configure the canvas
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        )
        
        self.main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.main_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack the canvas and scrollbar
        self.main_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Setup UI
        self.setup_ui()
        
        # Initialize TTS engine
        self.init_tts_engine()
        
        # Load the default model
        self.load_model()
        
    def setup_ui(self):
        # Title and instructions
        title_label = tk.Label(self.scrollable_frame, text="VisionDrop", font=("Helvetica", 24, "bold"),
                              bg="#1e1e1e", fg="#4CAF50")
        title_label.pack(pady=(20, 5))
        
        self.label = tk.Label(self.scrollable_frame, text="Drag an image here or click to select",
                               font=("Helvetica", 14), bg="#1e1e1e", fg="white")
        self.label.pack(pady=(5, 15))
        
        # Drop area
        self.drop_area = tk.Label(self.scrollable_frame, text="⬇️ Drop Image Here", font=("Helvetica", 18),
                                   width=40, height=6, bg="#333333", fg="white",
                                   relief="solid", bd=2)
        self.drop_area.pack(pady=10)
        self.drop_area.drop_target_register(DND_FILES)
        self.drop_area.dnd_bind("<<Drop>>", self.handle_drop)
        self.drop_area.bind("<Button-1>", self.select_file)
        
        # Image display area
        self.image_label = tk.Label(self.scrollable_frame, bg="#1e1e1e")
        self.image_label.pack(pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.scrollable_frame, orient="horizontal",
                                        length=400, mode="indeterminate")
        
        # Results area
        results_frame = tk.Frame(self.scrollable_frame, bg="#1e1e1e")
        results_frame.pack(pady=10, fill=tk.BOTH, expand=True, padx=20)
        results_label = tk.Label(results_frame, text="Recognition Results:",
                                 font=("Helvetica", 12, "bold"), bg="#1e1e1e", fg="white")
        results_label.pack(anchor="w", pady=(0, 5))
        # Add scrollbar for results
        result_scroll = tk.Scrollbar(results_frame)
        result_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text = tk.Text(results_frame, height=6, width=50, font=("Courier", 12),
                                   bg="#2d2d2d", fg="white", relief="flat", padx=10, pady=10,
                                  yscrollcommand=result_scroll.set)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        result_scroll.config(command=self.result_text.yview)
        
        # Debug info
        self.debug_var = tk.StringVar()
        self.debug_label = tk.Label(self.scrollable_frame, textvariable=self.debug_var,
                                    font=("Courier", 10), bg="#1e1e1e", fg="#AAAAAA")
        self.debug_label.pack(pady=(0, 5))
        
        # Control buttons
        button_frame = tk.Frame(self.scrollable_frame, bg="#1e1e1e")
        button_frame.pack(pady=(5, 20), fill=tk.X)
        
        self.speak_button = tk.Button(button_frame, text="Speak Results",
                                      command=self.speak_current_results,
                                     bg="#4CAF50", fg="white", font=("Helvetica", 10),
                                     relief="flat", padx=10, pady=5)
        self.speak_button.pack(side=tk.LEFT, padx=(20, 10))
        self.speak_button.config(state=tk.DISABLED)
        
        clear_button = tk.Button(button_frame, text="Clear",
                                 command=self.clear_results,
                                bg="#F44336", fg="white", font=("Helvetica", 10),
                                relief="flat", padx=10, pady=5)
        clear_button.pack(side=tk.LEFT, padx=10)
        
        # Try different model button
        self.model_var = tk.StringVar(value="MobileNetV2")
        model_menu = tk.OptionMenu(button_frame, self.model_var,
                                   "MobileNetV2", "ResNet50", "InceptionV3",
                                   command=self.change_model)
        model_menu.config(bg="#555555", fg="white", font=("Helvetica", 10),
                         relief="flat", padx=10, pady=5)
        model_menu.pack(side=tk.RIGHT, padx=20)
        
        model_label = tk.Label(button_frame, text="Model:",
                               bg="#1e1e1e", fg="white", font=("Helvetica", 10))
        model_label.pack(side=tk.RIGHT, padx=(0, 5))
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var,
                              bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#333333", fg="white")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Current results storage
        self.current_predictions = None
        self.current_image_path = None
        
        # Configure mouse wheel scrolling
        self.scrollable_frame.bind("<MouseWheel>", self._on_mousewheel)
        self.main_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.root.bind("<MouseWheel>", self._on_mousewheel)
        
        self.root.update()
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
    
    def _on_mousewheel(self, event):
        self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def load_model(self, model_name="MobileNetV2"):
        self.status_var.set(f"Loading {model_name} model...")
        self.root.update()
        
        try:
            if model_name == "MobileNetV2":
                from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
                self.model = MobileNetV2(weights='imagenet')
                self.preprocess_input = preprocess_input
                self.decode_predictions = decode_predictions
            elif model_name == "ResNet50":
                from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
                self.model = ResNet50(weights='imagenet')
                self.preprocess_input = preprocess_input
                self.decode_predictions = decode_predictions
            elif model_name == "InceptionV3":
                from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
                self.model = InceptionV3(weights='imagenet')
                self.preprocess_input = preprocess_input
                self.decode_predictions = decode_predictions
                
            self.status_var.set(f"{model_name} model loaded successfully")
            self.debug_var.set(f"Model: {model_name} | Input shape: {self.model.input_shape}")
            
            # If we have a current image, reprocess it with the new model
            if self.current_image_path and os.path.exists(self.current_image_path):
                self.process_image(self.current_image_path)
            
        except Exception as e:
            messagebox.showerror("Model Loading Error", f"Failed to load model: {str(e)}")
            self.status_var.set("Error loading model")

    def change_model(self, selection):
        self.load_model(selection)

    def init_tts_engine(self):
        try:
            self.tts_engine = pyttsx3.init()
        except Exception as e:
            messagebox.showwarning("TTS Warning", f"Text-to-speech engine could not be initialized: {str(e)}")
            self.tts_engine = None

    def handle_drop(self, event):
        file_path = event.data
        # Clean up the file path (remove curly braces if present)
        file_path = file_path.strip("{").strip("}")
        
        if os.path.isfile(file_path):
            self.process_image(file_path)
        else:
            messagebox.showerror("Error", "Invalid file dropped")

    def select_file(self, event=None):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")])
        if file_path:
            self.process_image(file_path)

    def process_image(self, file_path):
        # Store current image path
        self.current_image_path = file_path
        
        # Show progress bar
        self.progress.pack(pady=10)
        self.progress.start()
        self.status_var.set(f"Processing {os.path.basename(file_path)}...")
        self.root.update()
        
        # Start processing in a separate thread to keep UI responsive
        threading.Thread(target=self._process_image_thread, args=(file_path,), daemon=True).start()

    def _process_image_thread(self, file_path):
        try:
            # Display the image
            img = Image.open(file_path).convert('RGB')  # Convert to RGB to handle all image types
            img.thumbnail((350, 350))
            img_tk = ImageTk.PhotoImage(img)
            
            # Prepare image for model
            target_size = self.model.input_shape[1:3]  # Get the expected input size from model
            img_for_model = img.resize(target_size)
            
            # Convert to array and preprocess
            img_array = np.array(img_for_model)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = self.preprocess_input(img_array)
            
            # Debug info
            debug_info = f"Image: {os.path.basename(file_path)} | Size: {img.size} | " \
                         f"Array shape: {img_array.shape} | Model input: {self.model.input_shape}"
            
            # Run prediction
            preds = self.model.predict(img_array)
            self.current_predictions = self.decode_predictions(preds, top=5)[0]  # Get top 5 instead of 3
            
            # Format results
            result_text = "\n".join([
                f"{i+1}. {label} ({prob*100:.2f}%)" 
                for i, (_, label, prob) in enumerate(self.current_predictions)
            ])
            
            # Update UI in the main thread
            self.root.after(0, self._update_ui_with_results, img_tk, result_text, debug_info)
            
        except Exception as e:
            self.root.after(0, lambda: self._show_error(str(e)))

    def _update_ui_with_results(self, img_tk, result_text, debug_info):
        # Update image
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk
        
        # Update results
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, result_text)
        
        # Update debug info
        self.debug_var.set(debug_info)
        
        # Enable speak button if TTS is available
        if self.tts_engine is not None:
            self.speak_button.config(state=tk.NORMAL)
        
        # Hide progress bar
        self.progress.stop()
        self.progress.pack_forget()
        self.status_var.set("Recognition complete")
        
        # Update scroll region
        self.root.update()
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))

    def _show_error(self, error_message):
        self.progress.stop()
        self.progress.pack_forget()
        messagebox.showerror("Error", f"An error occurred: {error_message}")
        self.status_var.set("Error occurred")

    def speak_current_results(self):
        if self.tts_engine is None or self.current_predictions is None:
            return
            
        self.status_var.set("Speaking results...")
        
        # Speak in a separate thread to keep UI responsive
        threading.Thread(target=self._speak_thread, daemon=True).start()

    def _speak_thread(self):
        try:
            for i, (_, label, prob) in enumerate(self.current_predictions):
                if i >= 3:  # Only speak top 3 results
                    break
                text = f"Prediction {i+1}: {label}, {prob*100:.2f} percent."
                self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            
            # Update status when done
            self.root.after(0, lambda: self.status_var.set("Ready"))
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"TTS Error: {str(e)}"))

    def clear_results(self):
        self.image_label.config(image='')
        self.result_text.delete("1.0", tk.END)
        self.current_predictions = None
        self.current_image_path = None
        self.speak_button.config(state=tk.DISABLED)
        self.debug_var.set("")
        self.status_var.set("Ready")

if __name__ == "__main__":
    try:
        root = TkinterDnD.Tk()
        app = VisionDropApp(root)
        
        # Enable mousewheel scrolling for Linux and Windows
        def _on_mousewheel(event):
            app.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            
        # For Linux
        root.bind("<Button-4>", lambda e: app.main_canvas.yview_scroll(-1, "units"))
        root.bind("<Button-5>", lambda e: app.main_canvas.yview_scroll(1, "units"))
        
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Application Error", f"Failed to start application: {str(e)}")

