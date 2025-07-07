import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
import subprocess
import sys
import webbrowser
from PIL import Image, ImageTk
import tempfile
import platform

class Furniture3DViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Furniture 3D Model Viewer")
        self.root.geometry("1200x800")
        
        # Default paths based on your pipeline structure
        self.base_dir = "furniture_3d_pipeline"
        self.png_dir = os.path.join(self.base_dir, "extracted_furniture")
        self.model_3d_dir = os.path.join(self.base_dir, "3d_models")
        self.log_file = os.path.join(self.base_dir, "processing_log.json")
        
        # Data storage
        self.furniture_data = []
        self.current_selection = None
        self.original_image_path = None
        
        self.create_widgets()
        self.load_data()
        
    def create_widgets(self):
        """Create the GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Furniture 3D Model Viewer", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # Left panel - Furniture list
        left_frame = ttk.LabelFrame(main_frame, text="Furniture Items", padding="5")
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Furniture listbox with scrollbar
        listbox_frame = ttk.Frame(left_frame)
        listbox_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)
        
        self.furniture_listbox = tk.Listbox(listbox_frame, width=30, height=20)
        scrollbar = ttk.Scrollbar(listbox_frame, orient="vertical", command=self.furniture_listbox.yview)
        self.furniture_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.furniture_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        listbox_frame.columnconfigure(0, weight=1)
        listbox_frame.rowconfigure(0, weight=1)
        
        # Bind selection event
        self.furniture_listbox.bind('<<ListboxSelect>>', self.on_furniture_select)
        
        # Buttons frame
        buttons_frame = ttk.Frame(left_frame)
        buttons_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.refresh_btn = ttk.Button(buttons_frame, text="Refresh", command=self.load_data)
        self.refresh_btn.grid(row=0, column=0, padx=(0, 5))
        
        self.browse_btn = ttk.Button(buttons_frame, text="Browse Folder", command=self.browse_folder)
        self.browse_btn.grid(row=0, column=1)
        
        # Right panel - Preview and controls
        right_frame = ttk.LabelFrame(main_frame, text="Preview & Controls", padding="5")
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_frame.columnconfigure(0, weight=1)
        
        # Image preview
        self.image_label = ttk.Label(right_frame, text="Select a furniture item to preview")
        self.image_label.grid(row=0, column=0, pady=(0, 10))
        
        # Info text
        self.info_text = tk.Text(right_frame, height=8, width=50, wrap=tk.WORD)
        info_scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scrollbar.set)
        
        self.info_text.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        info_scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        
        # 3D Model buttons
        model_buttons_frame = ttk.LabelFrame(right_frame, text="3D Models", padding="5")
        model_buttons_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.base_model_btn = ttk.Button(model_buttons_frame, text="Open Base Model", 
                                        command=lambda: self.open_3d_model("base"),
                                        state="disabled")
        self.base_model_btn.grid(row=0, column=0, padx=(0, 5))
        
        self.pbr_model_btn = ttk.Button(model_buttons_frame, text="Open PBR Model", 
                                       command=lambda: self.open_3d_model("pbr"),
                                       state="disabled")
        self.pbr_model_btn.grid(row=0, column=1)
        
        # File operations
        file_ops_frame = ttk.LabelFrame(right_frame, text="File Operations", padding="5")
        file_ops_frame.grid(row=3, column=0, sticky=(tk.W, tk.E))
        
        self.open_png_btn = ttk.Button(file_ops_frame, text="Open PNG", 
                                      command=self.open_png_file,
                                      state="disabled")
        self.open_png_btn.grid(row=0, column=0, padx=(0, 5))
        
        self.open_original_btn = ttk.Button(file_ops_frame, text="View Original", 
                                           command=self.view_original_image,
                                           state="disabled")
        self.open_original_btn.grid(row=0, column=1, padx=(0, 5))
        
        self.open_folder_btn = ttk.Button(file_ops_frame, text="Open in Explorer", 
                                         command=self.open_in_explorer,
                                         state="disabled")
        self.open_folder_btn.grid(row=0, column=2)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def load_data(self):
        """Load furniture data from processing log"""
        try:
            self.furniture_data = []
            self.furniture_listbox.delete(0, tk.END)
            self.original_image_path = None
            
            # Check if log file exists
            if not os.path.exists(self.log_file):
                self.scan_files_directly()
                return
            
            # Load from log file
            with open(self.log_file, 'r') as f:
                log_data = json.load(f)
            
            processed_items = log_data.get("processed_items", [])
            
            if not processed_items:
                self.scan_files_directly()
                return
            
            # Try to find the original image path
            self.find_original_image()
            
            # Load processed items
            for item in processed_items:
                self.furniture_data.append(item)
                display_name = f"{item['class_name']} - {item['filename']}"
                self.furniture_listbox.insert(tk.END, display_name)
            
            self.status_var.set(f"Loaded {len(self.furniture_data)} items from log")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.scan_files_directly()
    
    def scan_files_directly(self):
        """Scan files directly if log is not available"""
        try:
            if not os.path.exists(self.png_dir):
                self.status_var.set("PNG directory not found")
                return
            
            # Try to find original image
            self.find_original_image()
            
            png_files = [f for f in os.listdir(self.png_dir) if f.endswith('.png')]
            
            for png_file in png_files:
                # Parse filename to extract class and index
                base_name = png_file.replace('.png', '')
                parts = base_name.split('_')
                
                if len(parts) >= 2:
                    class_name = '_'.join(parts[:-1])
                    index = parts[-1]
                else:
                    class_name = base_name
                    index = "001"
                
                # Check for corresponding 3D models
                base_glb = os.path.join(self.model_3d_dir, f"{base_name}_base.glb")
                pbr_glb = os.path.join(self.model_3d_dir, f"{base_name}_pbr.glb")
                
                downloaded_models = {}
                if os.path.exists(base_glb):
                    downloaded_models["base"] = base_glb
                if os.path.exists(pbr_glb):
                    downloaded_models["pbr"] = pbr_glb
                
                item = {
                    "filename": png_file,
                    "class_name": class_name,
                    "downloaded_models": downloaded_models,
                    "filepath": os.path.join(self.png_dir, png_file)
                }
                
                self.furniture_data.append(item)
                display_name = f"{class_name} - {png_file}"
                self.furniture_listbox.insert(tk.END, display_name)
            
            self.status_var.set(f"Scanned {len(self.furniture_data)} PNG files")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to scan files: {str(e)}")
            self.status_var.set("Error scanning files")
    
    def on_furniture_select(self, event):
        """Handle furniture selection"""
        try:
            selection = self.furniture_listbox.curselection()
            if not selection:
                return
            
            index = selection[0]
            self.current_selection = self.furniture_data[index]
            
            # Update preview
            self.update_preview()
            
            # Enable/disable buttons
            self.update_buttons()
            
        except Exception as e:
            messagebox.showerror("Error", f"Selection error: {str(e)}")
    
    def update_preview(self):
        """Update the preview image and info"""
        if not self.current_selection:
            return
        
        try:
            # Load and display image
            png_path = self.current_selection.get('filepath', 
                                                os.path.join(self.png_dir, self.current_selection['filename']))
            
            if os.path.exists(png_path):
                image = Image.open(png_path)
                # Resize image to fit preview
                image.thumbnail((300, 300), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                self.image_label.configure(image=photo, text="")
                self.image_label.image = photo  # Keep a reference
            else:
                self.image_label.configure(image="", text="Image not found")
                self.image_label.image = None
            
            # Update info text
            self.info_text.delete(1.0, tk.END)
            
            info = f"Filename: {self.current_selection['filename']}\n"
            info += f"Class: {self.current_selection['class_name']}\n"
            
            if 'confidence' in self.current_selection:
                info += f"Confidence: {self.current_selection['confidence']:.2f}\n"
            
            if 'box' in self.current_selection:
                box = self.current_selection['box']
                info += f"Bounding Box: {box}\n"
            
            if 'processed_at' in self.current_selection:
                info += f"Processed: {self.current_selection['processed_at']}\n"
            
            models = self.current_selection.get('downloaded_models', {})
            info += f"\nAvailable 3D Models:\n"
            
            if 'base' in models:
                info += f"✅ Base Model: {os.path.basename(models['base'])}\n"
            else:
                info += f"❌ Base Model: Not available\n"
            
            if 'pbr' in models:
                info += f"✅ PBR Model: {os.path.basename(models['pbr'])}\n"
            else:
                info += f"❌ PBR Model: Not available\n"
            
            # Add original image info
            if self.original_image_path:
                info += f"\nOriginal Image: {os.path.basename(self.original_image_path)}\n"
            else:
                info += f"\nOriginal Image: Not found\n"
            
            self.info_text.insert(1.0, info)
            
        except Exception as e:
            messagebox.showerror("Error", f"Preview error: {str(e)}")
    
    def update_buttons(self):
        """Update button states based on current selection"""
        if not self.current_selection:
            self.base_model_btn.configure(state="disabled")
            self.pbr_model_btn.configure(state="disabled")
            self.open_png_btn.configure(state="disabled")
            self.open_original_btn.configure(state="disabled")
            self.open_folder_btn.configure(state="disabled")
            return
        
        models = self.current_selection.get('downloaded_models', {})
        
        # 3D model buttons
        if 'base' in models and os.path.exists(models['base']):
            self.base_model_btn.configure(state="normal")
        else:
            self.base_model_btn.configure(state="disabled")
        
        if 'pbr' in models and os.path.exists(models['pbr']):
            self.pbr_model_btn.configure(state="normal")
        else:
            self.pbr_model_btn.configure(state="disabled")
        
        # File operation buttons
        png_path = self.current_selection.get('filepath', 
                                            os.path.join(self.png_dir, self.current_selection['filename']))
        
        if os.path.exists(png_path):
            self.open_png_btn.configure(state="normal")
            self.open_folder_btn.configure(state="normal")
        else:
            self.open_png_btn.configure(state="disabled")
            self.open_folder_btn.configure(state="disabled")
        
        # Original image button
        if self.original_image_path and os.path.exists(self.original_image_path):
            self.open_original_btn.configure(state="normal")
        else:
            self.open_original_btn.configure(state="disabled")
    
    def open_3d_model(self, model_type):
        """Open 3D model file"""
        if not self.current_selection:
            return
        
        models = self.current_selection.get('downloaded_models', {})
        
        if model_type not in models:
            messagebox.showwarning("Warning", f"{model_type.upper()} model not available")
            return
        
        model_path = models[model_type]
        
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Model file not found: {model_path}")
            return
        
        try:
            # Try to open with default application
            if platform.system() == "Windows":
                os.startfile(model_path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", model_path])
            else:  # Linux
                subprocess.run(["xdg-open", model_path])
            
            self.status_var.set(f"Opened {model_type.upper()} model: {os.path.basename(model_path)}")
            
        except Exception as e:
            # If default application fails, try browser
            try:
                file_url = f"file://{os.path.abspath(model_path)}"
                webbrowser.open(file_url)
                self.status_var.set(f"Opened {model_type.upper()} model in browser")
            except:
                messagebox.showerror("Error", f"Failed to open 3D model: {str(e)}")
    
    def open_png_file(self):
        """Open PNG file"""
        if not self.current_selection:
            return
        
        png_path = self.current_selection.get('filepath', 
                                            os.path.join(self.png_dir, self.current_selection['filename']))
        
        if not os.path.exists(png_path):
            messagebox.showerror("Error", f"PNG file not found: {png_path}")
            return
        
        try:
            if platform.system() == "Windows":
                os.startfile(png_path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", png_path])
            else:  # Linux
                subprocess.run(["xdg-open", png_path])
            
            self.status_var.set(f"Opened PNG: {os.path.basename(png_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open PNG: {str(e)}")
    
    def open_in_explorer(self):
        """Open file location in explorer"""
        if not self.current_selection:
            return
        
        png_path = self.current_selection.get('filepath', 
                                            os.path.join(self.png_dir, self.current_selection['filename']))
        
        folder_path = os.path.dirname(png_path)
        
        if not os.path.exists(folder_path):
            messagebox.showerror("Error", f"Folder not found: {folder_path}")
            return
        
        try:
            if platform.system() == "Windows":
                subprocess.run(["explorer", folder_path])
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", folder_path])
            else:  # Linux
                subprocess.run(["xdg-open", folder_path])
            
            self.status_var.set(f"Opened folder: {folder_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open folder: {str(e)}")
    
    def browse_folder(self):
        """Browse for a different project folder"""
        folder_path = filedialog.askdirectory(title="Select Furniture 3D Pipeline Folder")
        
        if folder_path:
            self.base_dir = folder_path
            self.png_dir = os.path.join(self.base_dir, "extracted_furniture")
            self.model_3d_dir = os.path.join(self.base_dir, "3d_models")
            self.log_file = os.path.join(self.base_dir, "processing_log.json")
            
            self.load_data()
            self.status_var.set(f"Loaded folder: {folder_path}")
    
    def find_original_image(self):
        """Find the original image file used for processing"""
        # Common image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        # Check in base directory first
        for ext in image_extensions:
            image_path = os.path.join(self.base_dir, f"image{ext}")
            if os.path.exists(image_path):
                self.original_image_path = image_path
                return
        
        # Check for common image names
        common_names = ['input', 'original', 'source', 'room', 'scene']
        for name in common_names:
            for ext in image_extensions:
                image_path = os.path.join(self.base_dir, f"{name}{ext}")
                if os.path.exists(image_path):
                    self.original_image_path = image_path
                    return
        
        # Check in parent directory
        parent_dir = os.path.dirname(self.base_dir)
        for ext in image_extensions:
            image_path = os.path.join(parent_dir, f"image{ext}")
            if os.path.exists(image_path):
                self.original_image_path = image_path
                return
        
        # Check for any image file in base directory
        try:
            for filename in os.listdir(self.base_dir):
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    self.original_image_path = os.path.join(self.base_dir, filename)
                    return
        except:
            pass
        
        # Check in parent directory for any image
        try:
            for filename in os.listdir(parent_dir):
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    self.original_image_path = os.path.join(parent_dir, filename)
                    return
        except:
            pass
        
        self.original_image_path = None
    
    def view_original_image(self):
        """View the original image"""
        if not self.original_image_path:
            messagebox.showwarning("Warning", "Original image not found")
            return
        
        if not os.path.exists(self.original_image_path):
            messagebox.showerror("Error", f"Original image file not found: {self.original_image_path}")
            return
        
        self.show_image_window(self.original_image_path, "Original Image")
    
    def show_image_window(self, image_path, title):
        """Show image in a new window"""
        try:
            # Create new window
            img_window = tk.Toplevel(self.root)
            img_window.title(title)
            img_window.geometry("800x600")
            
            # Load and display image
            image = Image.open(image_path)
            
            # Calculate size to fit window while maintaining aspect ratio
            window_width, window_height = 750, 550
            img_width, img_height = image.size
            
            # Calculate scaling factor
            scale_w = window_width / img_width
            scale_h = window_height / img_height
            scale = min(scale_w, scale_h)
            
            # Resize image
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Create label to display image
            img_label = tk.Label(img_window, image=photo)
            img_label.image = photo  # Keep a reference
            img_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
            
            # Add file info
            info_frame = ttk.Frame(img_window)
            info_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
            
            info_text = f"File: {os.path.basename(image_path)}\nPath: {image_path}\nSize: {img_width}x{img_height}"
            info_label = ttk.Label(info_frame, text=info_text, font=("Arial", 9))
            info_label.pack()
            
            # Add close button
            close_btn = ttk.Button(info_frame, text="Close", command=img_window.destroy)
            close_btn.pack(pady=(10, 0))
            
            # Make window modal
            img_window.transient(self.root)
            img_window.grab_set()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {str(e)}")

def main():
    root = tk.Tk()
    app = Furniture3DViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()