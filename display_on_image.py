import os
import subprocess
import platform
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button
import numpy as np
from PIL import Image
import json

class InteractiveFurnitureViewer:
    def __init__(self, base_directory):
        self.base_directory = base_directory
        self.current_image_index = 0
        self.show_segmentation = False
        
        # Initialize data structures
        self.original_images = {}
        self.segmentation_images = {}
        self.model_3d_files = {}
        self.extracted_pieces = {}
        self.furniture_coordinates = {}  # Store actual bounding box coordinates
        
        # Scan directory structure and load coordinates
        self.scan_directory()
        self.load_furniture_coordinates()
        
        # Get list of available original images
        self.available_images = list(self.original_images.keys())
        
        if not self.available_images:
            self.scan_base_directory_for_images()
            
        if not self.available_images:
            print("No images found!")
            return
            
        # Setup matplotlib and load first image
        self.setup_plot()
        self.load_image()
        
    def scan_base_directory_for_images(self):
        """Scan base directory for any image files"""
        print("Scanning base directory for images...")
        
        try:
            files = os.listdir(self.base_directory)
            print(f"Files in base directory: {files}")
            
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                    name = os.path.splitext(file)[0]
                    full_path = os.path.join(self.base_directory, file)
                    self.original_images[name] = full_path
                    print(f"Found image in base directory: {name} -> {full_path}")
            
            self.available_images = list(self.original_images.keys())
            print(f"Total images found in base directory: {len(self.available_images)}")
            
        except Exception as e:
            print(f"Error scanning base directory: {e}")
        
    def scan_directory(self):
        """Scan the directory structure for images and 3D models"""
        if not os.path.exists(self.base_directory):
            print(f"ERROR: Base directory does not exist: {self.base_directory}")
            return
        
        # Scan for original images
        original_dir = os.path.join(self.base_directory, "original_images")
        if os.path.exists(original_dir):
            for file in os.listdir(original_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                    name = os.path.splitext(file)[0]
                    self.original_images[name] = os.path.join(original_dir, file)
        
        # Scan for segmentation images
        segmentation_dir = os.path.join(self.base_directory, "segmentation_images")
        if os.path.exists(segmentation_dir):
            for file in os.listdir(segmentation_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                    name = os.path.splitext(file)[0]
                    self.segmentation_images[name] = os.path.join(segmentation_dir, file)
        
        # Scan for 3D models
        for models_dir in [
            os.path.join(self.base_directory, "furniture_3d_pipeline", "3d_models"),
            os.path.join(self.base_directory, "3d_models")
        ]:
            if os.path.exists(models_dir):
                for file in os.listdir(models_dir):
                    if file.lower().endswith('.glb'):
                        name = os.path.splitext(file)[0]
                        self.model_3d_files[name] = os.path.join(models_dir, file)
        
        # Scan for extracted pieces
        for extracted_dir in [
            os.path.join(self.base_directory, "furniture_3d_pipeline", "extracted_furniture"),
            os.path.join(self.base_directory, "extracted_furniture")
        ]:
            if os.path.exists(extracted_dir):
                for file in os.listdir(extracted_dir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                        name = os.path.splitext(file)[0]
                        self.extracted_pieces[name] = os.path.join(extracted_dir, file)
    
    def load_furniture_coordinates(self):
        """Load furniture bounding box coordinates from SAM segmentation output"""
        coordinates_file = os.path.join(self.base_directory, "furniture_3d_pipeline", "furniture_coordinates.json")
        
        if not os.path.exists(coordinates_file):
            print("No furniture coordinates found. Run sam_segment.py first.")
            return
        
        try:
            with open(coordinates_file, 'r') as f:
                coordinates_data = json.load(f)
            
            print(f"📍 Loaded {len(coordinates_data['furniture_pieces'])} furniture coordinates")
            
            # Store coordinates indexed by furniture name
            for piece in coordinates_data['furniture_pieces']:
                furniture_name = os.path.splitext(piece['filename'])[0]
                self.furniture_coordinates[furniture_name] = {
                    'bounding_box': piece['bounding_box'],
                    'center': piece['center'],
                    'class_name': piece['class_name'],
                    'confidence': piece['confidence']
                }
            
        except Exception as e:
            print(f"Error loading furniture coordinates: {e}")
    
    def open_3d_model(self, model_path):
        """Open a 3D model file in the default system application"""
        try:
            if not os.path.exists(model_path):
                print(f"3D model file not found: {model_path}")
                return False
                
            system = platform.system()
            
            if system == "Windows":
                # Windows - use start command
                subprocess.run(['start', model_path], shell=True, check=True)
            elif system == "Darwin":  # macOS
                # macOS - use open command
                subprocess.run(['open', model_path], check=True)
            elif system == "Linux":
                # Linux - use xdg-open
                subprocess.run(['xdg-open', model_path], check=True)
            else:
                print(f"Unsupported operating system: {system}")
                return False
                
            print(f"Opening 3D model: {model_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error opening 3D model: {e}")
            return False
        except FileNotFoundError:
            print("Could not find system command to open 3D model")
            return False
    
    def setup_plot(self):
        """Setup the matplotlib plot and UI elements"""
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title("Interactive Furniture 3D Model Viewer")
        
        # Connect click event and keyboard event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Create buttons
        ax_toggle = plt.axes([0.02, 0.02, 0.15, 0.05])
        self.btn_toggle = Button(ax_toggle, 'Toggle Segmentation')
        self.btn_toggle.on_clicked(self.toggle_segmentation)
        
        ax_next = plt.axes([0.2, 0.02, 0.1, 0.05])
        self.btn_next = Button(ax_next, 'Next Image')
        self.btn_next.on_clicked(self.next_image)
        
        ax_prev = plt.axes([0.32, 0.02, 0.1, 0.05])
        self.btn_prev = Button(ax_prev, 'Previous Image')
        self.btn_prev.on_clicked(self.previous_image)
        
        ax_show_all = plt.axes([0.44, 0.02, 0.12, 0.05])
        self.btn_show_all = Button(ax_show_all, 'Show All Models')
        self.btn_show_all.on_clicked(self.show_all_models)
        
        # Add debug info button
        ax_debug = plt.axes([0.58, 0.02, 0.1, 0.05])
        self.btn_debug = Button(ax_debug, 'Debug Info')
        self.btn_debug.on_clicked(self.show_debug_info)
        
        # Add region overlay button
        ax_regions = plt.axes([0.7, 0.02, 0.12, 0.05])
        self.btn_regions = Button(ax_regions, 'Show Regions')
        self.btn_regions.on_clicked(self.toggle_region_overlay)
        
        # Initialize region overlay state
        self.show_region_overlay = False
    
    def show_debug_info(self, event):
        """Show debug information"""
        print(f"\nDEBUG INFO:")
        print(f"- Images: {len(self.available_images)}")
        print(f"- 3D models: {len(self.model_3d_files)}")
        print(f"- Furniture coordinates: {len(self.furniture_coordinates)}")
        if self.available_images:
            current_name = self.available_images[self.current_image_index]
            print(f"- Current image: {current_name}")
    
    def toggle_region_overlay(self, event):
        """Toggle the display of furniture region boundaries"""
        self.show_region_overlay = not self.show_region_overlay
        print(f"Region overlay: {'ON' if self.show_region_overlay else 'OFF'}")
        self.load_image()
    
    def load_image(self):
        """Load and display the current image"""
        if not self.available_images:
            self.ax.clear()
            self.ax.text(0.5, 0.5, 'No images found!', 
                        ha='center', va='center', transform=self.ax.transAxes, fontsize=12)
            self.ax.set_title("No Images Available")
            plt.draw()
            return
            
        current_name = self.available_images[self.current_image_index]
        
        if current_name in self.original_images:
            img_path = self.original_images[current_name]
            
            try:
                # Try to load with matplotlib first
                try:
                    img = mpimg.imread(img_path)
                except Exception:
                    # Fallback to PIL
                    pil_img = Image.open(img_path)
                    img = np.array(pil_img)
                
                self.current_image = img
                self.current_image_name = current_name
                
                # Display image
                self.ax.clear()
                self.ax.imshow(img)
                self.ax.set_title(f"{current_name} - Click on furniture to view 3D model")
                self.ax.axis('off')
                
                # Show segmentation overlay if enabled
                if self.show_segmentation and current_name in self.segmentation_images:
                    seg_path = self.segmentation_images[current_name]
                    if os.path.exists(seg_path):
                        try:
                            seg_img = mpimg.imread(seg_path)
                            self.ax.imshow(seg_img, alpha=0.5)
                        except Exception as e:
                            print(f"Error loading segmentation: {e}")
                
                plt.draw()
                
            except Exception as e:
                print(f"Error loading image: {e}")
                self.ax.clear()
                self.ax.text(0.5, 0.5, f'Error loading image:\n{str(e)}', 
                            ha='center', va='center', transform=self.ax.transAxes, fontsize=12)
                self.ax.set_title(f"Error - {current_name}")
                plt.draw()
    
    def on_click(self, event):
        """Handle mouse clicks on the image"""
        if event.inaxes != self.ax:
            return
        
        if hasattr(self, 'current_image_name'):
            x, y = int(event.xdata), int(event.ydata)
            print(f"Clicked at coordinates: ({x}, {y})")
            
            # Try to detect which furniture piece was clicked based on extracted furniture masks
            detected_furniture = self.detect_furniture_at_position(x, y)
            
            if detected_furniture:
                furniture_name = detected_furniture
                print(f"Detected furniture: {furniture_name}")
                
                # Look for corresponding 3D model
                found_model = self.find_matching_3d_model(furniture_name)
                
                if found_model:
                    model_name, model_path = found_model
                    print(f"Found matching 3D model: {model_name}")
                    
                    success = self.open_3d_model(model_path)
                    if success:
                        print(f"Successfully opened 3D model for {furniture_name}")
                    else:
                        print(f"Failed to open 3D model for {furniture_name}")
                else:
                    print(f"No 3D model found for detected furniture: {furniture_name}")
                    self.show_available_models()
            else:
                print("No furniture detected at this position")
                print("Available furniture regions:")
                self.show_available_furniture_regions()
                self.show_available_models()
    
    def detect_furniture_at_position(self, x, y):
        """Detect which furniture piece is at the given position using SAM coordinates"""
        return self.detect_from_sam_coordinates(x, y)
    
    def detect_from_sam_coordinates(self, x, y):
        """Detect furniture using actual bounding box coordinates from SAM segmentation"""
        if not self.furniture_coordinates:
            print("No furniture coordinates loaded - run sam_segment.py first")
            return None
        
        # Check each furniture piece's bounding box for direct hit
        for furniture_name, coord_data in self.furniture_coordinates.items():
            bbox = coord_data['bounding_box']
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            
            if x1 <= x <= x2 and y1 <= y <= y2:
                confidence = coord_data['confidence']
                print(f"✓ Direct hit on {furniture_name} (confidence: {confidence:.2f})")
                return furniture_name
        
        # If no direct hit, find the closest furniture piece
        closest_furniture = None
        min_distance = float('inf')
        
        for furniture_name, coord_data in self.furniture_coordinates.items():
            center_x, center_y = coord_data['center']['x'], coord_data['center']['y']
            distance = ((x - center_x)**2 + (y - center_y)**2)**0.5
            
            if distance < min_distance:
                min_distance = distance
                closest_furniture = furniture_name
        
        # Return closest if within reasonable distance
        if closest_furniture and min_distance < 200:
            print(f"Closest furniture: {closest_furniture} (distance: {min_distance:.1f}px)")
            return closest_furniture
        
        print("No furniture found near click")
        return None
    
    
    def find_matching_3d_model(self, furniture_name):
        """Find a 3D model that matches the detected furniture"""
        furniture_base = os.path.splitext(furniture_name)[0]
        furniture_type = furniture_base.split('_')[0].lower()
        
        # Try exact match first
        for model_name, model_path in self.model_3d_files.items():
            model_base = os.path.splitext(model_name)[0]
            if furniture_base == model_base:
                return (model_name, model_path)
        
        # Try type match
        for model_name, model_path in self.model_3d_files.items():
            model_base = os.path.splitext(model_name)[0]
            model_type = model_base.split('_')[0].lower()
            
            if furniture_type == model_type:
                return (model_name, model_path)
        
        # Fallback for missing model types
        type_fallbacks = {
            'vase': 'plant',
            'pillow': 'couch',
            'items': 'light',
            'wardrobe': 'bookshelf',
        }
        
        fallback_type = type_fallbacks.get(furniture_type)
        if fallback_type:
            for model_name, model_path in self.model_3d_files.items():
                model_type = os.path.splitext(model_name)[0].split('_')[0].lower()
                if model_type == fallback_type:
                    print(f"Using fallback: {model_name} for {furniture_name}")
                    return (model_name, model_path)
        
        return None
    
    def show_available_furniture_regions(self):
        """Show available furniture regions that can be clicked"""
        if self.extracted_pieces:
            print("Available furniture regions to click:")
            for furniture_name in self.extracted_pieces.keys():
                # Check if there's a matching 3D model
                matching_model = self.find_matching_3d_model(furniture_name)
                if matching_model:
                    print(f"  - {furniture_name} → {matching_model[0]}")
                else:
                    print(f"  - {furniture_name} (no 3D model)")
        else:
            print("No furniture regions found")
    
    def on_key_press(self, event):
        """Handle keyboard input for model selection"""
        if event.key.isdigit():
            model_index = int(event.key) - 1
            model_list = list(self.model_3d_files.items())
            
            if 0 <= model_index < len(model_list):
                model_name, model_path = model_list[model_index]
                print(f"\nOpening 3D model {model_index + 1}: {model_name}")
                success = self.open_3d_model(model_path)
                if success:
                    print(f"Successfully opened 3D model: {model_name}")
                else:
                    print(f"Failed to open 3D model: {model_name}")
            else:
                print(f"\nInvalid model number. Available models: 1-{len(model_list)}")
                self.show_available_models()
        elif event.key == 'h' or event.key == 'H':
            print("\nKEYBOARD SHORTCUTS:")
            print("1-9: Open specific 3D model by number")
            print("h/H: Show this help")
            print("\nAVAILABLE MODELS:")
            self.show_available_models()
    
    def show_available_models(self):
        """Show all available 3D models"""
        if self.model_3d_files:
            print("Available 3D models:")
            for i, (name, path) in enumerate(self.model_3d_files.items(), 1):
                print(f"  {i}. {name}")
            print("Click 'Show All Models' button to open all models")
        else:
            print("No 3D models found in any directory")
    
    def show_all_models(self, event):
        """Open all available 3D models"""
        if not self.model_3d_files:
            print("No 3D models available")
            return
        
        print("Opening all 3D models...")
        for name, path in self.model_3d_files.items():
            print(f"Opening: {name}")
            self.open_3d_model(path)
    
    def toggle_segmentation(self, event):
        """Toggle segmentation overlay"""
        self.show_segmentation = not self.show_segmentation
        print(f"Segmentation overlay: {'ON' if self.show_segmentation else 'OFF'}")
        self.load_image()
    
    def next_image(self, event):
        """Load next image"""
        if self.available_images:
            self.current_image_index = (self.current_image_index + 1) % len(self.available_images)
            print(f"Moving to next image: {self.current_image_index + 1}/{len(self.available_images)}")
            self.load_image()
    
    def previous_image(self, event):
        """Load previous image"""
        if self.available_images:
            self.current_image_index = (self.current_image_index - 1) % len(self.available_images)
            print(f"Moving to previous image: {self.current_image_index + 1}/{len(self.available_images)}")
            self.load_image()
    
    def show_summary(self):
        """Display summary of available content"""
        print("\n📋 Interactive Furniture 3D Model Viewer")
        print(f"📁 Images: {len(self.original_images)} | 🎯 3D Models: {len(self.model_3d_files)} | 📍 Coordinates: {len(self.furniture_coordinates)}")
        print("\n🖱️  Click on furniture items in the image to view their 3D models")
        print("🔧 Use buttons below for additional controls")
        print("\nFurniture-to-Model Mappings:")
        if self.extracted_pieces:
            for furniture_name in self.extracted_pieces.keys():
                matching_model = self.find_matching_3d_model(furniture_name)
                if matching_model:
                    print(f"  ✓ {furniture_name} → {matching_model[0]}")
                else:
                    print(f"  ✗ {furniture_name} (no matching 3D model)")
        else:
            print("  No extracted furniture pieces found")
        
        print("\n" + "=" * 50)
    
    def run(self):
        """Run the interactive viewer"""
        if not self.available_images:
            print("No images found!")
            return
        
        self.show_summary()
        plt.show()

def main():
    base_dir = os.getcwd()
    print(f"Using base directory: {base_dir}")
    
    try:
        viewer = InteractiveFurnitureViewer(base_dir)
        viewer.run()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()