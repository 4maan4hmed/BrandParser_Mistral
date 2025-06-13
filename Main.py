import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk # Used for displaying images in Tkinter
import cv2
from paddleocr import PaddleOCR
from collections import defaultdict
import numpy as np
import os
import threading # To run video capture and OCR in a separate thread, avoiding freezing UI
import json # For saving data to JSON
import datetime # For adding timestamps to data

# --- Dummy implementations for comparision.py and data_operation.py ---
# In a real scenario, these would be in separate files and handle actual data.

# comparision.py
def compare(ocr_text_string):
    """
    Dummy comparison function.
    Returns a dummy item number based on keywords.
    In a real app, this would use more sophisticated matching against an inventory.
    """
    if not isinstance(ocr_text_string, str):
        return "UNKNOWN_ITEM"
    
    ocr_text_string = ocr_text_string.lower()
    if "apple" in ocr_text_string or "aapl" in ocr_text_string:
        return "ITEM001"
    elif "banana" in ocr_text_string or "bana" in ocr_text_string:
        return "ITEM002"
    elif "orange" in ocr_text_string or "orgn" in ocr_text_string:
        return "ITEM003"
    else:
        return "UNKNOWN_ITEM"

# data_operation.py
def get_item_details(item_num):
    """
    Dummy function to get item details based on a dummy item number.
    In a real app, this would query a database or a comprehensive inventory list.
    """
    items_db = {
        "ITEM001": {"item_name": "Apple (Red)"},
        "ITEM002": {"item_name": "Banana (Yellow)"},
        "ITEM003": {"item_name": "Orange (Citrus)"},
        "UNKNOWN_ITEM": {"item_name": "Unidentified Item"} # Fallback for unrecognized items
    }
    return items_db.get(item_num, {"item_name": "Unknown Item Details"})

# --- End Dummy Implementations ---


# Path for temporary OCR text file (can be removed if not needed)
PATH_TEMP_OCR_TEXT = "output_ocr_text.txt"
# Path for the main dataset JSON file
DATASET_FILE = "item_dataset.json"

class OCRProcessor:
    def __init__(self):
        # Initialize PaddleOCR. Setting use_gpu=False for broader compatibility.
        # Ensure PaddleOCR model is downloaded or available.
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
        
        # Stores frequency of each detected word in the CURRENT LIVE BUFFER
        self.current_buffer_text_frequencies = defaultdict(int)
        # Stores the highest confidence encountered for each word in the CURRENT LIVE BUFFER
        self.current_buffer_best_confidences = {}
        # Stores the currently detected item name (from comparison, based on accumulated total OCR)
        self.detected_item = ""
        # Flag to indicate if any text has been detected in the current LIVE BUFFER session
        self.has_detected_text_in_buffer = False

    def process_ocr_result(self, result):
        """
        Processes the raw OCR result from PaddleOCR, updating text frequencies
        and best confidences for the current live buffer.
        """
        if not result or not isinstance(result, list):
            return
        
        if result and isinstance(result[0], list):
            for detection in result[0]:
                if detection is None or len(detection) != 2:
                    continue
                box, text_info = detection
                if text_info is None or len(text_info) != 2:
                    continue
                
                text, confidence = text_info
                text = text.strip()
                if not text:
                    continue

                self.current_buffer_text_frequencies[text] += 1
                if text not in self.current_buffer_best_confidences or \
                   confidence > self.current_buffer_best_confidences[text]:
                    self.current_buffer_best_confidences[text] = confidence
                
                self.has_detected_text_in_buffer = True

    def run(self, frame, total_ocr_string_for_comparison=""):
        """
        Performs OCR on a single video frame and updates live buffer states.
        Also runs comparison on the total OCR string provided.
        """
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.ocr.ocr(np.array(img), cls=True)

        if result:
            self.process_ocr_result(result)
            
        # Run comparison logic on the combined OCR string (from OCRApp)
        # This string represents ALL OCR attempts for the current item
        if total_ocr_string_for_comparison:
            detected_item_num = compare(total_ocr_string_for_comparison)
            item_details = get_item_details(detected_item_num)
            if item_details and 'item_name' in item_details:
                self.detected_item = item_details['item_name']
            else:
                self.detected_item = "Item not recognized"
        else:
            self.detected_item = "No OCR collected for comparison yet" # Indicate no basis for comparison


    def get_current_buffer_ocr_string(self):
        """
        Returns a single string of unique OCR texts from the current live buffer,
        sorted by frequency and then confidence.
        """
        if not self.current_buffer_text_frequencies:
            return ""

        results_list = []
        for text, freq in self.current_buffer_text_frequencies.items():
            conf = self.current_buffer_best_confidences.get(text, 0.0)
            results_list.append((text, freq, conf))

        # Sort by frequency (desc) then confidence (desc)
        sorted_unique_texts = sorted(results_list, key=lambda x: (-x[1], -x[2]))
        
        # Return only the text strings, joined by spaces
        return " ".join([text for text, freq, conf in sorted_unique_texts])

    def reset_buffer(self):
        """
        Resets only the current live OCR buffer (text_frequencies, best_confidences).
        """
        self.current_buffer_text_frequencies.clear()
        self.current_buffer_best_confidences.clear()
        self.has_detected_text_in_buffer = False
        print("Live OCR buffer reset.")

    def reset_all(self):
        """
        Resets all OCR data managed by OCRProcessor.
        """
        self.reset_buffer()
        self.detected_item = ""
        # Remove the temporary text file if it exists
        if os.path.exists(PATH_TEMP_OCR_TEXT):
            os.remove(PATH_TEMP_OCR_TEXT)
        print("OCRProcessor fully reset. Ready for a new item.")


# --- Tkinter Application Class ---
class OCRApp:
    # Predefined list of storage recommendations
    STORAGE_RECOMMENDATIONS = [
        "Warehouse Shelf",
        "Cold Storage",
        "Dry Storage",
        "Fragile Goods Area",
        "Hazardous Material Storage"
    ]

    def __init__(self, window, window_title="Real-time OCR Item Detection"):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1000x700") # Set initial window size
        self.window.resizable(True, True) # Allow window resizing

        # Initialize the OCR processor and webcam
        self.ocr_processor = OCRProcessor()
        self.video_capture = cv2.VideoCapture(0) # 0 for default webcam

        # Check if webcam opened successfully
        if not self.video_capture.isOpened():
            messagebox.showerror("Camera Error", "Failed to open webcam. Please ensure it's connected and not in use.")
            self.window.destroy()
            return

        self.is_processing_frame = False # Flag to prevent multiple OCR threads on same frame
        # This list will store each OCR "attempt" as a separate string for the current item
        self.current_item_ocr_attempts_list = [] 

        # Setup GUI elements
        self.create_widgets()
        
        # Setup keyboard bindings
        self.setup_keyboard_bindings()

        # Start video stream update. 'delay' controls update frequency.
        self.delay = 10 # milliseconds, updates every 10ms (approx 100 FPS)
        self.update_video_feed()
        # Start updating the detected item label and accumulated text labels
        self.update_info_labels() 

        # Handle window closing to release webcam resources
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_keyboard_bindings(self):
        """
        Setup keyboard shortcuts for the application.
        """
        # Make the window focusable to receive key events
        self.window.focus_set()
        
        # Bind keyboard events
        self.window.bind('<space>', self._keyboard_add_to_item)
        self.window.bind('<Return>', self._keyboard_finalize_item)
        self.window.bind('<Delete>', self._keyboard_clear_buffer)
        self.window.bind('<BackSpace>', self._keyboard_clear_buffer)  # Alternative for some systems
        
        # Make sure the window can receive focus
        self.window.bind('<Button-1>', self._focus_window)
        self.window.bind('<FocusIn>', self._on_focus_in)

    def _focus_window(self, event=None):
        """Focus the main window to ensure it receives keyboard events."""
        self.window.focus_set()

    def _on_focus_in(self, event=None):
        """Handle focus events."""
        pass

    def _keyboard_add_to_item(self, event=None):
        """Keyboard shortcut handler for adding current buffer to item (Space key)."""
        if self.add_to_item_button.cget('state') == 'normal':
            self._add_current_ocr_to_item()
        return 'break'  # Prevent default space behavior

    def _keyboard_finalize_item(self, event=None):
        """Keyboard shortcut handler for finalizing item (Enter key)."""
        if self.finalize_item_button.cget('state') == 'normal':
            self._finalize_item_and_start_new()
        return 'break'  # Prevent default enter behavior

    def _keyboard_clear_buffer(self, event=None):
        """Keyboard shortcut handler for clearing current buffer (Delete key)."""
        self._clear_current_buffer()
        return 'break'  # Prevent default delete behavior

    def _clear_current_buffer(self):
        """
        Clears the current live OCR buffer without adding it to the item.
        """
        if self.ocr_processor.get_current_buffer_ocr_string():
            self.ocr_processor.reset_buffer()
            messagebox.showinfo("Buffer Cleared", "Current live OCR buffer has been cleared.")
            self.update_info_labels() # Force update labels immediately
        else:
            messagebox.showinfo("No Buffer", "Live OCR buffer is already empty.")

    def create_widgets(self):
        # Configure grid for responsive layout
        self.window.grid_columnconfigure(0, weight=3) # Video column
        self.window.grid_columnconfigure(1, weight=1) # Controls column
        self.window.grid_rowconfigure(0, weight=1)

        # Video Frame (left column)
        video_frame = ttk.LabelFrame(self.window, text="Live Camera Feed", padding="10")
        video_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        video_frame.grid_rowconfigure(0, weight=1)
        video_frame.grid_columnconfigure(0, weight=1)

        self.video_label = ttk.Label(video_frame)
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # Control Frame (right column)
        control_frame = ttk.LabelFrame(self.window, text="Controls & Detection", padding="15")
        control_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        # Configure rows within control_frame for vertical stacking
        control_frame.grid_columnconfigure(0, weight=1) # Make column expandable
        for i in range(9): # Updated to accommodate new widgets
            control_frame.grid_rowconfigure(i, weight=0) # Don't stretch rows by default

        # Keyboard shortcuts info
        shortcuts_frame = ttk.LabelFrame(control_frame, text="Keyboard Shortcuts", padding="5")
        shortcuts_frame.grid(row=0, column=0, pady=(0, 10), padx=5, sticky="ew")
        
        shortcuts_text = "Space: Add to Item\nEnter: Finalize Item\nDelete: Clear Buffer"
        shortcuts_label = ttk.Label(shortcuts_frame, text=shortcuts_text, 
                                   font=("Arial", 8), justify=tk.LEFT)
        shortcuts_label.pack()

        # Detected Item Label (based on total OCR)
        self.detected_item_label = ttk.Label(control_frame, 
                                            text="Waiting for detection...", 
                                            font=("Arial", 14, "bold"), 
                                            wraplength=250, 
                                            justify=tk.CENTER,
                                            background="#e0e0e0", 
                                            relief=tk.RIDGE, 
                                            padding=10)
        self.detected_item_label.grid(row=1, column=0, pady=10, padx=10, sticky="ew")

        # Label for Live OCR Buffer
        ttk.Label(control_frame, text="Live OCR Buffer:").grid(row=2, column=0, sticky="w", padx=10, pady=(10, 0))
        self.live_buffer_label = ttk.Label(control_frame,
                                           text="No text yet...",
                                           font=("Arial", 9),
                                           wraplength=250,
                                           justify=tk.LEFT,
                                           background="#f8f8f8", # Lighter background
                                           relief=tk.GROOVE,
                                           padding=5)
        self.live_buffer_label.grid(row=3, column=0, pady=(0, 10), padx=10, sticky="ew")

        # Button to add current buffer to total OCR
        self.add_to_item_button = ttk.Button(control_frame, 
                                            text="Add to Current Item's OCR (Space)", 
                                            command=self._add_current_ocr_to_item)
        self.add_to_item_button.grid(row=4, column=0, pady=(10, 5), padx=10, sticky="ew")
        self.add_to_item_button.config(state=tk.DISABLED) # Start disabled

        # Button to clear current buffer
        self.clear_buffer_button = ttk.Button(control_frame, 
                                             text="Clear Current Buffer (Delete)", 
                                             command=self._clear_current_buffer)
        self.clear_buffer_button.grid(row=5, column=0, pady=(5, 5), padx=10, sticky="ew")

        # Label for Total OCR for Current Item
        ttk.Label(control_frame, text="Total OCR for Current Item:").grid(row=6, column=0, sticky="w", padx=10, pady=(10, 0))
        self.total_ocr_label = ttk.Label(control_frame,
                                          text="No total OCR yet...",
                                          font=("Arial", 9, "italic"),
                                          wraplength=250,
                                          justify=tk.LEFT,
                                          background="#f0f0f0",
                                          relief=tk.GROOVE,
                                          padding=5)
        self.total_ocr_label.grid(row=7, column=0, pady=(0, 10), padx=10, sticky="ew")
        
        # Button to finalize item and start new
        self.finalize_item_button = ttk.Button(control_frame, 
                                               text="Finalize Item & Start New (Enter)", 
                                               command=self._finalize_item_and_start_new)
        self.finalize_item_button.grid(row=8, column=0, pady=(10, 5), padx=10, sticky="ew")
        self.finalize_item_button.config(state=tk.DISABLED) # Start disabled

        # Optional: Button to save temporary results to text file
        self.save_temp_ocr_button = ttk.Button(control_frame, 
                                              text="Save Live Buffer to Text File", 
                                              command=self._save_temp_ocr_to_file)
        self.save_temp_ocr_button.grid(row=9, column=0, pady=(5, 10), padx=10, sticky="ew")
        self.save_temp_ocr_button.config(state=tk.DISABLED) # Start disabled

    def update_video_feed(self):
        """
        Captures a frame from the webcam, displays it, and triggers OCR processing.
        """
        ret, frame = self.video_capture.read()
        if ret:
            # Get current size of the video_label to resize frame dynamically
            label_width = self.video_label.winfo_width()
            label_height = self.video_label.winfo_height()

            if label_width > 0 and label_height > 0:
                frame = cv2.resize(frame, (label_width, label_height))

            # Convert OpenCV image (BGR) to RGBA for Pillow, then to PhotoImage for Tkinter
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            self.current_photo = ImageTk.PhotoImage(image=Image.fromarray(cv2image))
            self.video_label.imgtk = self.current_photo # Keep a reference to prevent garbage collection
            self.video_label.configure(image=self.current_photo)

            # Only run OCR on a new frame if previous OCR thread is not still processing
            if not self.is_processing_frame:
                self.is_processing_frame = True
                # Pass the combined OCR string for comparison to OCRProcessor
                # Join the list elements to form a single string for comparison
                total_ocr_for_comparison_str = " ".join(self.current_item_ocr_attempts_list)
                threading.Thread(target=self._run_ocr_on_frame, 
                                 args=(frame.copy(), total_ocr_for_comparison_str)).start()
        
        # Schedule the next video feed update
        self.window.after(self.delay, self.update_video_feed)

    def _run_ocr_on_frame(self, frame_copy, total_ocr_for_comparison):
        """
        Internal method to run OCR on a frame. Executed in a separate thread.
        """
        try:
            self.ocr_processor.run(frame_copy, total_ocr_for_comparison)
        finally:
            self.is_processing_frame = False # Reset flag after processing

    def update_info_labels(self):
        """
        Updates the detected item label, live buffer text, and total OCR text periodically.
        Also manages button states.
        """
        # Update detected item label
        if self.ocr_processor.detected_item:
            self.detected_item_label.config(text=f"Detected: {self.ocr_processor.detected_item}", background="#a0e0a0")
        else:
            self.detected_item_label.config(text="Waiting for detection...", background="#e0e0e0")

        # Update live buffer label
        live_buffer_text = self.ocr_processor.get_current_buffer_ocr_string()
        if live_buffer_text:
            self.live_buffer_label.config(text=live_buffer_text)
            self.add_to_item_button.config(state=tk.NORMAL) # Enable "Add to Item" if buffer has text
            self.save_temp_ocr_button.config(state=tk.NORMAL) # Enable "Save Live Buffer"
        else:
            self.live_buffer_label.config(text="No text yet...")
            self.add_to_item_button.config(state=tk.DISABLED) # Disable "Add to Item"
            self.save_temp_ocr_button.config(state=tk.DISABLED) # Disable "Save Live Buffer"

        # Update total OCR label
        if self.current_item_ocr_attempts_list:
            # Show a truncated version of the joined attempts for display
            joined_attempts_display = " ".join(self.current_item_ocr_attempts_list)
            if len(joined_attempts_display) > 100: # Truncate for display purposes
                display_text = joined_attempts_display[:97] + "..."
            else:
                display_text = joined_attempts_display
            self.total_ocr_label.config(text=display_text)
            self.finalize_item_button.config(state=tk.NORMAL) # Enable "Finalize Item" if total OCR exists
        else:
            self.total_ocr_label.config(text="No total OCR yet...")
            self.finalize_item_button.config(state=tk.DISABLED) # Disable "Finalize Item"

        # Schedule the next update
        self.window.after(200, self.update_info_labels) # Update every 200ms

    def _add_current_ocr_to_item(self):
        """
        Takes the current live OCR buffer, appends it as a new attempt
        to the list of OCR attempts for the current item, and then clears the live buffer.
        """
        current_buffer_text = self.ocr_processor.get_current_buffer_ocr_string()
        if not current_buffer_text:
            messagebox.showwarning("No Data", "No new OCR text in the live buffer to add to the item.")
            return

        # Append the current buffer text as a new attempt
        self.current_item_ocr_attempts_list.append(current_buffer_text)
        
        self.ocr_processor.reset_buffer() # Clear the live buffer
        messagebox.showinfo("OCR Added", "Live OCR text added as a new attempt to current item.")
        self.update_info_labels() # Force update labels immediately
        
        # Return focus to main window after dialog
        self.window.focus_set()

    def _show_item_details_dialog(self):
        """
        Displays a custom dialog to collect item details (company, item name, category, storage).
        Returns a dictionary of collected data or None if canceled.
        """
        dialog = tk.Toplevel(self.window)
        dialog.title("Enter Item Details")
        dialog.transient(self.window) # Make dialog transient to main window
        dialog.grab_set() # Grab all events, making main window unresponsive
        dialog.geometry("350x300") # Set dialog size
        dialog.resizable(False, False)

        # Center the dialog
        self.window.update_idletasks()
        x = self.window.winfo_x() + (self.window.winfo_width() // 2) - (dialog.winfo_width() // 2)
        y = self.window.winfo_y() + (self.window.winfo_height() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

        # Frame for inputs
        input_frame = ttk.Frame(dialog, padding="15")
        input_frame.pack(fill=tk.BOTH, expand=True)

        # Dictionary to store collected data
        data = {}

        # --- Company Name ---
        ttk.Label(input_frame, text="Company Name:").grid(row=0, column=0, sticky="w", pady=5)
        company_entry = ttk.Entry(input_frame, width=30)
        company_entry.grid(row=0, column=1, sticky="ew", pady=5, padx=5)

        # --- Item Name ---
        ttk.Label(input_frame, text="Item Name:").grid(row=1, column=0, sticky="w", pady=5)
        item_name_entry = ttk.Entry(input_frame, width=30)
        item_name_entry.grid(row=1, column=1, sticky="ew", pady=5, padx=5)

        # --- Category ---
        ttk.Label(input_frame, text="Category:").grid(row=2, column=0, sticky="w", pady=5)
        category_entry = ttk.Entry(input_frame, width=30)
        category_entry.grid(row=2, column=1, sticky="ew", pady=5, padx=5)

        # --- Storage Recommendation ---
        ttk.Label(input_frame, text="Storage Rec.:").grid(row=3, column=0, sticky="w", pady=5)
        storage_var = tk.StringVar(dialog)
        storage_combobox = ttk.Combobox(input_frame, textvariable=storage_var,
                                        values=self.STORAGE_RECOMMENDATIONS, state="readonly", width=27)
        storage_combobox.grid(row=3, column=1, sticky="ew", pady=5, padx=5)
        # Set a default value if desired
        if self.STORAGE_RECOMMENDATIONS:
            storage_combobox.set(self.STORAGE_RECOMMENDATIONS[0]) # Default to first option

        # --- Buttons ---
        def on_save():
            comp_name = company_entry.get().strip()
            itm_name = item_name_entry.get().strip()
            cat = category_entry.get().strip()
            storage_rec = storage_var.get().strip()

            if not comp_name or not itm_name or not cat or not storage_rec:
                messagebox.showwarning("Missing Information", "Please fill in all fields.", parent=dialog)
                return

            data["company_name"] = comp_name
            data["item_name"] = itm_name
            data["category"] = cat
            data["storage_recommendation"] = storage_rec
            dialog.destroy() # Close the dialog

        def on_cancel():
            dialog.destroy() # Close the dialog without saving data

        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)

        save_button = ttk.Button(button_frame, text="Save", command=on_save)
        save_button.pack(side=tk.LEFT, padx=5)
        cancel_button = ttk.Button(button_frame, text="Cancel", command=on_cancel)
        cancel_button.pack(side=tk.LEFT, padx=5)

        # Focus on the first entry field
        company_entry.focus_set()

        self.window.wait_window(dialog) # Wait for dialog to close
        return data if data else None # Return collected data or None if canceled

    def _finalize_item_and_start_new(self):
        """
        Finalizes the current item: prompts for name, saves to JSON,
        and completely resets for a new item.
        """
        # Ensure any remaining live buffer text is added before finalizing
        if self.ocr_processor.get_current_buffer_ocr_string():
            self._add_current_ocr_to_item() 

        if not self.current_item_ocr_attempts_list:
            messagebox.showwarning("No Data", "No OCR data collected for this item to finalize.")
            # Return focus to main window after dialog
            self.window.focus_set()
            return

        # Show the custom dialog to get item details
        item_details = self._show_item_details_dialog()
        
        if item_details: # Check if user clicked 'Save' and provided data
            item_data = {
                "company_name": item_details["company_name"],
                "item_name": item_details["item_name"],
                "category": item_details["category"],
                "storage_recommendation": item_details["storage_recommendation"],
                "ocr_text_list": self.current_item_ocr_attempts_list, # List of OCR attempt strings
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            try:
                # Read existing data
                if os.path.exists(DATASET_FILE) and os.path.getsize(DATASET_FILE) > 0:
                    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
                        dataset = json.load(f)
                else:
                    dataset = [] # Initialize as empty list if file doesn't exist or is empty
                
                # Append new item data
                dataset.append(item_data)
                
                # Write updated data back to file
                with open(DATASET_FILE, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=4)
                
                messagebox.showinfo("Success", f"'{item_details['item_name']}' data saved to {DATASET_FILE}. Ready for new item.")
                self.ocr_processor.reset_all() # Reset OCR data for the next item
                self.current_item_ocr_attempts_list = [] # Reset the list of attempts for the next item
            except json.JSONDecodeError:
                messagebox.showerror("Error", f"Could not read existing {DATASET_FILE}. File might be corrupted.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save item data: {e}")
        else:
            messagebox.showwarning("Canceled", "Item details input canceled. Data not saved.")
        
        # Return focus to main window after dialog
        self.window.focus_set()

    def _save_temp_ocr_to_file(self):
        """
        Saves the current live OCR buffer text to a simple text file.
        This is separate from the dataset JSON save and total item OCR.
        """
        live_buffer_text = self.ocr_processor.get_current_buffer_ocr_string()
        if not live_buffer_text:
            messagebox.showwarning("No Data", "No OCR text in live buffer to save to text file.")
            # Return focus to main window after dialog
            self.window.focus_set()
            return

        try:
            with open(PATH_TEMP_OCR_TEXT, 'w', encoding='utf-8') as f:
                f.write("=== Current Live OCR Buffer Texts ===\n\n")
                f.write(live_buffer_text + "\n")
            messagebox.showinfo("Saved", f"Current live OCR buffer texts saved to {PATH_TEMP_OCR_TEXT}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save temporary OCR text: {e}")
        
        # Return focus to main window after dialog
        self.window.focus_set()

    def on_closing(self):
        """
        Handles the window closing event, releasing webcam resources.
        """
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            if self.video_capture.isOpened():
                self.video_capture.release() # Release webcam
            self.window.destroy() # Close Tkinter window


# --- Main execution block ---
if __name__ == "__main__":
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop() # Start the Tkinter event loop