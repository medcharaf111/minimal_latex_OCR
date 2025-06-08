import pytesseract
from PIL import Image, ImageFilter, ImageEnhance, ImageTk
from pix2tex.cli import LatexOCR
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
import threading
import cloudinary
import cloudinary.uploader
import cloudinary.api
import requests
from io import BytesIO
import numpy as np
import torch
import queue
import cv2

# Cloudinary configuration
cloudinary.config(
    cloud_name = "dmgdghld7",
    api_key = "133444181494425",
    api_secret = "kU4ps9z1kqylkhGW-QU9ihOoXDY"
)

# Load LatexOCR model globally for efficiency
try:
    latexocr_model = LatexOCR()
except Exception as e:
    latexocr_model = None
    print(f"Failed to load LatexOCR model: {e}")

# Add a global reference to the Tk root
root = None

result_queue = queue.Queue()

last_preprocessed_img = [None]
last_tesseract_text = [None]

def preprocess_image(image):
    """Preprocess image for better OCR results"""
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    # Resize to a reasonable size for pix2tex (not too large)
    max_width = 1200
    if image.width > max_width:
        ratio = max_width / image.width
        new_size = (max_width, int(image.height * ratio))
        image = image.resize(new_size, Image.LANCZOS)
    # Increase contrast
    img_array = np.array(image)
    img_array = np.clip(img_array * 1.2, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)

def upscale_image_if_needed(image_path):
    """Upscale image using Cloudinary only if width < 800px"""
    img = Image.open(image_path)
    if img.width >= 800:
        return image_path  # No need to upscale
    try:
        upload_result = cloudinary.uploader.upload(image_path)
        upscaled_url = cloudinary.CloudinaryImage(upload_result['public_id']).build_url(
            transformation=[
                {'quality': 'auto:best'},
                {'fetch_format': 'auto'},
                {'width': 1600, 'crop': 'scale'},
                {'effect': 'upscale:2'}
            ]
        )
        response = requests.get(upscaled_url)
        upscaled_image = Image.open(BytesIO(response.content))
        temp_path = os.path.join(os.path.dirname(image_path), 'upscaled_' + os.path.basename(image_path))
        upscaled_image.save(temp_path)
        return temp_path
    except Exception as e:
        messagebox.showerror("Error", f"Failed to upscale image: {e}")
        return image_path

def downscale_if_needed(pil_img, max_width=1280, max_height=720):
    w, h = pil_img.size
    if w > max_width or h > max_height:
        scale = min(max_width / w, max_height / h)
        new_size = (int(w * scale), int(h * scale))
        return pil_img.resize(new_size, Image.LANCZOS)
    return pil_img

def crop_image_gui(image_path):
    """Open a simple cropping GUI and return the cropped image."""
    from tkinter import Toplevel, Canvas, NW
    img = Image.open(image_path)
    img = downscale_if_needed(img)  # Downscale before cropping
    crop_coords = {'x0': 0, 'y0': 0, 'x1': img.width, 'y1': img.height}
    rect = [None]
    def on_mouse_down(event):
        crop_coords['x0'], crop_coords['y0'] = event.x, event.y
        crop_coords['x1'], crop_coords['y1'] = event.x, event.y
        if rect[0] is not None:
            canvas.delete(rect[0])
        rect[0] = canvas.create_rectangle(event.x, event.y, event.x, event.y, outline='red', width=2)
    def on_mouse_move(event):
        crop_coords['x1'], crop_coords['y1'] = event.x, event.y
        if rect[0] is not None:
            canvas.coords(rect[0], crop_coords['x0'], crop_coords['y0'], event.x, event.y)
    def on_mouse_up(event):
        crop_coords['x1'], crop_coords['y1'] = event.x, event.y
        if rect[0] is not None:
            canvas.coords(rect[0], crop_coords['x0'], crop_coords['y0'], event.x, event.y)
        top.quit()  # Quit the event loop, but don't destroy yet
    top = Toplevel()
    top.title('Crop Image - Drag to select region')
    canvas = Canvas(top, width=img.width, height=img.height)
    canvas.pack()
    tk_img = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, anchor=NW, image=tk_img)
    canvas.bind('<Button-1>', on_mouse_down)
    canvas.bind('<B1-Motion>', on_mouse_move)
    canvas.bind('<ButtonRelease-1>', on_mouse_up)
    top.mainloop()
    top.destroy()  # Now destroy the window after mainloop exits
    x0, y0, x1, y1 = crop_coords['x0'], crop_coords['y0'], crop_coords['x1'], crop_coords['y1']
    box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    cropped = img.crop(box)
    cropped = downscale_if_needed(cropped)  # Downscale after cropping
    # Save cropped image for further processing
    cropped_path = os.path.join(os.path.dirname(image_path), 'cropped_' + os.path.basename(image_path))
    cropped.save(cropped_path)
    return cropped_path

def deskew_image(cv_img):
    # Compute skew angle and rotate to deskew
    coords = np.column_stack(np.where(cv_img < 255))
    if coords.shape[0] == 0:
        return cv_img  # nothing to deskew
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = cv_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(cv_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def advanced_preprocess_image(image):
    """Very robust preprocessing: background removal, grayscale, CLAHE, adaptive thresholding, denoising, morph, deskew, sharpening, resizing."""
    img_np = np.array(image)
    # 1. Background removal (estimate background with large Gaussian blur and subtract)
    if len(img_np.shape) == 3:
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_np
    bg = cv2.GaussianBlur(img_gray, (51, 51), 0)
    norm_img = cv2.divide(img_gray, bg, scale=255)
    # 2. CLAHE (local contrast enhancement)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    norm_img = clahe.apply(norm_img)
    # 3. Aggressive denoising (non-local means)
    norm_img = cv2.fastNlMeansDenoising(norm_img, None, h=30, templateWindowSize=7, searchWindowSize=21)
    # 4. Adaptive thresholding
    bin_img = cv2.adaptiveThreshold(norm_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 15)
    # 5. Morphological opening (remove small noise)
    kernel = np.ones((2,2), np.uint8)
    morph = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    # 6. Morphological closing (fill small holes)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    # 7. Deskew
    morph = deskew_image(morph)
    # 8. Sharpening
    sharp_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    morph = cv2.filter2D(morph, -1, sharp_kernel)
    # 9. Invert if needed (text should be dark on light)
    white_pix = np.sum(morph == 255)
    black_pix = np.sum(morph == 0)
    if white_pix < black_pix:
        morph = cv2.bitwise_not(morph)
    # 10. Resize to optimal width
    target_width = 1000
    if morph.shape[1] < target_width:
        scale = target_width / morph.shape[1]
        morph = cv2.resize(morph, (target_width, int(morph.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(morph)

def create_gui():
    global result_text, root, select_btn, busy_label, retry_btn
    root = tk.Tk()
    root.title("Handwriting OCR with LaTeX Support")
    root.geometry('700x500')
    select_btn = tk.Button(root, text="Select Image", command=select_image, font=("Arial", 14))
    select_btn.pack(pady=10)
    retry_btn = tk.Button(root, text="Retry LaTeX (pix2tex)", command=retry_pix2tex, font=("Arial", 12), state=tk.DISABLED)
    retry_btn.pack(pady=5)
    busy_label = tk.Label(root, text="", font=("Arial", 12), fg="red")
    busy_label.pack()
    result_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=25, font=("Consolas", 10))
    result_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    root.protocol("WM_DELETE_WINDOW", on_close)
    poll_queue()  # Start polling the queue
    root.mainloop()

def on_close():
    if getattr(root, 'processing', False):
        show_messagebox_safe('info', "Busy", "Please wait until processing is finished before closing.")
    else:
        root.destroy()

def set_processing_state(is_processing):
    root.processing = is_processing
    if is_processing:
        select_btn.config(state=tk.DISABLED)
        retry_btn.config(state=tk.DISABLED)
        busy_label.config(text="Processing... Please wait.")
    else:
        select_btn.config(state=tk.NORMAL)
        retry_btn.config(state=tk.NORMAL if last_preprocessed_img[0] is not None else tk.DISABLED)
        busy_label.config(text="")

def show_messagebox_safe(kind, title, message):
    def show():
        if root and root.winfo_exists():
            if kind == 'info':
                messagebox.showinfo(title, message)
            elif kind == 'error':
                messagebox.showerror(title, message)
    if root:
        try:
            root.after(0, show)
        except RuntimeError:
            pass  # Mainloop has exited

def run_ocr(image_path):
    print("Starting run_ocr")
    try:
        upscaled_path = upscale_image_if_needed(image_path)
        print(f"Upscaling done: {upscaled_path}")
        img = Image.open(upscaled_path)
        img = advanced_preprocess_image(img)
        print("Preprocessing done")
        text = pytesseract.image_to_string(img, lang='eng')
        print(f"Tesseract OCR done: {text[:30]}")
        if latexocr_model is not None:
            latex = latexocr_model(img)
            print(f"LatexOCR done: {latex[:30]}")
        else:
            latex = "[LatexOCR model not loaded]"
        if upscaled_path != image_path:
            show_messagebox_safe('info', "Upscaled Image", f"Upscaled image saved at: {upscaled_path}")
        print("run_ocr finished successfully")
        return text, latex
    except Exception as e:
        print(f"Exception in run_ocr: {e}")
        show_messagebox_safe('error', "Error", f"OCR processing failed: {str(e)}")
        return "", ""

def show_preprocessed_preview(pre_img, proceed_callback):
    from tkinter import Toplevel, Button, Label
    max_preview_w, max_preview_h = 900, 600
    w, h = pre_img.width, pre_img.height
    scale = min(max_preview_w / w, max_preview_h / h, 1.0)
    if scale < 1.0:
        preview_img = pre_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    else:
        preview_img = pre_img
    preview_win = Toplevel(root)
    preview_win.title("Preprocessed Image Preview")
    preview_win.geometry(f"{preview_img.width+40}x{preview_img.height+80}")
    tk_pre_img = ImageTk.PhotoImage(preview_img)
    lbl = Label(preview_win, image=tk_pre_img)
    lbl.image = tk_pre_img
    lbl.pack(pady=10)
    def on_proceed():
        preview_win.destroy()
        proceed_callback()
    def on_cancel():
        preview_win.destroy()
        set_processing_state(False)
    btn_frame = tk.Frame(preview_win)
    btn_frame.pack(pady=10)
    Button(btn_frame, text="Proceed with OCR", command=on_proceed, font=("Arial", 12)).pack(side=tk.LEFT, padx=10)
    Button(btn_frame, text="Cancel", command=on_cancel, font=("Arial", 12)).pack(side=tk.LEFT, padx=10)
    preview_win.transient(root)
    preview_win.grab_set()
    root.wait_window(preview_win)

def select_image():
    file_path = filedialog.askopenfilename(
        filetypes=[('Image Files', '*.png;*.jpg;*.jpeg;*.bmp;*.tiff')]
    )
    if file_path:
        # Ask user if they want to crop
        if messagebox.askyesno("Crop", "Do you want to crop the image before processing?"):
            file_path = crop_image_gui(file_path)
        result_text.delete(1.0, tk.END)
        set_processing_state(True)
        upscaled_path = upscale_image_if_needed(file_path)
        img = Image.open(upscaled_path)
        def proceed_ocr():
            threading.Thread(target=threaded_ocr_from_img, args=(file_path, img), daemon=True).start()
        # Directly proceed, no preprocessing preview
        proceed_ocr()

def poll_queue():
    try:
        while True:
            result = result_queue.get_nowait()
            if result['type'] == 'result':
                text, latex, out_path = result['text'], result['latex'], result['out_path']
                result_text.delete(1.0, tk.END)
                result_text.insert(tk.END, f"Plain OCR Text (Tesseract):\n{text}\n\nLaTeX (pix2tex):\n{latex}")
                show_messagebox_safe('info', "Done", f"OCR and LaTeX extraction complete. Output saved to {out_path}")
                set_processing_state(False)
            elif result['type'] == 'error':
                show_messagebox_safe('error', "Error", result['message'])
                set_processing_state(False)
    except queue.Empty:
        pass
    if root and root.winfo_exists():
        root.after(100, poll_queue)

def threaded_ocr_from_img(file_path, img):
    print(f"Starting threaded_ocr_from_img for {file_path}")
    try:
        text = pytesseract.image_to_string(img, lang='eng')
        latex = latexocr_model(img) if latexocr_model is not None else "[LatexOCR model not loaded]"
        last_preprocessed_img[0] = img
        last_tesseract_text[0] = text
        out_path = os.path.join(os.path.dirname(file_path), 'output.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(f"Plain OCR Text (Tesseract):\n{text}\n\nLaTeX (pix2tex):\n{latex}")
        result_queue.put({'type': 'result', 'text': text, 'latex': latex, 'out_path': out_path})
        print("Result put in queue")
    except Exception as e:
        print(f"Exception in threaded_ocr_from_img: {e}")
        result_queue.put({'type': 'error', 'message': f"An error occurred: {e}"})
        print("Error put in queue")

def retry_pix2tex():
    if last_preprocessed_img[0] is not None:
        set_processing_state(True)
        def do_retry():
            try:
                latex = latexocr_model(last_preprocessed_img[0]) if latexocr_model is not None else "[LatexOCR model not loaded]"
                result_text.delete(1.0, tk.END)
                result_text.insert(tk.END, f"Plain OCR Text (Tesseract):\n{last_tesseract_text[0]}\n\nLaTeX (pix2tex):\n{latex}")
                set_processing_state(False)
            except Exception as e:
                show_messagebox_safe('error', "Error", f"Retry failed: {e}")
                set_processing_state(False)
        root.after(100, do_retry)

if __name__ == "__main__":
    try:
        create_gui()
    except Exception as e:
        print(f"Exception in main thread: {e}")
    print("Main loop exited") 