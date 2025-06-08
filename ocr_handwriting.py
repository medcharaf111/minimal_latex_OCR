import pytesseract
from PIL import Image
from pix2tex.cli import LatexOCR
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
import threading

# Explicitly set the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def run_ocr(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang='eng')
    model = LatexOCR()
    latex = model(img)
    return text, latex

def threaded_ocr(file_path):
    try:
        text, latex = run_ocr(file_path)
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"Plain OCR Text (Tesseract):\n{text}\n\nLaTeX (pix2tex):\n{latex}")
        # Save to output.txt in the same directory as the image
        out_path = os.path.join(os.path.dirname(file_path), 'output.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(f"Plain OCR Text (Tesseract):\n{text}\n\nLaTeX (pix2tex):\n{latex}")
        messagebox.showinfo("Done", f"OCR and LaTeX extraction complete. Output saved to {out_path}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def select_image():
    file_path = filedialog.askopenfilename(
        filetypes=[('Image Files', '*.png;*.jpg;*.jpeg;*.bmp;*.tiff')]
    )
    if file_path:
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, "Processing... Please wait.")
        threading.Thread(target=threaded_ocr, args=(file_path,), daemon=True).start()

def create_gui():
    global result_text
    root = tk.Tk()
    root.title("Handwriting OCR with LaTeX Support")
    root.geometry('700x500')

    select_btn = tk.Button(root, text="Select Image", command=select_image, font=("Arial", 14))
    select_btn.pack(pady=10)

    result_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=25, font=("Consolas", 10))
    result_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    root.mainloop()

if __name__ == "__main__":
    create_gui() 