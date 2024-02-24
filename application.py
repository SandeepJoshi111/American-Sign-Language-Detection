import tkinter as tk
from tkinter import ttk
import subprocess

def run_script(script_path):
    subprocess.Popen(['python', script_path])

def switch_to_script1():
    run_script("see-num.py")

def switch_to_script2():
    run_script("see.py")

root = tk.Tk()
root.title("ASL Detection")

# Set the width and height of the root window
root.geometry("400x400")

# Create a frame for ASL detection
frame_asl = ttk.Frame(root)
label_asl = ttk.Label(frame_asl, text="American Sign", font=("Courier", 25))
label_asl.pack(pady=5)
label_asl = ttk.Label(frame_asl, text="Language Detection", font=("Courier", 25))
label_asl.pack(pady=5)
label_asl = ttk.Label(frame_asl, text="(WITH ARITHMETIC OPERATIONS)", font=("Courier", 15))
label_asl.pack(pady=5)

# Place the ASL detection frame at the center
frame_asl.place(relx=0.5, rely=0.25, anchor=tk.CENTER)

# Create a frame for the buttons
frame_buttons = ttk.Frame(root)
frame_buttons.place(relx=0.5, rely=0.6, anchor=tk.CENTER)

# Create buttons to switch between scripts
btn_script2 = ttk.Button(frame_buttons, text="Predict Sign", command=switch_to_script2, width=25, style='Large.TButton')
btn_script2.grid(row=0, column=0, padx=5, pady=20)

btn_script1 = ttk.Button(frame_buttons, text="Arithmetic Operation", command=switch_to_script1, width=25, style='Large.TButton')
btn_script1.grid(row=1, column=0, padx=5, pady=20)

# Define a custom style for larger buttons
style = ttk.Style()
style.configure('Large.TButton', font=('Courier', 14))  # Adjust font size as needed

root.mainloop()
