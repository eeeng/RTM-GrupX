import os
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import mne
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class EEGDataHandler:
    def __init__(self, data_path):
        self.data_path = data_path
        self.raw = None

    def load_data(self):
        try:
            if os.path.exists(self.data_path):
                self.raw = mne.io.read_raw_edf(self.data_path, preload=True)
                return self.raw
            else:
                raise FileNotFoundError(f"File not found: {self.data_path}")
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def preprocess_data(self):
        try:
            if self.raw:
                self.raw.filter(1., 40., fir_design='firwin')
                return self.raw.get_data()
            else:
                print("No raw data to preprocess.")
                return None
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            return None

    def generate_synthetic_data(self, noise_level=0.01):
        try:
            if self.raw:
                data = self.raw.get_data()
                noise = np.random.randn(*data.shape) * noise_level
                synthetic_data = data + noise
                return synthetic_data
            else:
                print("No raw data to generate synthetic data from.")
                return None
        except Exception as e:
            print(f"Error generating synthetic data: {e}")
            return None

    def export_data(self, data, output_path):
        try:
            df = pd.DataFrame(data.T)
            df.to_csv(output_path, index=False)
            print(f"Data exported to {output_path}")
        except Exception as e:
            print(f"Error exporting data: {e}")

class Application(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("EEG Data Generator")
        self.geometry("800x600")

        self.data_handler = EEGDataHandler(r"D:\EEG\PythonApplication1\ds004196\sub-001\eeg\sub-001_task-rest_eeg.edf")
        self.create_widgets()

    def create_widgets(self):
        frame = ttk.Frame(self)
        frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Data Path:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.data_path_entry = ttk.Entry(frame, width=50)
        self.data_path_entry.grid(row=0, column=1, sticky=tk.W, pady=5)
        self.data_path_entry.insert(0, self.data_handler.data_path)

        load_button = ttk.Button(frame, text="Load Data", command=self.load_data)
        load_button.grid(row=1, column=0, columnspan=2, pady=5)

        generate_button = ttk.Button(frame, text="Generate Synthetic Data", command=self.generate_synthetic_data)
        generate_button.grid(row=2, column=0, columnspan=2, pady=5)

        export_button = ttk.Button(frame, text="Export Data", command=self.export_data)
        export_button.grid(row=3, column=0, columnspan=2, pady=5)

        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=4, column=0, columnspan=2, sticky=tk.NSEW)

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(4, weight=1)

    def load_data(self):
        self.data_handler.data_path = self.data_path_entry.get()
        self.raw = self.data_handler.load_data()
        if self.raw:
            messagebox.showinfo("Success", "Data loaded successfully.")
        else:
            messagebox.showerror("Error", "Failed to load data.")

    def generate_synthetic_data(self):
        synthetic_data = self.data_handler.generate_synthetic_data()
        if synthetic_data is not None:
            self.display_data(synthetic_data)
            messagebox.showinfo("Success", "Synthetic data generated.")
        else:
            messagebox.showerror("Error", "Failed to generate synthetic data.")

    def export_data(self):
        output_path = "D:/EEG/PythonApplication1/synthetic_data.csv"
        data = self.data_handler.generate_synthetic_data()
        if data is not None:
            self.data_handler.export_data(data, output_path)
            messagebox.showinfo("Success", f"Data exported to {output_path}.")
        else:
            messagebox.showerror("Error", "No data to export.")

    def display_data(self, data):
        self.ax.clear()
        for i, channel_data in enumerate(data):
            time = np.arange(channel_data.size) / self.raw.info['sfreq']
            self.ax.plot(time, channel_data + i * 10, label=f'Channel {i+1}')

        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Amplitude (uV)')
        self.ax.set_title('Synthetic EEG Data')
        self.ax.legend(loc='upper right', fontsize='small')
        self.ax.grid(True)
        self.canvas.draw()

def main():
    app = Application()
    app.mainloop()

if __name__ == "__main__":
    main()
