import os
import numpy as np
import mne
import logging
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ctgan import CTGAN

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class EEGDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.raw_data = []

    def load_data(self):
        try:
            data_files = []
            for root, dirs, files in os.walk(self.data_path):
                for file in files:
                    if file.endswith('.edf') or file.endswith('.bdf'):
                        data_files.append(os.path.join(root, file))
            
            if not data_files:
                raise FileNotFoundError("No EDF or BDF files found in the data path.")

            self.raw_data = []
            for file in data_files:
                raw = mne.io.read_raw_edf(file, preload=True) if file.endswith('.edf') else mne.io.read_raw_bdf(file, preload=True)
                self.raw_data.append(raw.get_data())

            logger.info("Data loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

class EEGSyntheticDataGenerator:
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.synthesizer = CTGAN()

    def generate_synthetic_data(self, num_samples):
        try:
            # Flatten the list of arrays and transpose to match (samples, features) shape
            data = np.hstack(self.raw_data).T
            self.synthesizer.fit(data)
            synthetic_data = self.synthesizer.sample(num_samples)
            logger.info("Synthetic data generated successfully.")
            return synthetic_data
        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")
            return None

class Application(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("EEG Synthetic Data Generator")
        self.geometry("800x600")

        self.data_loader = EEGDataLoader(r"D:\EEG\PythonApplication1\EEGData")
        self.data_loader.load_data()

        self.synthetic_data_generator = EEGSyntheticDataGenerator(self.data_loader.raw_data)
        self.create_widgets()

    def create_widgets(self):
        frame = ttk.Frame(self)
        frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Number of Samples:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.num_samples_entry = ttk.Entry(frame)
        self.num_samples_entry.grid(row=0, column=1, sticky=tk.W, pady=5)
        self.num_samples_entry.insert(0, "100")

        generate_button = ttk.Button(frame, text="Generate Data", command=self.generate_and_display_data)
        generate_button.grid(row=1, column=0, columnspan=2, pady=10)

        self.fig = Figure(figsize=(10, 6))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=2, column=0, columnspan=2, sticky=tk.NSEW)

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(2, weight=1)

    def generate_and_display_data(self):
        try:
            num_samples = int(self.num_samples_entry.get())
            synthetic_data = self.synthetic_data_generator.generate_synthetic_data(num_samples)

            if synthetic_data is not None:
                self.ax.clear()
                for i, sample in enumerate(synthetic_data):
                    time = np.arange(len(sample))
                    self.ax.plot(time, sample, label=f'Sample {i+1}')

                self.ax.set_xlabel('Time')
                self.ax.set_ylabel('Amplitude')
                self.ax.set_title('Synthetic EEG Data')
                self.ax.legend(loc='upper right')
                self.ax.grid(True)
                self.canvas.draw()

                logger.info("Displayed synthetic data.")
            else:
                messagebox.showerror("Error", "Failed to generate synthetic data.")
        except Exception as e:
            logger.error(f"Error in generating data: {e}")
            messagebox.showerror("Error", "An error occurred while generating data.")

def main():
    app = Application()
    app.mainloop()

if __name__ == "__main__":
    main()
