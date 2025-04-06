import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
from datetime import datetime
import threading
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from PIL import ImageGrab
import time
import win32gui
import win32con
import subprocess


class DesktopRecorder:
    def __init__(self):
        self.recording = False
        self.quality_settings = {
            'high': {'fps': 30, 'codec': 'mp4v'},
            'medium': {'fps': 24, 'codec': 'mp4v'},
            'low': {'fps': 15, 'codec': 'mp4v'}
        }

        # Create output directory
        self.output_dir = Path(os.path.dirname(os.path.abspath(__file__))) / 'recordings'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize main window
        self.root = tk.Tk()
        self.setup_window()

        # For storing initial window positions
        self.windows_info = {}

    def setup_window(self):
        """Setup the main control window"""
        self.root.title("Desktop 1 Recorder")
        self.root.attributes('-topmost', True)
        self.root.resizable(False, False)
        self.root.geometry('300x250')

        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        ttk.Label(
            main_frame,
            text="Desktop 1 Recorder",
            font=('Arial', 12, 'bold')
        ).grid(row=0, column=0, pady=5)

        # Status
        self.status_label = ttk.Label(main_frame, text="Ready to Record")
        self.status_label.grid(row=1, column=0, pady=5)

        # Record button
        self.record_button = ttk.Button(
            main_frame,
            text="Start Recording",
            command=self.toggle_recording
        )
        self.record_button.grid(row=2, column=0, pady=5)

        # Instructions
        instructions = ttk.Label(
            main_frame,
            text=(
                "Instructions:\n"
                "1. Arrange windows on Desktop 1\n"
                "2. Click 'Start Recording'\n"
                "3. Switch to Desktop 2 (Win+Ctrl+→)\n"
                "4. Recording continues on Desktop 1"
            ),
            justify=tk.LEFT,
            wraplength=280
        )
        instructions.grid(row=3, column=0, pady=10)

        # Close protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def store_windows_state(self):
        """Store the state of all windows on Desktop 1"""

        def enum_handler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title and not title.startswith("Desktop 1 Recorder"):
                    rect = win32gui.GetWindowRect(hwnd)
                    self.windows_info[hwnd] = {
                        'title': title,
                        'rect': rect,
                        'style': win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
                    }

        self.windows_info.clear()
        win32gui.EnumWindows(enum_handler, None)

    def toggle_recording(self):
        """Toggle recording state"""
        if not self.recording:
            # Store window states before starting recording
            self.store_windows_state()

            self.record_button.configure(text="Stop Recording")
            self.status_label.configure(text="Recording Desktop 1...")
            threading.Thread(target=self.record_screen, daemon=True).start()
        else:
            self.recording = False
            self.record_button.configure(text="Start Recording")
            self.status_label.configure(text="Ready to Record")

    def record_screen(self, quality='high', record_audio=False):
        """Record the screen"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_file = self.output_dir / f'desktop1_recording_{timestamp}.mp4'
        audio_file = self.output_dir / f'audio_recording_{timestamp}.wav' if record_audio else None

        # Get screen dimensions
        screen = ImageGrab.grab()
        screen_size = screen.size

        # Set quality parameters
        quality_params = self.quality_settings.get(quality, self.quality_settings['high'])
        fps = quality_params['fps']

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*quality_params['codec'])
        out = cv2.VideoWriter(str(video_file), fourcc, fps, screen_size)

        # Initialize audio recording if needed
        audio_frames = []
        if record_audio:
            audio_thread = threading.Thread(
                target=self._record_audio,
                args=(audio_frames, audio_file)
            )
            audio_thread.start()

        self.recording = True
        print(f"Recording started... Output will be saved to: {video_file}")

        try:
            last_frame_time = time.time()
            frame_interval = 1.0 / fps

            while self.recording and self.root.winfo_exists():
                current_time = time.time()

                if current_time - last_frame_time >= frame_interval:
                    # Capture screen
                    screen = ImageGrab.grab()
                    frame = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)
                    out.write(frame)
                    last_frame_time = current_time

                # Prevent high CPU usage
                time.sleep(0.001)

        except Exception as e:
            print(f"Error during recording: {e}")
            if self.root.winfo_exists():
                self.status_label.configure(text="Error occurred!")

        finally:
            # Cleanup
            self.recording = False
            out.release()
            cv2.destroyAllWindows()

            if record_audio:
                audio_thread.join()
                print("Audio recording completed.")

            print("Recording finished!")
            print(f"Video saved to: {video_file}")
            if record_audio and audio_file and audio_file.exists():
                print(f"Audio saved to: {audio_file}")

    def _record_audio(self, audio_frames, audio_file):
        """Record audio to a separate WAV file"""
        sample_rate = 44100
        try:
            with sd.InputStream(samplerate=sample_rate, channels=2) as stream:
                while self.recording:
                    audio_chunk, overflowed = stream.read(1024)
                    audio_frames.append(audio_chunk)

            # Save audio when recording stops
            if audio_frames:
                audio_data = np.concatenate(audio_frames, axis=0)
                sf.write(str(audio_file), audio_data, sample_rate)

        except Exception as e:
            print(f"Error recording audio: {e}")

    def on_closing(self):
        """Handle window closing"""
        self.recording = False
        self.root.destroy()

    def run(self):
        """Start the application"""
        try:
            self.root.mainloop()
        except Exception as e:
            print(f"Error in main loop: {e}")


if __name__ == "__main__":
    try:
        app = DesktopRecorder()
        app.run()
    except Exception as e:
        print(f"An error occurred: {e}")