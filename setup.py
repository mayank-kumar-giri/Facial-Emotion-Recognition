import cx_Freeze
import sys


base = None
if sys.platform == 'win32':
    base = "Win32GUI"

executables = [cx_Freeze.Executable("main.py", base=base, icon = "icon.ico")]

cx_Freeze.setup(
    name = "Human Emotion Recognizer",
    author = "Mayank Kumar Giri",
    options = {"build_exe":{"packages":["tkinter", "PIL", "playsound", "os", "cv2", "keras","time", "numpy", "decimal"], "include_files":['Angry.mp3', 'ddown.jpg', 'Disgusted.mp3', 'Fearful.mp3', 'folder_selection.jpg', 'Happy.mp3', 'icon.ico', 'main.ico', 'Neutral.mp3', 'realtime.jpg', 'Sad.mp3', 'save.ico', 'splash.py', 'Surprised.mp3']}},
    version = "1.0",
    description = "A Windows app that recognizes human emotion from images and from real time camera feed",
    executables = executables
)