"""
Python Tkinter Splash Screen

This script holds the class SplashScreen, which is simply a window without
the top bar/borders of a normal window.

The window width/height can be a factor based on the total screen dimensions
or it can be actual dimensions in pixels. (Just edit the useFactor property)

Very simple to set up, just create an instance of SplashScreen, and use it as
the parent to other widgets inside it.

www.sunjay-varma.com
"""

from tkinter import *
from PIL import ImageTk, Image
import time
import os


class SplashScreen(Frame):
    def __init__(self, master=None, width=0.3, height=0.2, useFactor=True):
        Frame.__init__(self, master)
        self.pack(side=TOP, fill=BOTH, expand=YES)

        # get screen width and height
        ws = self.master.winfo_screenwidth()
        hs = self.master.winfo_screenheight()
        w = (useFactor and ws * width) or width
        h = (useFactor and ws * height) or height
        # calculate position x, y
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        self.master.geometry('%dx%d+%d+%d' % (w, h, x, y))

        self.master.overrideredirect(True)
        self.lift()

def splash():
    root = Tk()
    sp = SplashScreen(root)
    img = ImageTk.PhotoImage(Image.open("splash\\frame (1).png"))
    panel = Label(root, image=img)
    panel.pack(side="bottom", fill="both", expand="yes")
    frame_list = [f for f in os.listdir("splash\\") if os.path.isfile(os.path.join("splash\\", f))]
    # print(frame_list)
    i = 1
    start = time.perf_counter()

    while time.perf_counter()-start<3.2:
        img = ImageTk.PhotoImage(Image.open("splash\\"+frame_list[i]))
        panel.configure(image=img)
        root.update_idletasks()
        root.update()
        time.sleep(0.05)
        i = (i+1)%197
    # panel.destroy()
    img = ImageTk.PhotoImage(Image.open("splash\\frame (161).png"))
    panel.configure(image=img)
    root.update_idletasks()
    root.update()
    time.sleep(1.8)
    ## finished loading so destroy splash
    root.destroy()
    # Button(sp, text="Press this button to kill the program", bg='red', command=root.destroy).pack(side=BOTTOM, fill=X)
    root.mainloop()

if __name__=="__main__":
    splash()