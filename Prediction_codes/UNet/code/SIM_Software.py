from tkinter import *
import tkinter.messagebox
from tkinter import filedialog
from tkinter import ttk
import os

root = Tk()
root.title("SIM RECONSTRUCTION SOFTWARE")
root.geometry('1000x500')
root.iconbitmap(default='cirl.ico')
title_bar = Frame(root, bg='white', relief='raised', bd=2)
root.config(background = "blue")



w = Label(root, text="Developed by Bereket Kebede @CIRL UofM\n", bg = "blue", fg = "white")
s = Label(root, text="Deep Learning Enabled SIM\n", font=("Arial Bold", 10), bg = "blue", fg = "white")
w.pack()
s.pack()

L2 = Label(root, text="Choose 3 SIM Images \n", bg = "blue", fg = "white")
L2.pack( side = LEFT)

def open(self):
    os.system("start C:/folder dir/")
    button1= Button(self, text="Pre TC", fg="red", font=("Ariel", 9, "bold"), command=self.open)

def openFolder():
    path='C:'
    command = 'explorer.exe ' + path
    os.system(command)


def browseFiles():
    filename = filedialog.askopenfilename(initialdir="/",
                                          title="Select a File",
                                          filetypes=(("Text files",
                                                      "*.txt*"),
                                                     ("all files",
                                                      "*.*")))

    # Change label contents
    label_file_explorer.configure(text="File Opened: " + filename)

#########################################################

label_file_explorer = Label(root,
                            text="Ready to Use",
                            width=100, height=4,
                            fg="blue")

button_explore = Button(root,
                        text="Upload Images",
                        command=browseFiles)

button_exit = Button(root,
                     text="Exit",
                     command=exit)

# Grid method is chosen for placing
# the widgets at respective positions
# in a table like structure by
# specifying rows and columns

#label_file_explorer.pack(side = RIGHT)



#button_exit.pack(side = LEFT)

#########################################################

B = tkinter.Button(root, text = "Reconstruct")
C = tkinter.Button(root, text = "Reset")

#btn = Button(root, text = 'Upload Images',command = openFolder)
#btn.pack(side = 'top')

C.pack(side = BOTTOM)
B.pack(side = RIGHT)
button_explore.pack(side = RIGHT)

root.mainloop()