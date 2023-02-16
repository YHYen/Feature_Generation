from tkinter import *

# frame setting
frame = Tk()
frame.title('Word Embedding')
frame.geometry('500x500')
frame.resizable(True, True)


# unit setting
label = Label(frame, text='當前文本', bg='#d3fbfb', fg='black', font=('華文新魏', 18), width=20, height=2, relief=SUNKEN)
label.pack()
frame.mainloop()