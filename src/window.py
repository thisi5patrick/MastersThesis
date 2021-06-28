from tkinter import filedialog
from tkinter import *
from tkinter.ttk import *
from PIL import ImageTk, Image
from .processing import LettersProcessing
import numpy as np


class Window(Frame):
    def __init__(self):
        super().__init__()
        self.master.title('Praca Magisterska')
        self.grid()
        self.master.geometry('1000x874')
        self.open_file_button = Button(self)
        self.image_canvas = Canvas(self)
        self.words_button = Button(self)
        self.init_recognize_button = Button(self)
        self.initUI()

    def selectFile(self):
        file_types = (
            ('Image files', '*.png'),
            ('All files', '*.*')
        )
        file_name = filedialog.askopenfilename(
            title='Select file',
            filetypes=file_types
        )
        self.image_file = Image.open(file_name)
        if self.image_file.size[0] != 614:
            self.image_file = self.image_file.resize(
                (614, int(self.image_file.size[1] * 614 / self.image_file.size[0])))

        self.image = ImageTk.PhotoImage(self.image_file)
        self.image_canvas.create_image((0, 0), anchor=NW, image=self.image)

    def highlightWords(self):
        lp = LettersProcessing(np.array(self.image_file))
        lines = lp.get_lines()
        for line in lines:
            words = lp.split_words_in_lines(line)
            for word in words:
                x_point_1 = word[0]['x']
                y_point_1 = word[0]['y'] + line['start']
                x_point_2 = word[0]['x'] + word[0]['w']
                y_point_2 = word[0]['y'] + word[0]['h'] + line['start']
                self.image_canvas.create_rectangle(x_point_1, y_point_1, x_point_2, y_point_2)

    def recognizeWords(self):
        ...

    def initUI(self):
        self.image_canvas.config(width=614, height=874)
        self.image_canvas.grid(row=0, column=0, sticky=NW, rowspan=100)

        self.open_file_button.config(command=self.selectFile, text='Import file', width=20)
        self.open_file_button.grid(row=10, column=1, padx=120, ipady=10)

        self.words_button.config(text='Highlight words', width=20, command=self.highlightWords)
        self.words_button.grid(row=11, column=1, padx=120, ipady=10)

        self.init_recognize_button.config(text='Recognize letters', width=20, command=self.recognizeWords)
        self.init_recognize_button.grid(row=12, column=1, padx=120, ipady=10)
