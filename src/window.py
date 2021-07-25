from tkinter import filedialog
from tkinter import *
from tkinter.ttk import *

import cv2
from PIL import ImageTk, Image

from neural_networks.models import GruModel, LstmModel, ConvModel
from .processing import ImageProcessing
from .processing import WordsProcessing
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
        self.words_images = []
        self.initModels()
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
        ip = ImageProcessing(np.array(self.image_file))
        lines = ip.get_lines()
        for line in lines:
            words = ip.split_words_in_lines(line)
            for word in words:
                x_point_1 = word['x']
                y_point_1 = word['y'] + line['start']
                x_point_2 = word['x'] + word['w']
                y_point_2 = word['y'] + word['h'] + line['start']
                self.words_images.append(ip.gray[y_point_1: y_point_2, x_point_1: x_point_2])
                self.image_canvas.create_rectangle(x_point_1, y_point_1, x_point_2, y_point_2)

    def initModels(self):
        self.gru_model = GruModel()
        self.lstm_model = LstmModel()
        self.conv_model = ConvModel()

    def recognizeWords(self):
        words = []
        wp = WordsProcessing()
        for idx, word in enumerate(self.words_images):
            cv2.imwrite('ttt.png', word)
            processed_img = wp.preprocess(word).T
            words.append(processed_img)

        words = np.array(words)
        words = words.reshape(words.shape[0], 128, 64, 1)
        conv_model_predictions = self.conv_model.predictWord(words)
        lstm_model_predictions = self.lstm_model.predictWord(words)
        gru_model_predictions = self.gru_model.predictWord(words)

        # TODO add Levenshtein distance for recognized words

    def initUI(self):
        self.image_canvas.config(width=614, height=874)
        self.image_canvas.grid(row=0, column=0, sticky=NW, rowspan=100)

        self.open_file_button.config(command=self.selectFile, text='Import file', width=20)
        self.open_file_button.grid(row=10, column=1, padx=120, ipady=10)

        self.words_button.config(text='Highlight words', width=20, command=self.highlightWords)
        self.words_button.grid(row=11, column=1, padx=120, ipady=10)

        self.init_recognize_button.config(text='Recognize letters', width=20, command=self.recognizeWords)
        self.init_recognize_button.grid(row=12, column=1, padx=120, ipady=10)
