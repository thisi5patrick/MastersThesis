from tkinter import filedialog
from tkinter import *
from tkinter.ttk import *

import cv2
from PIL import ImageTk, Image

from neural_networks.models import GruModel, LstmModel, ConvModel
from .processing import ImageProcessing
from .processing import WordsProcessing
import numpy as np
from .helpers import levenshtein


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
        self.image_text_input_label = Label(self)
        self.image_text_input = Text(self)
        self.highlight_word_button_pressed = False
        self.conv_distance_label = Label(self)
        self.lstm_distance_label = Label(self)
        self.gru_distance_label = Label(self)
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

        self.highlight_word_button_pressed = True

    def initModels(self):
        self.gru_model = GruModel()
        self.lstm_model = LstmModel()
        self.conv_model = ConvModel()

    def recognizeWords(self):
        if not self.highlight_word_button_pressed:
            self.highlightWords()

        words = []
        wp = WordsProcessing()
        for idx, word in enumerate(self.words_images):
            processed_img = wp.preprocess(word)
            words.append(processed_img)

        words = np.array(words)
        words = words.reshape(words.shape[0], 128, 64, 1)
        conv_model_predictions = ' '.join(word for word in self.conv_model.predictWord(words))
        lstm_model_predictions = ' '.join(word for word in self.lstm_model.predictWord(words))
        gru_model_predictions = ' '.join(word for word in self.gru_model.predictWord(words))

        input_text = self.image_text_input.get(1.0, END).replace('\n', ' ')

        conv_distance = levenshtein(input_text, conv_model_predictions)
        lstm_distance = levenshtein(input_text, lstm_model_predictions)
        gru_distance = levenshtein(input_text, gru_model_predictions)

        self.showResults(conv_distance=conv_distance, lstm_distance=lstm_distance, gru_distance=gru_distance)

        self.highlight_word_button_pressed = False

    def showResults(self, **kwargs):
        conv_distance = kwargs.get('conv_distance')
        lstm_distance = kwargs.get('lstm_distance')
        gru_distance = kwargs.get('gru_distance')

        self.conv_distance_label.config(text=f'Convolution model distance: {conv_distance}')
        self.lstm_distance_label.config(text=f'LSTM model distance: {lstm_distance}')
        self.gru_distance_label.config(text=f'Recurrent model distance: {gru_distance}')

    def initUI(self):
        self.image_canvas.config(width=614, height=874)
        self.image_canvas.grid(row=0, column=0, sticky=NW, rowspan=100)

        self.open_file_button.config(command=self.selectFile, text='Import file', width=20)
        self.open_file_button.grid(row=10, column=1, padx=120, ipady=10)

        self.image_text_input_label.config(text='Write text from image')
        self.image_text_input_label.grid(row=11, column=1, pady=(10, 0))

        self.image_text_input.config(width=40, height=4, font='Calibri')
        self.image_text_input.grid(row=12, column=1, pady=(0, 10))

        self.words_button.config(text='Highlight words', width=20, command=self.highlightWords)
        self.words_button.grid(row=13, column=1, padx=120, ipady=10)

        self.init_recognize_button.config(text='Recognize letters', width=20, command=self.recognizeWords)
        self.init_recognize_button.grid(row=14, column=1, padx=120, ipady=10)

        self.conv_distance_label.config(text='')
        self.conv_distance_label.grid(row=15, column=1, pady=(10, 0))

        self.lstm_distance_label.config(text='')
        self.lstm_distance_label.grid(row=16, column=1)

        self.gru_distance_label.config(text='')
        self.gru_distance_label.grid(row=17, column=1)
