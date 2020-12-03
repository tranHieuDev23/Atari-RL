import os
from os import path, getenv, getcwd
import shutil
from statistics import mean
import csv
from matplotlib import pyplot as plt
from dotenv import load_dotenv

load_dotenv(path.join(getcwd(), '.env'))

TRAINING_UPDATE_FREQUENCY = int(getenv('TRAINING_UPDATE_FREQUENCY'))
RUN_UPDATE_FREQUENCY = int(getenv('RUN_UPDATE_FREQUENCY'))


class Logger:
    def __init__(self, header, directory_path):
        self.__score = Stat(
            'Run', 'Score', RUN_UPDATE_FREQUENCY, directory_path, header)
        self.__step = Stat(
            'Run', 'Step', RUN_UPDATE_FREQUENCY, directory_path, header)
        self.__loss = Stat(
            'Update', 'Loss', TRAINING_UPDATE_FREQUENCY, directory_path, header)
        self.__q = Stat(
            'Update', 'Q', TRAINING_UPDATE_FREQUENCY, directory_path, header)
        if (os.path.exists(directory_path)):
            shutil.rmtree(directory_path, ignore_errors=True)
        os.makedirs(directory_path)

    def add_score(self, value):
        self.__score.add_entry(value)

    def add_step(self, value):
        self.__step.add_entry(value)

    def add_loss(self, value):
        self.__loss.add_entry(value)

    def add_q(self, value):
        self.__q.add_entry(value)


class Stat:
    def __init__(self, x_label, y_label, update_freq, directory_path, header):
        self.__x_label = x_label
        self.__y_label = y_label
        self.__update_freq = update_freq
        self.__header = header
        self.__values = []

        self.__csv_filepath = path.join(
            directory_path, '%s - %s.csv' % (header, y_label))
        self.__png_filepath = path.join(
            directory_path, '%s - %s.png' % (header, y_label))

    def add_entry(self, value):
        self.__values.append(value)
        if (len(self.__values) % self.__update_freq == 0):
            avg = mean(self.__values)
            self.__dump_csv(avg)
            self.__draw_png(self.__update_freq)

    def __dump_csv(self, value):
        if (not os.path.exists(self.__csv_filepath)):
            with open(self.__csv_filepath, 'w'):
                pass
        with open(self.__csv_filepath, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([value])

    def __draw_png(self, x_distance):
        x = []
        y = []
        with open(self.__csv_filepath, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            for i in range(0, len(data)):
                x.append(float(i) * x_distance)
                y.append(float(data[i][0]))
        plt.plot(x, y, label="Last %d average" % self.__update_freq)
        plt.title(self.__header)
        plt.xlabel(self.__x_label)
        plt.ylabel(self.__y_label)
        plt.legend(loc="upper left")
        plt.savefig(self.__png_filepath, bbox_inches="tight")
        plt.close()
