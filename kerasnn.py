import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense
import numpy as np
from numpy import loadtxt
import pandas as pd
import sys
from threading import *
from tkinter import *
from tkinter.filedialog import asksaveasfile
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import ttk
from tkinter.filedialog import *

class KerasNN(Tk):
    def __init__(self):
        super().__init__()

        self.filename = 'model.json'
        self.dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
        self.datasetName = 'pima-indians-diabetes.csv'
        self.X = self.dataset[:,0:8]
        self.y = self.dataset[:,8]
        self.model = None

        # Create the window
        self.wm_title('Keras Neural Network')
        self.geometry("1280x768")
        self.protocol("WM_DELETE_WINDOW", self.exit) # Call the exit function when clicking the red X button
        
        # Create the menubar
        self.menubar = Menu(self)
        self.config(menu=self.menubar)

        # Create the file menu
        self.fileMenu = Menu(self.menubar, tearoff=False)
        self.fileMenu.add_command(label='New Model', command=self.newModel, underline=0)
        self.fileMenu.add_command(label='Open Model', command=self.openModel, underline=0)
        self.fileMenu.add_command(label='Close Model', command=self.closeModel, underline=0)
        self.fileMenu.add_separator()
        self.fileMenu.add_command(label='Save Model', command=lambda: self.saveModel(self.filename), underline=0)
        self.fileMenu.add_command(label='Save Model As', command=self.saveModelAs, underline=0)
        self.fileMenu.add_separator()
        #self.fileMenu.add_command(label='Load Dataset', command=self.loadDataset, underline=0)
        #self.fileMenu.add_separator()
        self.fileMenu.add_command(label='Exit', command=self.exit, underline=1)

        # Create the view menu
        self.viewMenu = Menu(self.menubar, tearoff=False)
        self.viewMenu.add_command(label='Dataset', command=self.viewDataset, underline=0)

        # Create the run menu
        self.runMenu = Menu(self.menubar, tearoff=False)
        self.runMenu.add_command(label='Train', command=self.train, underline=0)
        self.runMenu.add_command(label='Predict', command=self.predict, underline=0)

        # Add menus to menubar
        self.menubar.add_cascade(label='File', menu=self.fileMenu, underline=0)
        self.menubar.add_cascade(label='View', menu=self.viewMenu, underline=0)
        self.menubar.add_cascade(label='Run', menu=self.runMenu, underline=0)

        # Create label with dataset name
        self.datasetLbl = Label(self, text='Dataset: ' + self.datasetName)
        self.datasetLbl.pack()

        # Create vertical scrollbar
        self.vScrollbar = Scrollbar(self, orient='vertical')
        self.vScrollbar.pack(side=RIGHT, fill='y')

        # Create horizontal scrollbar
        self.hScrollbar = Scrollbar(self, orient='horizontal')
        self.hScrollbar.pack(side=BOTTOM, fill='x')

        # Create textbox
        self.textbox = Text(self, width=1280, height=710, wrap=NONE, xscrollcommand=self.hScrollbar.set, yscrollcommand=self.vScrollbar.set)

        # Attach scrollbars to textbox
        self.vScrollbar.config(command=self.textbox.yview)
        self.hScrollbar.config(command=self.textbox.xview)
        self.textbox.pack()

        # Create the model
        self.newModel()

    """
    This closes the currently open model file.
    self: object reference
    """
    def closeModel(self):
        # Display model closed message
        self.textbox.insert(END, "Model file " + self.filename + " closed.\n")

        self.model = None
        self.filename = ''

        # Update window title
        self.wm_title("Keras Neural Network")

        # Disable menu items
        self.fileMenu.entryconfig("Close Model", state="disabled")
        self.fileMenu.entryconfig("Save Model", state="disabled")
        self.fileMenu.entryconfig("Save Model As", state="disabled")
        self.runMenu.entryconfig("Train", state="disabled")
        self.runMenu.entryconfig("Predict", state="disabled")

    """
    This handles exiting the program.
    self: object reference
    """
    def exit(self):
        self.destroy()
        sys.exit(0)

    """
    This loads the specified dataset file for the neural network to process.
    self: object reference
    """
    def loadDataset(self):
        pass
        """
        dataFile = askopenfile(mode='r', filetypes=[('CSV File', '.csv'), ('XLSX File', '.xlsx')])

        if dataFile is not None:
            if dataFile[-4:] == '.csv':
                self.dataset = loadtxt(fname=dataFile.name, delimiter=',')
                self.datasetName = dataFile[-4:]
            else:
                self.dataset = np.array(pd.read_excel(dataFile.name))
                self.datasetName = dataFile[-5:]
                """

    """
    This creates a new, untrained Keras neural net model.
    self: object reference
    """
    def newModel(self):
        # Define the keras model
        self.model = Sequential()
        self.model.add(Dense(12, input_shape=(8,), activation='relu'))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        # Compile the keras model
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['Precision', 'TrueNegatives', 'TruePositives', 'FalseNegatives', 'FalsePositives', 'Accuracy'])

        # Update window title
        self.wm_title("Keras Neural Network: " + self.filename)

        # Display model summary
        self.textbox.insert(END, 'Model File: ' + self.filename + '\n')
        self.model.summary(print_fn=lambda x: self.textbox.insert(END, x + '\n'))
        self.textbox.yview_moveto(1)

        # Enable menu items
        self.fileMenu.entryconfig("Close Model", state="normal")
        self.fileMenu.entryconfig("Save Model", state="normal")
        self.fileMenu.entryconfig("Save Model As", state="normal")
        self.runMenu.entryconfig("Train", state="normal")
        self.runMenu.entryconfig("Predict", state="normal")

    """
    This opens the selected model file.
    self: object reference
    """
    def openModel(self):
        # Open selected file
        modelFile = askopenfile(mode='r', filetypes=[('JSON File', '.json')])

        if modelFile is not None:
            self.filename = modelFile.name
            self.filename = self.filename[(self.filename.rfind('/') + 1):]

            # Load json and create model
            self.model = model_from_json(modelFile.read())
            modelFile.close()
            self.model.load_weights(self.filename[:-5] + '.h5')

            # Compile the keras model
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            # Update window title
            self.wm_title("Keras Neural Network: " + self.filename)

            # Display model summary
            self.textbox.insert(END, 'Model File: ' + self.filename + '\n')
            self.model.summary(print_fn=lambda x: self.textbox.insert(END, x + '\n'))
            self.textbox.yview_moveto(1)

            # Enable menu items
            self.fileMenu.entryconfig("Close Model", state="normal")
            self.fileMenu.entryconfig("Save Model", state="normal")
            self.fileMenu.entryconfig("Save Model As", state="normal")
            self.runMenu.entryconfig("Train", state="normal")
            self.runMenu.entryconfig("Predict", state="normal")

    """
    This has the neural network predict the output of the dataset
    self: object reference
    """
    def predict(self):
        if self.model is not None:
            t = Thread(target=self.predictModel)
            t.start()
            t.join()

    """
    This is the thread function for predicting with the model.
    self: object reference
    """
    def predictModel(self):
        # Make probability predictions with the model
        (self.model.predict(x=self.X, verbose=2, callbacks=[ModelCallback(self.textbox)]) > 0.5).astype(int)

    """
    This saves the current model with the specified filename.
    self: object reference
    filename: name of the file
    """
    def saveModel(self, filename='model.json'):
        if self.model is not None:
            # Save as JSON file
            self.filename = filename

            if self.filename[-5:] != '.json':
                self.filename += '.json'

            # Save model to JSON
            modelFile = self.model.to_json()
            with open(self.filename, "w") as jsonFile:
                jsonFile.write(modelFile)
                jsonFile.close()

            # Save weights to h5 file
            self.model.save_weights(filename[:-5] + '.h5')

            # Display save success message
            self.textbox.insert(END, "Model saved successfully.\n")

    """
    This use the Save As file dialog to save the current model with the specified filename.
    self: object reference
    """
    def saveModelAs(self):
        if self.model is not None:
            modelFile = asksaveasfile(filetypes=[("json file", ".json")], defaultextension=".json")

            self.filename = modelFile.name
            self.filename = self.filename[(self.filename.rfind('/') + 1):]

            modelFile.close()

            self.saveModel(self.filename)

            # Update window title
            self.wm_title("Keras Neural Network: " + self.filename)

            # Display updated filename
            self.textbox.insert(END, "Model File Saved As: " + self.filename + "\n")

    """
    This trains the neural network on the current dataset.
    self: object reference
    """
    def train(self):
        if self.model is not None:
            t = Thread(target=self.trainModel)
            t.start()
            t.join()

    """
    This is the thread function for training the model.
    self: object reference
    """
    def trainModel(self):
        # Fit the keras model on the dataset
        self.model.fit(self.X, self.y, epochs=150, batch_size=10, verbose=0, callbacks=[ModelCallback(self.textbox)])

        # Evaluate the keras model
        self.model.evaluate(self.X, self.y, verbose=0, callbacks=[ModelCallback(self.textbox)])

    """
    This displays a popup window containing a table view of the current dataset.
    self: object reference
    """
    def viewDataset(self):
        # If a dataset is present
        if self.dataset is not None:
            # Create a popup displaying a table containing the dataset
            popup = Tk()
            popup.geometry("800x400")
            popup.wm_title('Dataset: ' + self.datasetName)

            # Create vertical scrollbar
            vScrollbar = Scrollbar(popup, orient='vertical')
            vScrollbar.pack(side=RIGHT, fill='y')

            # Create horizontal scrollbar
            hScrollbar = Scrollbar(popup, orient='horizontal')
            hScrollbar.pack(side=BOTTOM, fill='x')

            # Create textbox
            textbox = Text(popup, width=400, height=800, xscrollcommand=hScrollbar.set, yscrollcommand=vScrollbar.set)

            for i in range(len(self.dataset)):
                for j in range(len(self.dataset[i])):
                    textbox.insert(END, "%9s " % str(self.dataset[i][j]))

                textbox.insert(END, "\n")


            # Attach scrollbars to textbox
            vScrollbar.config(command=textbox.yview)
            hScrollbar.config(command=textbox.xview)
            textbox.pack()
            
            popup.mainloop()

"""
ProgressDialog: This class is used to create a progressbar dialog box that closes only after the progressbar completes.
"""
class ProgressDialog(simpledialog.SimpleDialog):
    """
    Initializes a simple dialog that can have a title and contain text along with the progressbar that is positioned at the bottom of the dialog.
    self: object reference
    master: parent of this dialog
    test: optional message to display in the dialog
    title: optional title of this dialog
    """
    def __init__(self, master, text='', title=None):
        super().__init__(master=master, text=text, title=title)

        self.default = None
        self.cancel = None

        self.pbar = ttk.Progressbar(self.root, orient="horizontal", length=200, mode="indeterminate")
        self.pbar.pack(expand=True, fill=X, side=BOTTOM)

        self.root.attributes("-topmost", True) # Keep the dialog on top

"""
ModelCallback: This class is used to display training/testing data in the textbox.
"""
class ModelCallback(keras.callbacks.Callback):
    def __init__(self, textbox):
        self.textbox = textbox

    def on_train_begin(self, logs=None):
        if self.textbox is not None:
            self.textbox.insert(END, "Model Training Report ************************\n")
            self.textbox.yview_moveto(1)

    def on_train_end(self, logs=None):
        if self.textbox is not None:
            self.textbox.insert(END, "Model Training Completed\n")
            self.textbox.yview_moveto(1)

    def on_epoch_begin(self, epoch, logs=None):
        if self.textbox is not None:
            self.textbox.insert(END, "Epoch " + str(epoch + 1) + "/150\n")
            self.textbox.yview_moveto(1)

    def on_epoch_end(self, epoch, logs=None):
        if self.textbox is not None:
            for x in logs.keys():
                self.textbox.insert(END, " - %s: %.2f" % (x, logs[x]))
            self.textbox.insert(END, "\n")
            self.textbox.yview_moveto(1)

    def on_test_begin(self, logs=None):
        if self.textbox is not None:
            self.textbox.insert(END, "Model Testing Report ************************\n")
            self.textbox.yview_moveto(1)

    def on_test_end(self, logs=None):
        if self.textbox is not None:
            self.textbox.insert(END, "Model Testing Completed ************************\n")
            self.textbox.yview_moveto(1)

    def on_predict_begin(self, logs=None):
        if self.textbox is not None:
            self.textbox.insert(END, "Model Prediction Report ************************\n")
            self.textbox.yview_moveto(1)

    def on_predict_end(self, logs=None):
        if self.textbox is not None:
            self.textbox.insert(END, "Model Prediction Completed ************************\n")
            self.textbox.yview_moveto(1)

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        if self.textbox is not None:
            for x in logs.keys():
                self.textbox.insert(END, " - %s: %.2f" % (x, logs[x]))
            self.textbox.insert(END, "\n")
            self.textbox.yview_moveto(1)

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        if self.textbox is not None:
            self.textbox.insert(END, "%s: " % ('Predicted Outputs'))
            for x in logs['outputs']:
                self.textbox.insert(END, "[%.2f] " % (x))
            self.textbox.insert(END, "\n")
            self.textbox.yview_moveto(1)


"""
This starts the program. It initializes the KerasNN GUI window.
"""
if __name__ == "__main__":
    kerasnn = KerasNN()
    kerasnn.mainloop()