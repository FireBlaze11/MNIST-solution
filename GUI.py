from tkinter import *                                                                  

mainWindow = Tk()
bg = "black"
fg = "white"
brush_size = 5
mainWindow.title("MNIST Solver")
mainWindow.config(background=bg)
mainWindow.geometry("600x500")
icon = PhotoImage(file="MNIST-Solver.png")
mainWindow.iconphoto(False, icon)
