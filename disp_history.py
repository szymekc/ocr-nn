import pickle as pkl
import pprint


with open("history_rect_64.pkl", "rb") as f:
    history = pkl.load(f)

with open("conv.txt", "wt") as f:
    string = ""
    for loss in [history["loss"], history["val_loss"]]:
        i = 1
        for el in loss:
            string += "(" + str(i) + "," + str(round(el, 2)) + ")"
            i += 1
        string += '\n'
    f.write(string)
