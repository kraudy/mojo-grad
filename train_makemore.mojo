from mojograd.engine import Value 
from mojograd.nn import Neuron, MLP, Layer

fn read_words() raises:
  with open("./datasets/names.txt", "r") as file:
      var content = file.read()
      print(content)


fn train_make_more() raises:
    # initialize a model 
    read_words()

fn main():   
    try:
        train_make_more()
    except e:
        print(e)