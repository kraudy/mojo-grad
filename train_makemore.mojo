from mojograd.engine import Value 
from mojograd.nn import Neuron, MLP, Layer
from pathlib import Path

fn read_words() raises -> List[String]:
  var words = Path("./datasets/names.txt").read_text().split("\n")

  # Print the words to verify
  for word in words:
      print(word[])

  return words


fn train_make_more() raises:
    # initialize a model 
    var words = read_words()

fn main():   
    try:
        train_make_more()
    except e:
        print(e)