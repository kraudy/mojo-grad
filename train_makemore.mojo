from mojograd.engine import Value 
from mojograd.nn import Neuron, MLP, Layer
from pathlib import Path
from collections import Set

fn read_words() raises -> List[String]:
  var words = Path("./datasets/names.txt").read_text().split("\n")

  var char_set = Set[String]()
  for word in words:
      for char in word[]:
          char_set.add(String(char))

  var chars = List[String]()
  for char in char_set:
      chars.append(char[])

  sort(chars)

  return chars


fn train_make_more() raises:
    # initialize a model 
    var chars = read_words()
    print("Sorted unique characters:")
    for char in chars:
        print(char[], end=" ")

fn main():   
    try:
        train_make_more()
    except e:
        print(e)