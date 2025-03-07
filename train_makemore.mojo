from mojograd.engine import Value 
from mojograd.nn import Neuron, MLP, Layer
from pathlib import Path
from collections import Set, Dict

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

fn char_to_int(chars: List[String]) -> Dict[String, Int]:
    var stoi = Dict[String, Int]()
    for i in range(len(chars)):
        stoi[chars[i]] = i + 1
    return stoi

fn train_make_more() raises:
    var chars = read_words()
    var stoi = char_to_int(chars)
    for char in chars:
        print(char[]," ", stoi[char[]], end=" ")

fn main():   
    try:
        train_make_more()
    except e:
        print(e)