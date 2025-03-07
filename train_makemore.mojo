from mojograd.engine import Value 
from mojograd.nn import Neuron, MLP, Layer
from pathlib import Path
from collections import Set, Dict

fn read_words() raises -> List[String]:
  return Path("./datasets/names.txt").read_text().split("\n")

fn words_to_chars(words: List[String]) raises -> List[String]:
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
    stoi['.'] = 0
    return stoi

fn make_inputs(words: List[String], stoi: Dict[String, Int]) raises -> Tuple[List[Int], List[Int]]:
    var xs = List[Int]()
    var ys = List[Int]()

    for w in words:
      chs = List[String]('.') 
      for ch in w[]:
        chs.append(String(ch))
      chs.append('.') 
      for i in range(len(chs) - 1):
        var ix1 = stoi[chs[i]]
        var ix2 = stoi[chs[i + 1]]
        xs.append(ix1)
        ys.append(ix2)
    
    return (xs, ys)

fn train_make_more() raises:
    var words = read_words()

    var xs = List[Int]()
    var ys = List[Int]()
    (xs, ys) = make_inputs(words, char_to_int(words_to_chars(words)))
    print("xs: ", len(xs))
    print("ys: ", len(ys))


    var x_hot = Value.one_hot(xs, 27) 
    var y_hot = Value.one_hot(ys, 27) 
    print("x_hot: ", len(x_hot))
    print("y_hot: ", len(y_hot))
    for i in range(len(x_hot[0])):
      print(x_hot[0][i].data[])

    var layer = Layer(27, 27, nonlin=False)
    """This is the same as a 27x27 matrix"""

fn main():   
    try:
        train_make_more()
    except e:
        print(e)