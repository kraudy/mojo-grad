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
      print(x_hot[0][i].data[], end=" ")

    var layer = Layer(27, 27, nonlin=False)
    """This is the same as a 27x27 matrix"""

    # Batch processing
    var batch_size = 1000
    var num_batches = (len(x_hot) + batch_size - 1) // batch_size  # Ceiling division
    print("Number of batches:", num_batches)

    for batch_idx in range(num_batches):
        var start = batch_idx * batch_size
        var end = min(start + batch_size, len(x_hot))
        if batch_idx % 10 == 0:
            print("Batch:", batch_idx, "Range:", start, "to", end)

        # Compute logits for this batch only
        var batch_logits = List[List[Value]]()
        for i in range(start, end):
            batch_logits.append(layer(x_hot[i]))

        var batch_probs = List[List[Value]]()
        for i in range(end - start):
            batch_probs.append(Value.soft_max(batch_logits[i]))
        print("  Batch probs size:", len(batch_probs))

        var batch_loss = List[Value]()
        var acum_loss = Value(0.0)
        var j = 0
        for i in range(start, end):
            batch_loss.append(batch_probs[j][ys[i]].log())
            acum_loss += batch_loss[j]
            j += 1
        print("  Batch loss size:", len(batch_loss))

        var loss = Value(0.0)
        #for i in range(end - start):
        #    loss += - (batch_loss[i] / acum_loss)
        for i in range(end - start):
            loss = loss - batch_loss[i]  # Sum negative log probs
        loss = loss / Value(Float64(end - start))  # Average over batch
        print("  Batch loss:", loss.data[])

        for par in layer.parameters():par[].grad[] = 0

        loss.backward()

        for p in layer.parameters(): p[].data[] -= 0.1 * p[].grad[]


fn main():   
    try:
        train_make_more()
    except e:
        print(e)