## 首先，我们引入一些库

```python
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve
```

### 读入文本

读文件稍微有些不一样，不是处理成list，而是直接读成一个字符串，因为后面用到的就是串数据。

```python
filename = 'text8.zip'

def read_data(filename):
  f = zipfile.ZipFile(filename)
  for name in f.namelist():
    return tf.compat.as_str(f.read(name))
  f.close()

text = read_data(filename)
print('Data size %d' % len(text))

```



### 生成训练数据集函数

切割一下，留1000个字符做检验，其他99999000个字符拿来训练。

```python
valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print(train_size, train_text[:64])
print(valid_size, valid_text[:64])
```

### 两个工具函数

建立两个函数`char2id`和`id2char`，用来把字符对应成数字。

本程序只考虑26个字母外加1个空格字符，其他字符都当做空格来对待。所以可以用两个函数，通过ascii码加减，直接算出对应的数值或字符。

```python
vocabulary_size = len(string.ascii_lowercase) + 1 # [a-z] + ' '
first_letter = ord(string.ascii_lowercase[0])

def char2id(char):
  if char in string.ascii_lowercase:
    return ord(char) - first_letter + 1
  elif char == ' ':
    return 0
  else:
    print('Unexpected character: %s' % char)
    return 0

def id2char(dictid):
  if dictid > 0:
    return chr(dictid + first_letter - 1)
  else:
    return ' '

print(char2id('a'), char2id('z'), char2id(' '), char2id('ï'))
print(id2char(1), id2char(26), id2char(0))
```

### 生成训练数据集函数

这次 `BatchGenerator` 做的比之前的那个要认真了，用了成员变量来记录位置，而不是用全局变量。

用 `BatchGenerator.next()` 方法，可以获取一批子字符串用于训练。

`batch_size` 是每批几串字符串，`num_unrollings` 是每串子字符串的长度（实际上字符串开头还加了上一次获取的最后一个字符，所以实际上字符串长度要比 `num_unrollings` 多一个）。

```python
batch_size=64
num_unrollings=10

class BatchGenerator(object):
  def __init__(self, text, batch_size, num_unrollings):
    self._text = text
    self._text_size = len(text)
    self._batch_size = batch_size
    self._num_unrollings = num_unrollings
    segment = self._text_size // batch_size
    self._cursor = [ offset * segment for offset in range(batch_size)]
    self._last_batch = self._next_batch()

  def _next_batch(self):
    """Generate a single batch from the current cursor position in the data."""
    batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
    for b in range(self._batch_size):
      batch[b, char2id(self._text[self._cursor[b]])] = 1.0
      self._cursor[b] = (self._cursor[b] + 1) % self._text_size
    return batch

  def next(self):
    """Generate the next array of batches from the data. The array consists of
    the last batch of the previous array, followed by num_unrollings new ones.
    """
    batches = [self._last_batch]
    for step in range(self._num_unrollings):
      batches.append(self._next_batch())
    self._last_batch = batches[-1]
    return batches
```

真不愧是优秀程序员写的代码，这个函数写的又让我学习了！

它在初始化的时候先根据 `batch_size` 把段分好，然后设立一组游标 `_cursor` ，是一组哦，不是一个哦！然后定义好 `_last_batch` 看或许到哪了。

然后获取需要的字符串的时候，是一批一批的获取各个字符。

这样做，就可以针对整段字符串均匀的取样，从而避免某些地方学的太细，某些地方又没有学到。

值得注意的是，在RNN准备数据的时候，所喂数据的结构是很容易搞错的。在前面博客中，也有很多同学对于他使用 `transpose` 的意义没法理解。这里需要详细记录一下。

`BatchGenerator.next()` 返回的数据格式，是一个list，list的长度是 `num_unrollings+1`，每一个元素，都是一个(`batch_size`,27)的array，27是 `vocabulary_size`，一个27维向量代表一个字符，是one-hot encoding的格式。

所以，**喂这一批数据进神经网络的时候，理论上是先进去一批的首字符，然后再进去同一批的第二个字符，然后再进去同一批的第三个字符…**

也就是说，下图才是真正的RNN的结构，我们要做的，是按照顺序一个一个的按顺序把东西喂进去。这个图，我看到名字叫 `RNN-rolled`：

![RNN-rolled](https://liusida.github.io/images/2016-11-16-study-lstm/RNN-rolled.png)

我们平时看到的向右一路展开的RNN其实向右方向（我用了虚线）是代表先后顺序（同时也带记忆数据流），跟上下方向意义是不一样的。有没有同学误解那么一排东西是可以同时喂进去的？这个图，我看到名字叫 `RNN-unrolled`。

![RNN-unrolled](https://liusida.github.io/images/2016-11-16-study-lstm/RNN-unrolled.png)

### 7. 另外两个工具函数

再定义两个用来把训练数据转换成可展现字符串的函数。

`characters` 先从one-hot encoding变回数字，再用id2char变成字符。

`batches2string` 则将训练数据变成可以展现的字符串。高手这么一批一批的处理数据逻辑还这么绕，而不是按凡人逻辑一个一个的处理让我觉得有点窒息的感觉，自感智商捉急了。

```python
def characters(probabilities):
  """Turn a 1-hot encoding or a probability distribution over the possible
  characters back into its (most likely) character representation."""
  return [id2char(c) for c in np.argmax(probabilities, 1)]

def batches2string(batches):
  """Convert a sequence of batches back into their (most likely) string
  representation."""
  s = [''] * batches[0].shape[0]
  for b in batches:
    s = [''.join(x) for x in zip(s, characters(b))]
  return s

train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, 1)

print(batches2string(train_batches.next()))
print(batches2string(train_batches.next()))
print(batches2string(valid_batches.next()))
print(batches2string(valid_batches.next()))
```

