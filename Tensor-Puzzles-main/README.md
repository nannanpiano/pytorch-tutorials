# Tensor Puzzles 张量谜题
- by [Sasha Rush](http://rush-nlp.com) - [srush_nlp](https://twitter.com/srush_nlp) (with Marcos Treviso)




When learning a tensor programming language like PyTorch or Numpy it
is tempting to rely on the standard library (or more honestly
StackOverflow) to find a magic function for everything.  But in
practice, the tensor language is extremely expressive, and you can
do most things from first principles and clever use of broadcasting.

在学习 PyTorch 或 NumPy 这类张量编程语言时，人们很容易依赖标准库（说实话，更多是依赖 StackOverflow）去为每件事找一个"魔法函数"。但实际上，张量语言的表达能力极强，大多数事情你都可以从第一性原理出发、巧妙地利用广播（broadcasting）来完成。


This is a collection of 21 tensor puzzles. Like chess puzzles these are
not meant to simulate the complexity of a real program, but to practice
in a simplified environment. Each puzzle asks you to reimplement one
function in the NumPy standard library without magic. 

这是一组共 21 道张量谜题。就像国际象棋残局题一样，它们并不是为了模拟真实程序的复杂度，而是让你在一个简化的环境中练习。每道谜题都要求你在不使用"魔法"的前提下，重新实现 NumPy 标准库中的一个函数。


I recommend running in Colab. Click here and copy the notebook to get start.

推荐在 Colab 中运行。点击下面的按钮，把 notebook 复制一份即可开始。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srush/Tensor-Puzzles/blob/main/Tensor%20Puzzlers.ipynb)

If you are interested, there is also a youtube walkthrough of the puzzles 

如果感兴趣，还有一个 YouTube 视频逐题讲解这些谜题。

[![Watch the video](https://img.youtube.com/vi/SiwTAyyvt5s/default.jpg)](https://youtu.be/Hafo7hIl8MU)

```python
!pip install -qqq torchtyping hypothesis pytest git+https://github.com/danoneata/chalk@srush-patch-1
!wget -q https://github.com/srush/Tensor-Puzzles/raw/main/lib.py
```


```python
from lib import draw_examples, make_test, run_test
import torch
import numpy as np
from torchtyping import TensorType as TT
tensor = torch.tensor
```

## Rules 规则

1. These puzzles are about *broadcasting*. Know this rule.

   这些谜题的核心是 *广播（broadcasting）*。务必掌握下图中的规则。

![](https://pbs.twimg.com/media/FQywor0WYAssn7Y?format=png&name=large)

2. Each puzzle needs to be solved in 1 line (<80 columns) of code.

   每道谜题都必须用 1 行代码（少于 80 列）解决。
3. You are allowed @, arithmetic, comparison, `shape`, any indexing (e.g. `a[:j], a[:, None], a[arange(10)]`), and previous puzzle functions.

   允许使用的有：`@`（矩阵乘法）、算术运算、比较运算、`shape`、任意索引（例如 `a[:j], a[:, None], a[arange(10)]`），以及前面谜题中你已实现的函数。
4. You are *not allowed* anything else. No `view`, `sum`, `take`, `squeeze`, `tensor`.

   除此之外的一切都 *不允许* 使用。不能用 `view`、`sum`、`take`、`squeeze`、`tensor`。

5. You can start with these two functions:

   你可以从下面这两个函数开始：


```python
def arange(i: int):
    "Use this function to replace a for-loop."
    return torch.tensor(range(i))

draw_examples("arange", [{"" : arange(i)} for i in [5, 3, 9]])
```




    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_7_0.svg)
    




```python
# Example of broadcasting.  广播的示例。
examples = [(arange(4), arange(5)[:, None]) ,
            (arange(3)[:, None], arange(2))]
draw_examples("broadcast", [{"a": a, "b":b, "ret": a + b} for a, b in examples])
```




    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_8_0.svg)
    




```python
def where(q, a, b):
    "Use this function to replace an if-statement."
    return (q * a) + (~q) * b

# In diagrams, orange is positive/True, where is zero/False, and blue is negative.
# 图示中，橙色表示正数/True，白色表示零/False，蓝色表示负数。

examples = [(tensor([False]), tensor([10]), tensor([0])),
            (tensor([False, True]), tensor([1, 1]), tensor([-10, 0])),
            (tensor([False, True]), tensor([1]), tensor([-10, 0])),
            (tensor([[False, True], [True, False]]), tensor([1]), tensor([-10, 0])),
            (tensor([[False, True], [True, False]]), tensor([[0], [10]]), tensor([-10, 0])),
           ]
draw_examples("where", [{"q": q, "a":a, "b":b, "ret": where(q, a, b)} for q, a, b in examples])
```




    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_9_0.svg)
    



## Puzzle 1 - ones 谜题 1 - ones

Compute [ones](https://numpy.org/doc/stable/reference/generated/numpy.ones.html) - the vector of all ones.

计算 [ones](https://numpy.org/doc/stable/reference/generated/numpy.ones.html) —— 全为 1 的向量。


```python
def ones_spec(out):
    for i in range(len(out)):
        out[i] = 1
        
def ones(i: int) -> TT["i"]:
    raise NotImplementedError

test_ones = make_test("one", ones, ones_spec, add_sizes=["i"])
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_11_0.svg)
    



```python
# run_test(test_ones)
```

## Puzzle 2 - sum 谜题 2 - sum

Compute [sum](https://numpy.org/doc/stable/reference/generated/numpy.sum.html) - the sum of a vector.

计算 [sum](https://numpy.org/doc/stable/reference/generated/numpy.sum.html) —— 向量所有元素之和。


```python
def sum_spec(a, out):
    out[0] = 0
    for i in range(len(a)):
        out[0] += a[i]
        
def sum(a: TT["i"]) -> TT[1]:
    raise NotImplementedError


test_sum = make_test("sum", sum, sum_spec)
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_14_0.svg)
    



```python
# run_test(test_sum)
```

## Puzzle 3 - outer 谜题 3 - outer

Compute [outer](https://numpy.org/doc/stable/reference/generated/numpy.outer.html) - the outer product of two vectors.

计算 [outer](https://numpy.org/doc/stable/reference/generated/numpy.outer.html) —— 两个向量的外积。


```python
def outer_spec(a, b, out):
    for i in range(len(out)):
        for j in range(len(out[0])):
            out[i][j] = a[i] * b[j]
            
def outer(a: TT["i"], b: TT["j"]) -> TT["i", "j"]:
    raise NotImplementedError
    
test_outer = make_test("outer", outer, outer_spec)
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_17_0.svg)
    



```python
# run_test(test_outer)
```

## Puzzle 4 - diag 谜题 4 - diag

Compute [diag](https://numpy.org/doc/stable/reference/generated/numpy.diag.html) - the diagonal vector of a square matrix.

计算 [diag](https://numpy.org/doc/stable/reference/generated/numpy.diag.html) —— 方阵的对角线向量。


```python
def diag_spec(a, out):
    for i in range(len(a)):
        out[i] = a[i][i]
        
def diag(a: TT["i", "i"]) -> TT["i"]:
    raise NotImplementedError


test_diag = make_test("diag", diag, diag_spec)
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_20_0.svg)
    



```python
# run_test(test_diag)
```

## Puzzle 5 - eye 谜题 5 - eye

Compute [eye](https://numpy.org/doc/stable/reference/generated/numpy.eye.html) - the identity matrix.

计算 [eye](https://numpy.org/doc/stable/reference/generated/numpy.eye.html) —— 单位矩阵。


```python
def eye_spec(out):
    for i in range(len(out)):
        out[i][i] = 1
        
def eye(j: int) -> TT["j", "j"]:
    raise NotImplementedError
    
test_eye = make_test("eye", eye, eye_spec, add_sizes=["j"])
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_23_0.svg)
    



```python
# run_test(test_eye)
```

## Puzzle 6 - triu 谜题 6 - triu

Compute [triu](https://numpy.org/doc/stable/reference/generated/numpy.triu.html) - the upper triangular matrix.

计算 [triu](https://numpy.org/doc/stable/reference/generated/numpy.triu.html) —— 上三角矩阵。


```python
def triu_spec(out):
    for i in range(len(out)):
        for j in range(len(out)):
            if i <= j:
                out[i][j] = 1
            else:
                out[i][j] = 0
                
def triu(j: int) -> TT["j", "j"]:
    raise NotImplementedError


test_triu = make_test("triu", triu, triu_spec, add_sizes=["j"])
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_26_0.svg)
    



```python
# run_test(test_triu)
```

## Puzzle 7 - cumsum 谜题 7 - cumsum

Compute [cumsum](https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html) - the cumulative sum.

计算 [cumsum](https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html) —— 累积和（前缀和）。


```python
def cumsum_spec(a, out):
    total = 0
    for i in range(len(out)):
        out[i] = total + a[i]
        total += a[i]

def cumsum(a: TT["i"]) -> TT["i"]:
    raise NotImplementedError

test_cumsum = make_test("cumsum", cumsum, cumsum_spec)
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_29_0.svg)
    



```python
# run_test(test_cumsum)
```

## Puzzle 8 - diff 谜题 8 - diff

Compute [diff](https://numpy.org/doc/stable/reference/generated/numpy.diff.html) - the running difference.

计算 [diff](https://numpy.org/doc/stable/reference/generated/numpy.diff.html) —— 相邻元素之差（逐项差分）。


```python
def diff_spec(a, out):
    out[0] = a[0]
    for i in range(1, len(out)):
        out[i] = a[i] - a[i - 1]

def diff(a: TT["i"], i: int) -> TT["i"]:
    raise NotImplementedError

test_diff = make_test("diff", diff, diff_spec, add_sizes=["i"])
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_32_0.svg)
    



```python
# run_test(test_diff)
```

## Puzzle 9 - vstack 谜题 9 - vstack

Compute [vstack](https://numpy.org/doc/stable/reference/generated/numpy.vstack.html) - the matrix of two vectors

计算 [vstack](https://numpy.org/doc/stable/reference/generated/numpy.vstack.html) —— 由两个向量（按行堆叠）组成的矩阵。


```python
def vstack_spec(a, b, out):
    for i in range(len(out[0])):
        out[0][i] = a[i]
        out[1][i] = b[i]

def vstack(a: TT["i"], b: TT["i"]) -> TT[2, "i"]:
    raise NotImplementedError


test_vstack = make_test("vstack", vstack, vstack_spec)
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_35_0.svg)
    



```python
# run_test(test_vstack)
```

## Puzzle 10 - roll 谜题 10 - roll

Compute [roll](https://numpy.org/doc/stable/reference/generated/numpy.roll.html) - the vector shifted 1 circular position.

计算 [roll](https://numpy.org/doc/stable/reference/generated/numpy.roll.html) —— 将向量循环移动 1 个位置。


```python
def roll_spec(a, out):
    for i in range(len(out)):
        if i + 1 < len(out):
            out[i] = a[i + 1]
        else:
            out[i] = a[i + 1 - len(out)]
            
def roll(a: TT["i"], i: int) -> TT["i"]:
    raise NotImplementedError


test_roll = make_test("roll", roll, roll_spec, add_sizes=["i"])
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_38_0.svg)
    



```python
# run_test(test_roll)
```

## Puzzle 11 - flip 谜题 11 - flip

Compute [flip](https://numpy.org/doc/stable/reference/generated/numpy.flip.html) - the reversed vector

计算 [flip](https://numpy.org/doc/stable/reference/generated/numpy.flip.html) —— 反转后的向量。


```python
def flip_spec(a, out):
    for i in range(len(out)):
        out[i] = a[len(out) - i - 1]
        
def flip(a: TT["i"], i: int) -> TT["i"]:
    raise NotImplementedError


test_flip = make_test("flip", flip, flip_spec, add_sizes=["i"])
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_41_0.svg)
    



```python
# run_test(test_flip)
```

## Puzzle 12 - compress 谜题 12 - compress


Compute [compress](https://numpy.org/doc/stable/reference/generated/numpy.compress.html) - keep only masked entries (left-aligned).

计算 [compress](https://numpy.org/doc/stable/reference/generated/numpy.compress.html) —— 只保留掩码（mask）为真的元素（左对齐）。


```python
def compress_spec(g, v, out):
    j = 0
    for i in range(len(g)):
        if g[i]:
            out[j] = v[i]
            j += 1
            
def compress(g: TT["i", bool], v: TT["i"], i:int) -> TT["i"]:
    raise NotImplementedError


test_compress = make_test("compress", compress, compress_spec, add_sizes=["i"])
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_44_0.svg)
    



```python
# run_test(test_compress)
```

## Puzzle 13 - pad_to 谜题 13 - pad_to


Compute pad_to - eliminate or add 0s to change size of vector.

计算 pad_to —— 通过删除元素或补 0 来改变向量的长度。


```python
def pad_to_spec(a, out):
    for i in range(min(len(out), len(a))):
        out[i] = a[i]


def pad_to(a: TT["i"], i: int, j: int) -> TT["j"]:
    raise NotImplementedError


test_pad_to = make_test("pad_to", pad_to, pad_to_spec, add_sizes=["i", "j"])
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_47_0.svg)
    



```python
# run_test(test_pad_to)
```

## Puzzle 14 - sequence_mask 谜题 14 - sequence_mask


Compute [sequence_mask](https://www.tensorflow.org/api_docs/python/tf/sequence_mask) - pad out to length per batch.

计算 [sequence_mask](https://www.tensorflow.org/api_docs/python/tf/sequence_mask) —— 按每个 batch 各自的长度做掩码填充（超出长度的位置置 0）。


```python
def sequence_mask_spec(values, length, out):
    for i in range(len(out)):
        for j in range(len(out[0])):
            if j < length[i]:
                out[i][j] = values[i][j]
            else:
                out[i][j] = 0
    
def sequence_mask(values: TT["i", "j"], length: TT["i", int]) -> TT["i", "j"]:
    raise NotImplementedError


def constraint_set_length(d):
    d["length"] = d["length"] % d["values"].shape[1]
    return d


test_sequence = make_test("sequence_mask",
    sequence_mask, sequence_mask_spec, constraint=constraint_set_length
)
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_50_0.svg)
    



```python
# run_test(test_sequence)
```

## Puzzle 15 - bincount 谜题 15 - bincount

Compute [bincount](https://numpy.org/doc/stable/reference/generated/numpy.bincount.html) - count number of times an entry was seen.

计算 [bincount](https://numpy.org/doc/stable/reference/generated/numpy.bincount.html) —— 统计每个值出现的次数。


```python
def bincount_spec(a, out):
    for i in range(len(a)):
        out[a[i]] += 1
        
def bincount(a: TT["i"], j: int) -> TT["j"]:
    raise NotImplementedError


def constraint_set_max(d):
    d["a"] = d["a"] % d["return"].shape[0]
    return d


test_bincount = make_test("bincount",
    bincount, bincount_spec, add_sizes=["j"], constraint=constraint_set_max
)
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_53_0.svg)
    



```python
# run_test(test_bincount)
```

## Puzzle 16 - scatter_add 谜题 16 - scatter_add

Compute [scatter_add](https://pytorch-scatter.readthedocs.io/en/1.3.0/functions/add.html) - add together values that link to the same location.

计算 [scatter_add](https://pytorch-scatter.readthedocs.io/en/1.3.0/functions/add.html) —— 把指向同一位置的值加在一起。


```python
def scatter_add_spec(values, link, out):
    for j in range(len(values)):
        out[link[j]] += values[j]
        
def scatter_add(values: TT["i"], link: TT["i"], j: int) -> TT["j"]:
    raise NotImplementedError


def constraint_set_max(d):
    d["link"] = d["link"] % d["return"].shape[0]
    return d


test_scatter_add = make_test("scatter_add",
    scatter_add, scatter_add_spec, add_sizes=["j"], constraint=constraint_set_max
)
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_56_0.svg)
    



```python
# run_test(test_scatter_add)
```

## Puzzle 17 - flatten 谜题 17 - flatten

Compute [flatten](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html)

计算 [flatten](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html) —— 将矩阵展平为一维向量。


```python
def flatten_spec(a, out):
    k = 0
    for i in range(len(a)):
        for j in range(len(a[0])):
            out[k] = a[i][j]
            k += 1

def flatten(a: TT["i", "j"], i:int, j:int) -> TT["i * j"]:
    raise NotImplementedError

test_flatten = make_test("flatten", flatten, flatten_spec, add_sizes=["i", "j"])
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_59_0.svg)
    



```python
# run_test(test_flatten)
```

## Puzzle 18 - linspace 谜题 18 - linspace

Compute [linspace](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html)

计算 [linspace](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html) —— 在区间内生成等间距的数。


```python
def linspace_spec(i, j, out):
    for k in range(len(out)):
        out[k] = float(i + (j - i) * k / max(1, len(out) - 1))

def linspace(i: TT[1], j: TT[1], n: int) -> TT["n", float]:
    raise NotImplementedError

test_linspace = make_test("linspace", linspace, linspace_spec, add_sizes=["n"])
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_62_0.svg)
    



```python
# run_test(test_linspace)
```

## Puzzle 19 - heaviside 谜题 19 - heaviside

Compute [heaviside](https://numpy.org/doc/stable/reference/generated/numpy.heaviside.html)

计算 [heaviside](https://numpy.org/doc/stable/reference/generated/numpy.heaviside.html) —— 赫维赛德阶跃函数。


```python
def heaviside_spec(a, b, out):
    for k in range(len(out)):
        if a[k] == 0:
            out[k] = b[k]
        else:
            out[k] = int(a[k] > 0)

def heaviside(a: TT["i"], b: TT["i"]) -> TT["i"]:
    raise NotImplementedError

test_heaviside = make_test("heaviside", heaviside, heaviside_spec)
```


    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_65_0.svg)
    



```python
# run_test(test_heaviside)
```

## Puzzle 20 - repeat (1d) 谜题 20 - repeat（一维）

Compute [repeat](https://pytorch.org/docs/stable/generated/torch.Tensor.repeat.html)

计算 [repeat](https://pytorch.org/docs/stable/generated/torch.Tensor.repeat.html) —— 将向量重复多次。


```python
def repeat_spec(a, d, out):
    for i in range(d[0]):
        for k in range(len(a)):
            out[i][k] = a[k]

def constraint_set(d):
    d["d"][0] = d["return"].shape[0]
    return d

            
def repeat(a: TT["i"], d: TT[1]) -> TT["d", "i"]:
    raise NotImplementedError

test_repeat = make_test("repeat", repeat, repeat_spec, constraint=constraint_set)
```
    
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_68_0.svg)

## Puzzle 21 - bucketize 谜题 21 - bucketize

Compute [bucketize](https://pytorch.org/docs/stable/generated/torch.bucketize.html)

计算 [bucketize](https://pytorch.org/docs/stable/generated/torch.bucketize.html) —— 根据边界把值分到对应的桶中，返回桶的索引。

```python
def bucketize_spec(v, boundaries, out):
    for i, val in enumerate(v):
        out[i] = 0
        for j in range(len(boundaries)-1):
            if val >= boundaries[j]:
                out[i] = j + 1
        if val >= boundaries[-1]:
            out[i] = len(boundaries)


def constraint_set(d):
    d["boundaries"] = np.abs(d["boundaries"]).cumsum()
    return d

            
def bucketize(v: TT["i"], boundaries: TT["j"]) -> TT["i"]:
    raise NotImplementedError

test_bucketize = make_test("bucketize", bucketize, bucketize_spec,
                           constraint=constraint_set)
```
![svg](Tensor%20Puzzlers_files/Tensor%20Puzzlers_69_0.svg)

## Speed Run Mode! 竞速模式！
What is the smallest you can make each of these?

你能把每道题的解法写到多短？

```python
import inspect
fns = (ones, sum, outer, diag, eye, triu, cumsum, diff, vstack, roll, flip,
       compress, pad_to, sequence_mask, bincount, scatter_add)

for fn in fns:
    lines = [l for l in inspect.getsource(fn).split("\n") if not l.strip().startswith("#")]
    
    if len(lines) > 3:
        print(fn.__name__, len(lines[2]), "(more than 1 line)")
    else:
        print(fn.__name__, len(lines[1]))
```

    ones 29
    sum 29
    outer 29
    diag 29
    eye 29
    triu 29
    cumsum 29
    diff 29
    vstack 29
    roll 29
    flip 29
    compress 29
    pad_to 29
    sequence_mask 29
    bincount 29
    scatter_add 29

