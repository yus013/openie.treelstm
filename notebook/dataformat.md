# Data Format

## Summary
* data/[train, dev, test]

## arb.npy
* [([ai], [ri], [bi]), ... ]
* list(tuple(list(int), list(int), list(int)))
* example: [[[0], [1, 2], [2, 3]] ... ]

```python
arbs = np.load(arb_file_path)
```

## parents.npy
* number refers to the node's father
* index starts from 1
* 0 stants for root
* list(list(int))
* example: [[1, 2, 4, 0,]]

```python
parents = np.load(parents_file_path)
```

## sent.pkl
* list(list(str))
* [['Hello', 'world'], ...]

```python
parents = pd.read_pickle(sentence_file_path)
```
