from typing import List
from math import log2
import numpy as np

def compute_bins(dataset: np.ndarray, k: int = 5):
   """Compute bin edges for each numeric column (excluding label)."""
   n_rows, n_cols = dataset.shape
   bin_edges = []
   for j in range(n_cols):
      col = dataset[:, j]
      try:
         col = col.astype(float)
      except ValueError:
         bin_edges.append(None)  # non-numeric
         continue
      edges = np.quantile(col, np.linspace(0, 1, k + 1))
      edges[0] -= 1e-3
      bin_edges.append(edges)
   return bin_edges


def apply_bins(dataset: np.ndarray, bin_edges):
   """Assign each numeric value to the corresponding bin."""
   data = dataset.copy().astype(object)
   for j, edges in enumerate(bin_edges):
      if edges is None:
         continue
      col = dataset[:, j].astype(float)
      bin_indices = np.digitize(col, edges)
      data[:, j] = [f"Col{j}Bin{b}" for b in bin_indices]
   return data



def argmax(f, D, *args):
   _dict = f(D, *args)
   assert _dict is not None
   # Sort by value, then by key. That returns max value for key,
   # and ties are broken by comparing keys.
   return sorted(_dict.items(), key=lambda x: (-x[1], x[0]))[0][0]

def count_labels(dataset) -> dict:
   assert dataset is not None
   cnt = {} 

   # count label occurences
   for vals in dataset:
      if vals[-1] not in cnt:
         cnt[vals[-1]] = 1
      else:
         cnt[vals[-1]] += 1

   return cnt

"""   
Calculate enthropy of dataset given as list of entries (lines), discriminated by labels. 
Set verbose = True to see debug.
"""
def label_enthropy(dataset, verbose = False) -> float:
   if verbose : print("calculating enthropy...")
   
   n = len(dataset)
   ret = 0
   for k in count_labels(dataset).values():
      if k != 0:              # 0*log0 := 0
         e = k/n * log2(k/n)
         if verbose : print(f"{k}/{n}", abs(e))
         ret -= e
   
   return ret

"""
Returns information gain for each x from X.
"""
def IG(D, X) -> dict:
   
   IGs = {}
   ED = label_enthropy(D) # start value for IG
   for i in range(len(X)):
      x = X[i]
      cnt = ED
      for v in vals(i, D):
         Dxv = subset(i, v, D)
         cnt -= len(Dxv) * label_enthropy(Dxv) / len(D)
      IGs[x] = cnt
   
   # optional output of information gain at each step
   """
    for key, val in IGs.items():
      print(f"IG({key})={val:.4f}", end = ' ')
   print()
   """
   
   return IGs

"""
X - all features (header without goal)
y - label
D - dataset
Set verbose = True for debug output.
"""
def id3(D, D_parent, X, y, depth = None, verbose = False):
   if D is None:
      if verbose : print("1st if")
      v = argmax(count_labels, D_parent) # most common label in parent node
      return Node(v)
   v = argmax(count_labels, D) # most common label in this node
   if X is None or np.all(D[:, -1] == v) or (depth is not None and depth <= 0):
      if verbose : print("2nd if", v, depth)
      return Node(v)
   x = argmax(IG, D, X) # most discriminative feature - best splits the dataset
   if verbose : print("argmax = ", x)
   subtrees = []
   if depth is not None:
      depth -= 1   
   i = int(np.where(X == x)[0][0])
   for v in vals(i, D):
      _Dxv = subset(i, v, D) # remove rows that don't have x=v
      Dxv = np.delete(_Dxv, i, axis=1) # remove column with feature = x
      _X = np.delete(X, i) # X \ {x}
      t = id3(Dxv, D, _X, y, depth, verbose = False)
      subtrees.append((v,t))
   
   return Node(x, subtrees, dataset=D)
   
"""
Split by feature.
E.g. for getting all entries that have Istra in 1st column i.e. D_{x0 = Istra}
   feature_ind = 0, feature_value = Istra.
feature_ind is the index of the feature in the original dataset and is 0-indexed.
"""
def subset(feature_ind : int, feature_value : str, dataset : List[List[str]]) -> List[List[str]]:
   return [line for line in dataset if line[feature_ind] == feature_value]

"""
Returns a set of all possible values of feature x that is in column `ind` of header (0-indexed).
"""
def vals(ind : int, dataset) -> set:
   return set(line[ind] for line in dataset)
   
"""
A decision tree node.
For creating a leaf, call Node(label).
"""
class Node:

   """
   The dataset subset that is relevant to this node is stored in `dataset`.
   Children is a list of pairs (value of feature that led to it, child node)
   """
   def __init__(self, label, children = None, dataset = None):
      self.label = label
      if children is None:
         self.children = []
      else:
         self.children = children 
      self.dataset = dataset


   """
   Returns ID3 tree branches as list of strings formatted as required
   """
   def branches(self, level = 1, line = None):
      if line is None:
         line = [] # setting line = [] in function args doesn't work because arg is a pointer to the arr
      bs = []
      if not self.children: # leaf
         bs.append(line + [self.label])
      else:
         for val, child in self.children:
            _line = line + [f"{level}:{self.label}={val}"]
            bs.extend(child.branches(level + 1, _line))
      
      return bs


   def get_representation(self):
      bs = self.branches()
      out = "[BRANCHES]:\n"
      for branch in bs:
         out += " ".join(map(str, branch)) + "\n"
      return out
   
"""
The ID3 tree model.
Use fit to build the tree, and predict to predict labels.
"""
class ID3():
   
   """
   Builds tree model. Feature names are to be passed via _header, and _dataset as list of entries.
   The last column should contain the label.
   Optional arguments: 
   depth, the maximum ID3 tree depth
   delta, the number of bins for numerical variables
   verbose, the toggle for verbose debugging output
   """
   def fit(self, _header : List[str], _dataset : List[List[str]], depth = None, verbose = False):
      self.header = np.array(_header)
      self.dataset = np.array(_dataset)
      self.classes = vals(len(self.header)-1, self.dataset)
      self.tree = id3(self.dataset, self.dataset, self.header[:-1], self.classes, depth, verbose) # root of ID3 tree
      # print(self.tree.get_representation())
      
   
   """
   Predicts labels based on dataset. Returns a list of predictions.
   
   """
   def predict(self, header : List[str], dataset : List[List[str]], verbose = False) -> List[str]:
      # print("[PREDICTIONS]:", end = ' ')
      predictions = []
      for entry in dataset:
         node = self.tree
         out = ""
         while(node.label in header): # until node is leaf
            # find value of current node's feature in test data entry
            i = int(np.where(header == node.label)[0][0])
            if verbose : print("node label = ", node.label)
            found = False
            for val, child in node.children:
               # print("testing equality", entry[i], val)
               if entry[i] == val: # train feature matches test feature
                  node = child # go to that branch
                  found = True
                  out = node.label
                  break
            # if new feature value is found, return the most common goal feature value. In case of ties, sort alphabetically
            if not found:
               out = argmax(count_labels, self.dataset)
               break
         if verbose : print ("exited while")
         predictions.append(out)
      return predictions

      

   def test(self):
      pass

   