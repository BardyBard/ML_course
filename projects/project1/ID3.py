from typing import List
from math import log2
import numpy as np
from helpers import *

"""
This file contains all helper functions used by the ID3 decision tree model, called from run.py.
"""

"""Calculate the loss using MSE.

   Args:
      y: numpy array of shape=(N, )
      tx: numpy array of shape=(N, D)
      w: numpy array of shape=(D, ). The vector of model parameters.

   Returns:
      the value of the loss (a scalar), corresponding to the input parameters w.
   """

def ID3_format(x, y):
   """
   Format the test data for the ID3 function.
   Namely, a header is generated and x and y are pasted into one matrix.
   Args:
      x: numpy array of shape=(N', D')
      y: numpy array of shape=(N', )
   Returns:
      dummy_header: numpy array of shape=(, D')
      train_data: numpy array of shape=(N', D'+1)
   """
   # we got rid of feature names in preprocess and I don't want to change that code so I assign some names here
   dummy_header = np.array([f"col{i}" for i in range(x.shape[1])] + ["label"]) 
   train_data = np.hstack((x, y)) # paste x and y together to comply to ID3 method's format
   return dummy_header, train_data

def compute_bins(dataset: np.ndarray, k: int = 3):
   """
   Compute bin edges for each numeric column (excluding the label column).
   Args:
      dataset: numpy aray of shape=(N, D), where N is the number of samples and D the number of columns.
      k: int, optional (default=3). The number of bins to divide numeric columns into.
   Returns:
      bin_edges: list of numpy arrays or None values. Each element coresponds to a column; numeric columns contain an array of bin edges, while non-numeric columns contain None.
   """
   _, n_cols = dataset.shape
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
   """
   Assign each numeric value in the dataset to a bin defined by bin__edges.
   Args:
      dataset: numpy array of shape=(N, D), original dataset.
      bin_edges: list of arrays or None, output of compute_bins().
   Returns:
      data: numpy array of shape=(N, D) with numeric values reeplaced by categorical bin labels.
   """
   data = dataset.copy().astype(object)
   for j, edges in enumerate(bin_edges):
      if edges is None:
         continue
      col = dataset[:, j].astype(float)
      bin_indices = np.digitize(col, edges)
      data[:, j] = [f"Col{j}Bin{b}" for b in bin_indices]
   return data



def argmax(f, D, *args):
   """
   Return the key corresponding to the maximum value in a dictionary returned by function f.
   Args:
      f: function that takes D and *args and returns a dictionary {key: value}.
      D: input data passed to f.
      *args: additional arguments for f.
   Returns:
      key: element with the highest value. Ties are broken by smallest key alphabetically.
   """
   _dict = f(D, *args)
   assert _dict is not None
   # Sort by value, then by key. That returns max value for key,
   # and ties are broken by comparing keys.
   return sorted(_dict.items(), key=lambda x: (-x[1], x[0]))[0][0]

def count_labels(dataset) -> dict:
   """
   Count no of occurrences of each label in the dataset.
   Args:
      dataset: numpy array or list of samples, where the last column is the one that contains labels.
   Returns:
      cnt: dict mapping label -> count.
   """
   assert dataset is not None
   cnt = {} 

   # count label occurences
   for vals in dataset:
      if vals[-1] not in cnt:
         cnt[vals[-1]] = 1
      else:
         cnt[vals[-1]] += 1

   return cnt


def label_enthropy(dataset, verbose = False) -> float:
   """
   Calculate enthropy of the label distribution in the dataset. 
   Set verbose = True to see debug.
   Args:
      dataset: list or numpy array of samples, where the last column contains labels.
      verbose: bool, optional (default=False). If True, prints intermediate entropy calculations.
   Returns:
      ret: float, entropy value of the label distribution.
   """
   if verbose : print("calculating enthropy...")
   
   n = len(dataset)
   ret = 0
   for k in count_labels(dataset).values():
      if k != 0:              # 0*log0 := 0
         e = k/n * log2(k/n)
         if verbose : print(f"{k}/{n}", abs(e))
         ret -= e
   
   return ret


def IG(D, X) -> dict:
   """
   Returns information gain for each x from X.
   Args:
      D: numpy array of shape=(N, D), where the last column is the label.
      X: list or numpy array of feature names corresponding to D's columns (excluding label).
   Returns:
      IGs: dict mapping feature name -> information gain value.
   """
   min_gain = 1e-3   
   IGs = {}
   ED = label_enthropy(D) # start value for IG
   for i in range(len(X)):
      x = X[i]
      cnt = ED
      for v in vals(i, D):
         Dxv = subset(i, v, D)
         cnt -= len(Dxv) * label_enthropy(Dxv) / len(D)
      
      if cnt > min_gain : # skip low information-gain features
         IGs[x] = cnt
   
   # optional output of information gain at each step
   """
    for key, val in IGs.items():
      print(f"IG({key})={val:.4f}", end = ' ')
   print()
   """
   
   return IGs


def id3(D, D_parent, X, y, depth = None, verbose = False):
   """
   Recursively build an ID3 decision tree.
   Args:
      D: numpy array, current dataset subset.
      D_parent: numpy array, dataset of the parent node.
      X: numpy array, remaining features (header without label).
      y: numpy array, possible label values.
      depth: int or None, optional (default=None). Maximum recursion depth; None for unlimited.
      verbose: bool, optional (default=False). Print debug output if True.
   Returns:
      Node: root node of the constructed decision tree.
   """
   if D is None:
      if verbose : print("1st if")
      v = argmax(count_labels, D_parent) # most common label in parent node
      return Node(v)
   v = argmax(count_labels, D) # most common label in this node
   if X is None or len(X) == 0 or np.all(D[:, -1] == v) or (depth is not None and depth <= 0):
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
   

def subset(feature_ind: int, feature_value: str, dataset: np.ndarray) -> np.ndarray:
   """
   Split by feature.
   E.g. for getting all entries that have Istra in 1st column i.e. D_{x0 = Istra}
      feature_ind = 0, feature_value = Istra.
   feature_ind is the index of the feature in the original dataset and is 0-indexed.
   Args:
      feature_ind: int, index of the feature column (0-indexed).
      feature_value: str, feature value to filter by.
      dataset: numpy array of shape=(N, D), input dataset.
   Returns:
      subset: numpy array containing only rows with feature_value at feature_ind.
   """
   return dataset[dataset[:, feature_ind] == feature_value]



def vals(ind: int, dataset: np.ndarray) -> set:
   """
   Get all unique values of the feature at a given column index.
   Args:
      ind: int, column index of the feature.
      dataset: numpy array of shape=(N, D), input dataset.
   Returns:
      values: set of unique feature values in column ind.
   """
   return set(dataset[:, ind])

class Node:
   """
   A decision tree node. Can represent either an internal decision node or a leaf.
   For creating a leaf, call Node(label).
   """


   def __init__(self, label, children = None, dataset = None):
      """
      Initialize a new Node. The dataset subset that is relevant to this node is stored in `dataset`.
      Children is a list of pairs (value of feature that led to it, child node)
      Args:
         label: str or object, name of the feature or label value if leaf.
         children: list of tuples (feature_value, child_node), optional (default=None).
         dataset: numpy array or None, dataset subset associated with this node.
      """
      
      self.label = label
      if children is None:
         self.children = []
      else:
         self.children = children 
      self.dataset = dataset


   def branches(self, level = 1, line = None):
      """
      Returns ID3 tree branches as list of strings formatted as required.
      Args:
         level: int, current tree depth (used internally for recursion).
         line: list, accumulated branch path.
      Returns:
         bs: list of lists, where each sublist represents one full branch from root to leaf.
      """
      
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
      """
      Get a string representation of all branches in the tree.
      Returns:
         out: str, formatted list of tree branches.
      """
      bs = self.branches()
      out = "[BRANCHES]:\n"
      for branch in bs:
         out += " ".join(map(str, branch)) + "\n"
      return out
   

class ID3():
   """
   The ID3 tree model.
   Use fit to build the tree, and predict to predict labels.
   """
   
   
   def fit(self, _header : List[str], _dataset : List[List[str]], depth = None, verbose = False):
      """
      Builds tree model. Feature names are to be passed via _header, and _dataset as list of entries.
      The last column should contain the label.
      Optional arguments: 
      depth, the maximum ID3 tree depth
      delta, the number of bins for numerical variables
      verbose, the toggle for verbose debugging output
      """
      assert(len(_header) == len(_dataset[0]))
      self.header = np.array(_header)
      self.dataset = np.array(_dataset)
      self.classes = set(self.dataset[:, -1])
      self.tree = id3(self.dataset, self.dataset, self.header[:-1], self.classes, depth, verbose) # root of ID3 tree
      
   

   def predict(self, header : List[str], dataset : List[List[str]], verbose = False) -> List[str]:
      """
      Predict labels for given dataset using the trained tree.
      Args:
         header: list of feature names corresponding to dataset columns.
         dataset: list of samples, each a list of feature values.
         verbose: bool, optional (default=False). Print debug info if True.
      Returns:
         predictions: list of predicted labels, one per input sample.
      """
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


def test_hyperparams(x_train, y_train, max_depth: int, k=3, seed = 42, subsample_size = 3000):
   """
   A helper function that finds the best tree depth for the ID3 model with k-fold cross-vallidation.
   Args:
      x_train: numpy array of training features, shape=(N, D).
      y_train: numpy array of training labels, shape=(N, ).
      max_depth: int, maximum tree depth to test.
      k: int, optional (default=3). Number of folds in cross-validation.
      seed: int, optional (default=42). Random seed for reproducibility.
      subsample_size: int or None, optional (default=3000). Size of random subsample for faster evaluation.
   Returns:
      best_depth: int, depth which achieved the best mean F1-score across folds.
   """
   assert len(x_train) == len(y_train)

   # take a subsample for hyperparameter testing
   if subsample_size and len(x_train) > subsample_size:
      np.random.seed(seed)
      idx = np.random.choice(len(x_train), subsample_size, replace=False)
      x_train = x_train[idx]
      y_train = y_train[idx]
   
   # permute the data randomly
   if seed is not None:
      np.random.seed(seed)
      perm = np.random.permutation(len(x_train))
      x_train = x_train[perm]
      y_train = y_train[perm]
   
   folds = kfold_inds(len(x_train), k)
   best_score = -1.0
   do_break = False
   
   for depth in range(1, max_depth + 1):
      print(f"Testing tree depth = {depth}")
      scores = []
      
      best_fold_score = -1.0

      for fold in folds:
         start, end = fold

         # Split into validation and training
         xi_test = x_train[start:end]
         yi_test = y_train[start:end]

         xi_train = np.concatenate((x_train[:start], x_train[end:]), axis=0)
         yi_train = np.concatenate((y_train[:start], y_train[end:]), axis=0)

         header, train_data = ID3_format(xi_train, yi_train)

         try:
            model = ID3()
            model.fit(header, train_data, depth, verbose=False)

            header_test, test_data = ID3_format(xi_test, yi_test)
            predictions = model.predict(header_test, test_data, verbose=False)

         except Exception as e:
            print("depth too large, interrupting...")
            do_break = True
            break
         score = metric(predictions, yi_test) 
         scores.append(score)
         
         if score > best_fold_score:
               best_fold_score = score
               best_fold_preds, true_ys = predictions, yi_test
      
      if do_break:
         break
      
      
      mean_acc = np.mean(scores)
      print(f"Mean F1-score (depth={depth}): {mean_acc:.4f}")

      if mean_acc > best_score:
         best_score = mean_acc
         best_depth = depth
         ### Generate confusion matrix
         y_true = np.array(true_ys, dtype=int).ravel()
         y_pred = np.array(best_fold_preds, dtype=int)
         cm = create_confusion_matrix(y_true, y_pred)
         

   print(f"\nBest depth = {best_depth} with mean F1-score = {best_score:.4f}")
   # display the best confusion matrix
   _ = cm_visualization(cm, f"depth = {depth}")
   return best_depth    