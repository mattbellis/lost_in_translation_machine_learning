import numpy as np
import matplotlib.pylab as plt
import random
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc


################################################################################
def sumfunc(dataset, dtype='normal', plot_range=(0,2)):               ## dataset = data_1(nentries,nfeatures)
  mysum = []

  nentries,nfeatures = dataset.shape

  if dtype=='squared':
    for i in range(nentries):
      s = sum(dataset[i]**2)
      mysum.append(s)
  elif dtype=='relativity':
    for i in range(nentries):
      # Don't worry about taking the square root
      s = dataset[i][0]**2 - (dataset[i][1]**2  + dataset[i][2]**2 + dataset[i][3]**2)
      mysum.append(s)
  else:
    for i in range(nentries):
      s = sum(dataset[i])
      mysum.append(s)

  plt.figure()
  plt.hist(mysum,bins=50,range=plot_range);
  #print(mysum)

################################################################################
def histfunc(dataset,plot_range=(0,1)):   ## dataset = data_1(nentries,nfeatures)
  nentries,nfeatures = dataset.shape
  plt.figure(figsize=(16,3))
  for i in range(nfeatures):
    plt.subplot(1,nfeatures,i+1)
    plt.hist(dataset.T[i],bins=50,range=plot_range)

################################################################################
#random
def data_ran(nentries,nfeatures):
  data = np.random.random((nentries,nfeatures))
  return data

##
################################################################################
############################################################################################
def gen_original_data(nentries,nfeatures, dtype='normal'):

  data = np.random.random((nentries,nfeatures))

  # Sum of the squares = 1
  if dtype=='squared':
    data_temp = []
    for i in range(nentries):
      # First square the entries
      sq = data[i]**2
      # Get the sum
      norm = np.sum(sq)
      # Normalize the entries
      dtmp = np.sqrt(sq / norm)
      #print(data[i], sq, norm, dtmp)
      data_temp.append(dtmp)

  elif dtype=='flat':
      data_temp = np.random.random((nentries,nfeatures)).tolist()

  # Sum of the squares = 1
  elif dtype=='relativity':
    data_temp = []
    for i in range(nentries):
      # First square the entries
      sq = data[i]**2
      # Get the sum
      norm_p = sq[1] + sq[2] + sq[3]
      sq[0] = norm_p + 1
      # Normalize the entries
      # Don't worry about taking the square root
      dtmp = np.sqrt(sq)
      #print(data[i], sq, norm, dtmp)
      data_temp.append(dtmp)

  # The "normal" way, Sum of the = 1
  else:
    data_temp = []
    for i in range(nentries):
      norm = np.sum(data[i])
      # Normalize the entries
      dtmp = data[i] / norm
      data_temp.append(dtmp)

  data = np.array(data_temp)
  return data





################################################################################
#
################################################################################
# Should be all or mostly here
# https://colab.research.google.com/drive/1uNKwJaxleCwpp2pOeErTY8ZfPp9aj08E?usp=sharing
################################################################################




####
#NOTES FOR GABBY (connecting this to og code)
#- mydataset = data2
#- shuffled data = data3 **this is what needs to be in beginning of shuffle_dataset function

################################################################################
# def shuffle_dataset  (formerly 'datafunc')
# CHANGES from original code:
# - removed dataset2 as arg and defined in function

def shuffle_dataset(dataset1):
  nentries, nfeatures = dataset1.shape

  # Shuffle dataset 1 to make dataset 2
  dataset2 = np.array(dataset1)
  for i in range(nfeatures):
    np.random.shuffle(dataset2.T[i])
  return dataset2

def concat_dataset(dataset1, dataset2, wantplots=False):
  # Put data and labels together
  X = np.concatenate([dataset1, dataset2])

  y1 = np.zeros(len(dataset1)) # 0 for dataset 1
  y2 = np.ones(len(dataset2))  # 1 for dataset 2
  y = np.concatenate([y1,y2])

  if wantplots==True:
    ## Histograms
    nfeatures = len(dataset1.transpose())
    print(f"nfeatures: {nfeatures}")

    # Dataset 1
    plt.figure(figsize=(14,4))
    for i in range(nfeatures):
      feature = dataset1.transpose()[i]
      plt.subplot(1,5,i+1)
      plt.hist(feature,bins=25,range=(0,1))
      plt.title('Dataset 1')

    # Dataset 2
    plt.figure(figsize=(14,4))
    for i in range(nfeatures):
      feature = dataset2.transpose()[i]
      plt.subplot(1,5,i+1)
      plt.hist(feature,bins=25,range=(0,1))
      plt.title('Dataset 2')
  # X: dataset 1 and 2 values concatenated
  # y: dataset 1 and 2 labels concatenated
  return X,y


################################################################################
# def plot_correlations (formerly 'correlations')
  # CHANGES from original code:
  # - removed X,y as args and defined in function
  # - add dataset1, dataset2
def correlations(dataset1, dataset2, label=0, colormap=plt.cm.viridis, wantplots=False, ax1=None, **kwds):
  """Calculate pairwise correlation between features.

  Extra arguments are passed on to DataFrame.corr()
  """
  X,y= concat_dataset(dataset1, dataset2, wantplots=False)

  num_features = len(X[0])

  num_features_str = [str(x + 1) for x in range(num_features)]
  print('num_features_str:', num_features_str)

  # Correlations wants the data as a dataframe
  # Use "label" to figure out which dataset we plot, once they
  # were merged
  data = pd.DataFrame(X[y == label], columns=num_features_str)

  # simply call df.corr() to get a table of
  # correlation values if you do not need
  # the fancy plotting
  corrmat = data.corr(**kwds)

  if wantplots == True:

    if ax1 is None:
      fig, ax1 = plt.subplots(ncols=1, figsize=(6, 5))

    #opts = {'cmap': plt.get_cmap("RdBu"), 'vmin': -1, 'vmax': +1}
    opts = {'cmap': colormap, 'vmin': -1, 'vmax': +1}
    heatmap1 = ax1.pcolor(corrmat, **opts)
    plt.colorbar(heatmap1, ax=ax1)

    ax1.set_title("Correlations")

    labels = corrmat.columns.values
    for ax in (ax1,):
      # shift location of ticks to center of the bins
      ax.set_xticks(np.arange(len(labels)) + 0.5, minor=False)
      ax.set_yticks(np.arange(len(labels)) + 0.5, minor=False)
      ax.set_xticklabels(labels, minor=False, ha='right', rotation=70)
      ax.set_yticklabels(labels, minor=False)

    plt.tight_layout()

    # How to Call:
    # correlations(dataset1, dataset2, label=0, colormap= plt.cm.viridis, wantplots=True, ax1=plt.gca())
    # label=0: dataset 1 , label=1: dataset 2

################################################################################
# def neuralnet
  # CHANGES from original code:
  # - removed X,y as args and defined in function
  # - removed num_features as args and defined in function
def neuralnet(dataset1, dataset2, num_hidden_layers, wantplots=False):
  # Let's make use of the datasets we created earlier

  X, y = concat_dataset(dataset1, dataset2, wantplots=False)
  num_features= len(X[0])

  X_dev, X_eval, y_dev, y_eval = train_test_split(X, y, test_size=0.5)
  X_train, X_test, y_train, y_test = train_test_split(X_dev, y_dev, test_size=0.5)

  # This is the neural net (MLPClassifier)
  # num_hidden_layers default = 2
  #clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(num_features, num_hidden_layers), random_state=1, max_iter=10000)
  clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=num_hidden_layers, random_state=1, max_iter=10000)


  clf.fit(X_train, y_train)

  ##############################################################################
  def compare_train_test(clf, X_train, y_train, X_test, y_test, bins=30):
    decisions = []
    for X, y in ((X_train, y_train), (X_test, y_test)):
      d1 = clf.predict_proba(X[y > 0.5])[:, 1]
      d2 = clf.predict_proba(X[y < 0.5])[:, 1]
      decisions += [d1, d2]

    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low, high)

    if wantplots == True:
      plt.figure(figsize=(12, 6))
      plt.hist(decisions[0],
               color='r', alpha=0.5, range=low_high, bins=bins,
               histtype='stepfilled',
               label='S (train)')
      plt.hist(decisions[1],
               color='b', alpha=0.5, range=low_high, bins=bins,
               histtype='stepfilled',
               label='B (train)')

      hist, bins = np.histogram(decisions[2],
                                bins=bins, range=low_high)
      scale = len(decisions[2]) / sum(hist)
      err = np.sqrt(hist * scale) / scale

      width = (bins[1] - bins[0])
      center = (bins[:-1] + bins[1:]) / 2

      plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')

      hist, bins = np.histogram(decisions[3],
                                bins=bins, range=low_high)
      scale = len(decisions[2]) / sum(hist)
      err = np.sqrt(hist * scale) / scale

      plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')

      plt.xlabel("Classifer output")
      plt.ylabel("Arbitrary units")
      plt.legend(loc='best')

  compare_train_test(clf, X_train, y_train, X_test, y_test, bins=100)

  ##############################################################################

  decisions = clf.predict_proba(X_test)[:, 1]
  # Compute ROC curve and area under the curve
  fpr, tpr, thresholds = roc_curve(y_test, decisions)
  roc_auc = auc(fpr, tpr)

  if wantplots == True:
    plt.figure()
    plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' % (roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

  return clf.coefs_, clf.intercepts_,roc_auc

  # How to Call:
  # weights,biases = neuralnet(num_hidden_layers= 2, wantplots=True)


################################################################################
# def plot_neural_net_weights_and_biases
## CHANGES from original code:
def draw_network(biases,weights, figsize=(12,8), colormap= plt.cm.viridis ,ax=None):

  if ax is None:
      plt.figure(figsize=figsize)

  markersize = 20  # Markersize
  linewidth = 3

  wnew = weights[1:len(weights)]

  # Find the greatest number of nodes
  nlayers = []
  for i in biases:
    nlayers.append(len(i))

  max_nodes = max(nlayers)
  #print('max_nodes', max_nodes)

  max_lo = 1
  max_hi = max_lo + max_nodes
  max_mid = max_lo + (max_nodes / 2)

  x_coords = []
  y_coords = []

##


  # Plot the nodes
  for i, bb in enumerate(biases):
    n = len(bb)  # Number of nodes
    lo = max_mid - (n / 2)

    xval = []
    yval = []
    for j, y in enumerate(bb):
      #color = plt.cm.rainbow(y)
      # Normalize the values to be between 0 and 1, even though they go from -1 to 1
      y = (y+1)/2
      #color = plt.cm.viridis(y)
      color = colormap(y)
      plt.plot([i], [j + lo], 'o', color=color, markersize=markersize)
      xval.append(i)
      yval.append(j + lo)
    x_coords.append(xval)
    y_coords.append(yval)

  # Plot the weights
  nweights = len(wnew)
  for i in range(0, nweights):
    w1 = wnew[i]
    for j in range(0, len(w1)):
      w2 = w1[j]
      for k in range(0, len(w2)):
        x1 = x_coords[i][j]
        y1 = y_coords[i][j]
        x2 = x_coords[i + 1][k]
        y2 = y_coords[i + 1][k]

        weight = w2[k]
        # print(weight)

        # Normalize the values to be between 0 and 1, even though they go from -1 to 1
        weight = (weight+1)/2
        color = colormap(weight)
        #color = plt.cm.viridis(weight)
        # print(x1,x2,y1,y2)
        plt.plot([x1, x2], [y1, y2], '-', color=color, linewidth=linewidth)

  # Plot the nodes again to cover up the lines
  for i, b in enumerate(biases):
    n = len(b)  # Number of nodes
    lo = max_mid - (n / 2)

    xval = []
    yval = []
    for j, y in enumerate(b):
        #color = plt.cm.rainbow(y)
      # Normalize the values to be between 0 and 1, even though they go from -1 to 1
      y = (y+1)/2
      #color = plt.cm.viridis(y)
      color = colormap(y)
      plt.plot([i], [j + lo], 'o', color=color, markersize=markersize)
      xval.append(i)
      yval.append(j + lo)
    x_coords.append(xval)
    y_coords.append(yval)


################################################################################
# def plot_diagnostics
# ROC
################################################################################

################################################################################



## Troubleshooting
#  when after pull if error "the files do not belong to the project"
#  File- repair IDE- rescan project indexes- reopen project
