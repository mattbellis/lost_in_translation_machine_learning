import numpy as np
import matplotlib.pylab as plt


################################################################################
def sumfunc(dataset, dtype='normal', plot_range=(0,2)):               ## dataset = data_1(nentries,nfeatures)
  mysum = []

  nentries,nfeatures = dataset.shape

  if dtype=='squared':
    for i in range(nentries):
      s = sum(dataset[i]**2)
      mysum.append(s)
  else:
    for i in range(nentries):
      s = sum(dataset[i])
      mysum.append(s)

  plt.figure()
  plt.hist(mysum,bins=50,range=plot_range);
  #print(mysum)

################################################################################
def histfunc(dataset):   ## dataset = data_1(nentries,nfeatures)
  nentries,nfeatures = dataset.shape
  plt.figure(figsize=(16,3))
  for i in range(nfeatures):
    plt.subplot(1,nfeatures,i+1)
    plt.hist(dataset.T[i],bins=50,range=(0,1))

################################################################################
#random
def data_ran(nentries,nfeatures):
  data = np.random.random((nentries,nfeatures))
  return data

##
################################################################################
#############################################################################################
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
# gabby was here
################################################################################
# Should be all or mostly here
# https://colab.research.google.com/drive/1uNKwJaxleCwpp2pOeErTY8ZfPp9aj08E?usp=sharing
################################################################################
# def shuffle_dataset
################################################################################
# def plot_correlations
################################################################################
# def do_training
################################################################################
# def plot_diagnostics
# ROC
################################################################################
# def plot_neural_net_weights_and_biases
################################################################################

