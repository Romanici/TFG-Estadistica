#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 19:38:35 2020

@author: agusferfel
"""




def grafico(space, target, fitxer=''):
    
  import matplotlib.pyplot as plt  
  
  fig, (ax1, ax2) = plt.subplots( nrows=1,ncols=2,figsize=(18, 10) )

  ax1.scatter( space[:,0], space[:,1], c= target, cmap=plt.cm.get_cmap('jet', 10), s=2 ,alpha=0.3 ) 
  ax1.axis('off')

  ax2.scatter( space[:,0], space[:,1],  c= "black", s=2 ,alpha=0.3 ) 
  ax2.axis('off')

  fig.savefig( fitxer )
  plt.show()
  
  
def grafico_comp(space1, space2, target, fitxer=''):
    
  import matplotlib.pyplot as plt  
  
  fig, (ax1, ax2) = plt.subplots( nrows=1,ncols=2,figsize=(18, 10) )

  ax1.scatter( space1[:,0], space1[:,1], c= target, cmap=plt.cm.get_cmap('jet', 10), s=2 ,alpha=0.3 ) 
  ax1.axis('off')
  ax1.set_title("Pretained CAE", fontsize=14)

  ax2.scatter( space2[:,0], space2[:,1],  c= target, cmap=plt.cm.get_cmap('jet', 10), s=2 ,alpha=0.3 ) 
  ax2.axis('off')
  ax2.set_title("DCEC", fontsize=14)

  fig.savefig( fitxer )
  plt.show()  
  
  
def plot_training( history_NN, min_epoch ):
  max_epoch = len( history_NN.history["val_loss"] )
  val_loss = history_NN.history["val_loss"][min_epoch:max_epoch]
  train_loss = history_NN.history["loss"][min_epoch:max_epoch]
  epochs = range(min_epoch, max_epoch)

  plt.plot( epochs , val_loss, 'b', label='Validation loss') 
  plt.plot( epochs , train_loss, 'b+', label='Training loss') 
  plt.title('Training and validation loss') 
  plt.xlabel('Epochs')
  plt.ylabel('Binary CE')
  plt.legend()

  plt.savefig("training_plot")
  plt.show()
  
  
  
def load_fashion_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import fashion_mnist
    import numpy as np
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape(-1, 28, 28, 1).astype('float32')
    x = x/255.
    print("fashion_mnist; x:", x.shape)
    print("fashion_mnist; y:", y.shape)
    
    label_dictionnary = {0:'T-shirt/top', 1:'Trouser', 2:'Pullover', 
                     3:'Dress', 4:'Coat', 5:'Sandal', 6:'Shirt', 
                     7:'Sneaker', 8:'Bag', 9:'Ankle boot' }
    noms = list( label_dictionnary.values() )
    def true_label(x):
        return label_dictionnary[x]
    target = np.vectorize(true_label)(y)

    return x, y, target
  
    
  
    
  

def metrics(y, y_pred):

   import numpy as np
   from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

   nmi = normalized_mutual_info_score(y, y_pred)
   ari = adjusted_rand_score(y, y_pred)

   from coclust.evaluation.external import accuracy
   acc = accuracy(y, y_pred)

   return acc, nmi, ari




def CAE():  
    
    #%tensorflow_version 2.x
    from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, LeakyReLU, Flatten, Reshape
    from tensorflow.keras.models import Model
    from tensorflow.keras import backend 
    
    input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

    x = Conv2D(26, (3, 3), padding='same')(input_img)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(22, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(18, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(14, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x =Conv2D(10, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Reshape( (10,), name ="embedding")(x)
    
    ###
    
    x = Reshape( (1,1,10) )(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(10, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    #x = UpSampling2D((2, 2))(x) # WTF=?¿
    x = Conv2D(14, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(18, (2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = UpSampling2D((3, 3))(x)
    x = Conv2D(22, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(26, (3, 3), strides=(1, 1), padding='valid', activation="relu")(x)
    decoded = Conv2DTranspose(1, (3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(x)

    CAE = Model(input_img, decoded)

    CAE.summary()
        
    return(CAE)    
    



    


def CAE_Guo(input_shape=(28, 28, 1), filters=[32, 64, 128, 10]):
    
    from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
    from keras.models import Sequential, Model
    from keras.utils.vis_utils import plot_model
    import numpy as np    
    
    model = Sequential()
    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    model.add(Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape))

    model.add(Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2'))

    model.add(Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3'))

    model.add(Flatten())
    model.add(Dense(units=filters[3], name='embedding'))
    model.add(Dense(units=filters[2]*int(input_shape[0]/8)*int(input_shape[0]/8), activation='relu'))

    model.add(Reshape((int(input_shape[0]/8), int(input_shape[0]/8), filters[2])))
    model.add(Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3'))

    model.add(Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2'))

    model.add(Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1'))
    model.summary()
    return model    
    
    
    
    
  
    
    