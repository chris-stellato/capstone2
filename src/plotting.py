import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    #change cmap
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
        print('Normalized confusion matrix')
    else: 
        print('Confusion matrix without normalization')
        
    print(cm)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i,j],
                 horizontalalignment='center', 
                 color='white' if cm[i,j] > thresh else 'black')
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
    
    
    
    
def showImages(images_arr, labels_arr):
    fig, axes = plt.subplots(2,5, figsize=(10,10))
    axes = axes.flatten()
    for img, ax, label in zip(images_arr, axes, labels_arr):
        if label[0] == 1:
            label = 'mask'
        else: 
            label = 'no_mask'
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(label)
    plt.tight_layout()
    plt.show()

    
    
def show_single(image_path, label):
    if label == 1:
        label = 'mask'
    else: 
        label = 'no mask'
        
    fig, ax = plt.subplots(figsize=(10,10))
    image = plt.imread(image_path)
    ax.imshow(image)
    ax.axis('off')
    ax.set_title(label)
    plt.tight_layout()
    plt.show()