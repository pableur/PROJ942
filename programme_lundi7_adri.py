# coding: utf8

#programme test

import sys
import os
import csv
# import numpy and matplotlib colormaps
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from skimage import io
import matplotlib.cm as cm

import cv2

############### FONCTIONS FACE_RECOGNITION ###############

def read_images (path , sz= None ):
    c = 0
    X,y = [], []
    for dirname, dirnames, filenames in os.walk ( path ):
        for subdirname in sorted(dirnames) :
            subject_path = os.path.join(dirname , subdirname )
            for filename in sorted(os.listdir(subject_path)):
                try :
                    im = Image.open(os.path.join (subject_path , filename ))
                    im = im.convert ("L")
                    # resize to given size (if given )
                    if (sz is not None ):
                        im = im.resize(sz , Image.ANTIALIAS )
                    X.append (np.asarray (im , dtype =np.uint8 ))
                    y.append (c)
                except IOError :
                    print "I/O error ({0}) : {1} ".format(errno , strerror )
                except :
                    print " Unexpected error :", sys.exc_info() [0]
                    raise
            c = c+1
    return [X,y]
    
def asRowMatrix (X):
    if len (X) == 0:
        return np.array([])
    mat = np.empty((0 , X [0].size), dtype=X [0].dtype )
    for row in X:
        mat = np.vstack((mat,np.asarray(row).reshape(1,-1)))
    return mat
    
def asColumnMatrix (X):
    if len (X) == 0:
        return np.array ([])
    mat = np.empty ((X [0].size , 0) , dtype =X [0].dtype )
    for col in X:
        mat = np.hstack (( mat , np.asarray ( col ).reshape( -1 ,1)))
    return mat  
    
def pca(X, y, num_components =0):
    [n,d] = X.shape
    if ( num_components <= 0) or ( num_components > n):
        num_components = n
    mu = X.mean ( axis =0)
    X = X - mu
    if n>d:
        C = np.dot (X.T,X)
        [ eigenvalues , eigenvectors ] = np.linalg.eigh (C)
    else :
        C = np.dot (X,X.T)
        [ eigenvalues , eigenvectors ] = np.linalg.eigh (C)
        eigenvectors = np.dot (X.T, eigenvectors )
        for i in xrange (n):
            eigenvectors [:,i] = eigenvectors [:,i]/ np.linalg.norm ( eigenvectors [:,i])
    # or simply perform an economy size decomposition
    # eigenvectors , eigenvalues , variance = np.linalg.svd (X.T, full_matrices = False )
    # sort eigenvectors descending by their eigenvalue
    idx = np.argsort (- eigenvalues )
    eigenvalues = eigenvalues [idx ]
    eigenvectors = eigenvectors [:, idx ]
    # select only num_components
    eigenvalues = eigenvalues [0: num_components ].copy ()
    eigenvectors = eigenvectors [: ,0: num_components ].copy ()
    return [ eigenvalues , eigenvectors , mu]
    
def project (W, X, mu= None ):
    if mu is None :
        return np.dot (X,W)
    return np.dot (X - mu , W)
    
def reconstruct (W, Y, mu= None ):
    if mu is None :
        return np.dot(Y,W.T)
    return np.dot (Y,W.T) + mu
        
def normalize (X, low , high , dtype = None ):
    X = np.asarray (X)
    minX , maxX = np.min (X), np.max (X)
    # normalize to [0...1].
    X = X - float ( minX )
    X = X / float (( maxX - minX ))
    # scale to [ low...high ].
    X = X * (high - low )
    X = X + low
    if dtype is None :
        return np.asarray (X)
    return np.asarray (X, dtype = dtype )
    
def create_font ( fontname ='Tahoma', fontsize =10) :
    return { 'fontname': fontname , 'fontsize': fontsize }
    
def subplot (title , images , rows , cols , sptitle =" subplot ", sptitles =[] , colormap =cm.gray , ticks_visible =True , filename = None ):
    fig = plt.figure()
    # main title
    fig.text (.5 ,.95 , title , horizontalalignment ='center')
    for i in xrange (len( images )):
        ax0 = fig.add_subplot (rows ,cols ,(i +1) )
        '''
        plt.setp ( ax0.get_xticklabels () , visible = False )
        plt.setp ( ax0.get_yticklabels () , visible = False )
        if len ( sptitles ) == len ( images ):
            plt.title ("%s #%s" % ( sptitle , str ( sptitles[i])), create_font ('Tahoma',10))
        else :
            plt.title ("%s #%d" % ( sptitle , (i +1)), create_font ('Tahoma',10))
        plt.imshow (np.asarray ( images [i]) , cmap = colormap )
        '''
    if filename is None :
        plt.show()
    else :
        fig.savefig( filename )
        
class AbstractDistance ( object ):
    
    def __init__(self , name ):
            self._name = name
    def __call__(self ,p,q):
        raise NotImplementedError (" Every AbstractDistance must implement the __call__method.")
    @property
    def name ( self ):
        return self._name
    def __repr__( self ):
        return self._name
        
class EuclideanDistance ( AbstractDistance ): 
    def __init__( self ):
        AbstractDistance.__init__(self ," EuclideanDistance ")
    def __call__(self , p, q):
        p = np.asarray(p).flatten()
        q = np.asarray(q).flatten()
        return np.sqrt(np.sum (np.power((p-q) ,2)))
    
class CosineDistance ( AbstractDistance ):
    def __init__( self ):
        AbstractDistance.__init__(self ," CosineDistance ")
    def __call__(self , p, q):
        p = np.asarray (p).flatten ()
        q = np.asarray (q).flatten ()
        return -np.dot(p.T,q) / (np.sqrt (np.dot(p,p.T)*np.dot(q,q.T)))
  
#def lda (X, y, num_components =0) :
#    y = np.asarray (y)
#    [n,d] = X.shape
#    c = np.unique (y)
#    if ( num_components <= 0) or ( num_component >( len (c) -1)):
#        num_components = ( len (c) -1)
#    meanTotal = X.mean ( axis =0)
#    Sw = np.zeros ((d, d), dtype =np.float32 )
#    Sb = np.zeros ((d, d), dtype =np.float32 )
#    for i in c:
#        Xi = X[np.where (y==i) [0] ,:]
#        meanClass = Xi.mean ( axis =0)
#        Sw = Sw + np.dot ((Xi - meanClass ).T, (Xi - meanClass ))
#        Sb = Sb + n * np.dot (( meanClass - meanTotal ).T, ( meanClass - meanTotal ))
#    eigenvalues , eigenvectors = np.linalg.eig (np.linalg.inv (Sw)*Sb)
#    idx = np.argsort (- eigenvalues.real )
#    eigenvalues , eigenvectors = eigenvalues [idx], eigenvectors [:, idx ]
#    eigenvalues = np.array ( eigenvalues [0: num_components ].real , dtype =np.float32 , copy = True )
#    eigenvectors = np.array ( eigenvectors [0: ,0: num_components ].real , dtype =np.float32 , copy = True )
#    return [ eigenvalues , eigenvectors ]

class BaseModel ( object ):
    def __init__ (self , X=None , y=None , dist_metric = EuclideanDistance () , num_components=0) :
        self.dist_metric = dist_metric
        self.num_components = 0
        self.projections = []
        self.W = []
        self.mu = []
        if (X is not None ) and (y is not None ):
            self.compute (X,y)
            
    def compute (self , X, y):
        raise NotImplementedError (" Every BaseModel must implement the compute method.")
        
    def predict (self , X):
        minDist = np.finfo('float').max
        minClass = -1
        Q = project ( self.W, X.reshape (1 , -1) , self.mu)
        for i in xrange (len( self.projections )):
            dist = self.dist_metric ( self.projections [i], Q)
            if dist < minDist :
                minDist = dist
                minClass = self.y[i]
        return minClass
        
class EigenfacesModel ( BaseModel ):
    def __init__ (self , X=None , y=None , dist_metric = EuclideanDistance () , num_components=0) :
        super ( EigenfacesModel , self ).__init__ (X=X,y=y, dist_metric = dist_metric , num_components = num_components )
        
    def compute (self , X, y):
        [D, self.W, self.mu] = pca ( asRowMatrix (X),y, self.num_components )
        # store labels
        self.y = y
        # store projections
        for xi in X:
            self.projections.append ( project ( self.W, xi.reshape (1 , -1) , self.mu))
    
#def fisherfaces (X,y, num_components =0) :
#    y = np.asarray (y)
#    [n,d] = X.shape
#    c = len (np.unique (y))
#    [ eigenvalues_pca , eigenvectors_pca , mu_pca ] = pca (X, y, (n-c))
#    [ eigenvalues_lda , eigenvectors_lda ] = lda ( project ( eigenvectors_pca , X, mu_pca ), y, num_components )
#    eigenvectors = np.dot ( eigenvectors_pca , eigenvectors_lda )
#    return [ eigenvalues_lda , eigenvectors , mu_pca ]
    
#class FisherfacesModel ( BaseModel ):
#    def __init__(self , X=None , y=None , dist_metric = EuclideanDistance() , num_components=0) :
#        super ( FisherfacesModel , self ).__init__(X=X,y=y, dist_metric = dist_metric , num_components = num_components )
#    
#    def compute (self , X, y):
#        [D, self.W, self.mu] = fisherfaces ( asRowMatrix (X),y, self.num_components )
#        # store labels
#        self.y = y
#        # store projections
#        for xi in X:
#            self.projections.append ( project ( self.W, xi.reshape (1 , -1) , self.mu))
            


initial_directory ='Test/'
imNumber = 'adri3.jpg'

if len(sys.argv) > 1:
    imName = sys.argv[1]
else:
    #imName = initial_directory + imNumber
    imName="D:\\Virtual_Machines\\VM_proj942_original_2016_new\\data\\upload\\1479716984.3298\\1479716968820-75923793.jpg"

if len(sys.argv) > 2:
    chemin_base = sys.argv[2]
else:
    #chemin_base = "E:\Documents\PROJET942 Reconnaissance Faciale\Base_jpg"
    chemin_base = "D:\\Virtual_Machines\\VM_proj942_original_2016_new\\data\programme_v1\\Base_jpg"
	
if len(sys.argv) > 3:
    final_directory = sys.argv[3] + '/'
else:
    #final_directory = 'Result/'
    final_directory ="D:\\Virtual_Machines\\VM_proj942_original_2016_new\\data\\upload\\1479716984.3298\\"

#Base globale à tout le monde
finalx=92
finaly=112

#### VISAGE OBLIGATOIREMENT CENTREE #########################################

# Lecture image initiale
im = io.imread(imName)
height, width, depth = im.shape
print "dimenssion de l'image (height, width, depth) = ",
print height, width, depth
#fig=plt.figure("image initiale")
#io.imshow(im)
#io.show()
im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#fig=plt.figure("image en grise")
#io.imshow(im)
#io.show()

ratio = float(width)/float(height)
print "ratio=:", ratio

ratio_f = float(finalx) / float(finaly)
print "ratio_f=", ratio_f


# Affichage profil intensite colonne milieu 1ere composante

x0_2=[0,width]#[2*width/5,3*width/5]
y0_2=[0,height]#[2*height/5,3*height/5]
XdetectionStep=width/200
YdetectionStep=height/200
x1_2 = []
y1_2 = []
x2_2 = []
y2_2 = []

###################### detection du haut du crane ######################
coeff_de_variation=0.2;
y1=0
for i in range(y0_2[0],y0_2[1],YdetectionStep):
    profil_hori = im[i,:]
    
    ## calcul de la moyenne sur la ligne
    seuil_horiz=0
    for pixelX in profil_hori:
        seuil_horiz=seuil_horiz+pixelX
    seuil_horiz=float(seuil_horiz)/float(len(profil_hori)) 
    y1_2.append(seuil_horiz)

    #detection de variation
    for pixel in profil_hori:
        if pixel>seuil_horiz*(1+coeff_de_variation):
            y1=i
            break
        if pixel<seuil_horiz*(1-coeff_de_variation):
            y1=i
            break
    if y1!=0:
        break

###################### detection des cotés ######################
height_head=(height-y1)/4
coeff_de_variation=0.1;
x1=0
x2=width-1

## coté gauche
for i in range(x0_2[0],x0_2[1],XdetectionStep):
    profil_vert=[]
    for numPixel in range(len(im[:,i])):
        if numPixel>y1 and numPixel<(y1+height_head):
            profil_vert.append(im[:,i][numPixel])
    moyenne=0
    for pixel in profil_vert:
        moyenne=moyenne+pixel
    moyenne=float(moyenne)/float(len(profil_vert))
    
    #detection de variation
    for pixel in profil_vert:
        if pixel>moyenne*(1+coeff_de_variation):
            x1=i
            break
        if pixel<moyenne*(1-coeff_de_variation):
            x1=i
            break        
    if x1!=0:
        break
    
## coté droit
for i in range(x0_2[1]-1,x0_2[0],-1*XdetectionStep):

    profil_vert=[]
    for numPixel in range(len(im[:,i])):
        if numPixel>y1 and numPixel<(y1+height_head):
            profil_vert.append(im[:,i][numPixel])
    moyenne=0
    for pixel in profil_vert:
        moyenne=moyenne+pixel
    moyenne=float(moyenne)/float(len(profil_vert))
    
    #detection de variation
    for pixel in profil_vert:
        if pixel>moyenne*(1+coeff_de_variation):
            x2=i
            break
        if pixel<moyenne*(1-coeff_de_variation):
            x2=i
            break        
    if x2!=width-1:
        break   
#x1 = min(x1_2)
#x2 = max(x1_2)
x2_2=x1_2
y2 = ((x2 - x1) / ratio_f) + y1
if y2>=height:
    y2=height-1

print("#######################")
print "x1", x1
print "x2", x2
print "y1", y1
print "y2", y2
print("#######################")
print " "

if x2 > width:
    print ""
    print "UNESPECTED ERROR, l'image peut être déformé : x2 out of bound =", x2
    print "New x2 value =", (width -1)
    print ""
    x2 = width - 1
    y2 = ((x2 - x1) / ratio_f) + y1

if y2 > height:
    print ""
    print "SCALING ERROR BY RATIO, l'image peut être déformé : y2 out of bound =", y2
    print "New y2 value =", (height -1)
    print ""
    y2 = height -1
    x2 = ((y2 - y1) * ratio_f) + x1

# redimmensionnement de l'image
im_cropped_a=im[y1:y2, x1:x2]
im_resize_a=cv2.resize(im_cropped_a,(finalx,finaly),interpolation = cv2.INTER_CUBIC)

cv2.imwrite(final_directory + "image_recadre.jpg", im_resize_a)

##############################          
#-----------------------------
# Programme pcp        
# append tinyfacerec to module search path
sys.path.append ("..")

# 1 - read images ATTENTION, chemin a changer - ATTENTION au \\
[X,y] = read_images (chemin_base)

# perform a full pca
[D, W, mu] = pca ( asRowMatrix(X), y)

# 2 images ( note : eigenvectors are stored by column !)
E = []
for i in xrange ( min( len (X), 16)):
    e = W[:,i].reshape(X [0].shape)
    E.append( normalize (e ,0 ,255) )
# plot them and store the plot to " python_eigenfaces.pdf"
subplot ( title =" Eigenfaces AT&T Facedatabase ", images = E, rows =4, cols =4, sptitle =" Eigenface", colormap =cm.jet , filename =final_directory +"python_pca_eigenfaces.png")

# reconstruction steps
steps =[i for i in xrange(10 , min (len(X), 320) , 20)]
E = []
for i in xrange (min(len(steps),16)):
    numEvs = steps[i]
    P = project(W[:,0: numEvs],X[0].reshape(1,-1),mu)
    R = reconstruct(W[:,0:numEvs],P,mu)
    #reshape and append to plots
    R = R.reshape(X[0].shape)
    E.append( normalize (R ,0 ,255) )
# plot them and store the plot to " python_reconstruction.pdf "
#subplot ( title =" Reconstruction AT&T Facedatabase ", images =E, rows =4, cols =4, sptitle ="Eigenvectors", sptitles =steps , colormap =cm.gray , filename ="python_pca_reconstruction.png")

# Test 1
# get a prediction for any observation within the database
#n = 121     # class 13 - image 02
#test = X[n]
#X[n] = X[n+1]   # observation 'n+1' will be used twice in the model ! (not exactly correct but quicker solution)
## model computation
#model = EigenfacesModel (X , y)
#print " expected =", y[n], "/", "predicted =", model.predict(test)
#cv2.imshow('image initiale',test)

# Test 2
# get a prediction withe a noisy image
#im_resize_a = im_resize_a.convert ("L")

test = np.asarray (im_resize_a , dtype =np.uint8 )
# model computation
model = EigenfacesModel (X , y)

# chargement des noms depuis le fichier CSV
file=open(chemin_base + "//name.csv", "rb")
csvFileName= csv.reader(file)
name = []
for row in csvFileName:
    name = row
file.close()

#name = ["adrien","enzo","florian","najwa","pierre"]
final_name = name[model.predict(test)]

print "Prenom de la personne :"
print final_name