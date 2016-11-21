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

####################### Infos pour test ###########

initial_directory ='Test/'
csv_filename = "\\name.csv"
imNumber = 'florian.jpg'

if len(sys.argv) > 1:
    imName = sys.argv[1]
else:
    imName = initial_directory + imNumber

if len(sys.argv) > 2:
    chemin_base = sys.argv[2]
else:
    chemin_base = "E:\Documents\PROJET942 Reconnaissance Faciale\Base_jpg"
    
if len(sys.argv) > 3:
    final_name = sys.argv[3] + '/' + imName
else:
    final_name = 'Result/' + imName

#Base globale à tout le monde
finalx = 92
finaly = 112
marge = 0.075

##################################################

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
        plt.setp ( ax0.get_xticklabels () , visible = False )
        plt.setp ( ax0.get_yticklabels () , visible = False )
        if len ( sptitles ) == len ( images ):
            plt.title ("%s #%s" % ( sptitle , str ( sptitles[i])), create_font ('Tahoma',10))
        else :
            plt.title ("%s #%d" % ( sptitle , (i +1)), create_font ('Tahoma',10))
        plt.imshow (np.asarray ( images [i]) , cmap = colormap )
    if filename is None :
        plt.show()
    else :
        fig.savefig( filename )

#def detecte_visages(image, image_out, show = False):
#    # on charge l'image en mémoire
#    img = cv2.imread(image)
#    # on charge le modèle de détection des visages
#    face_model = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
#     
#     
#    # détection du ou des visages
#    faces = face_model.detectMultiScale(img)
#     
#    # on place un cadre autour des visages
#    print ("nombre de visages", len(faces), "dimension de l'image", img.shape, "image", image)
#    for face in faces:
#        cv2.rectangle(img, (face[0], face[1]), (face[0] + face[2], face[0] + face[3]), (255, 0, 0), 3)
#         
#    # on sauvegarde le résultat final
#    cv2.imwrite(image_out, img)
#     
#    # pour voir l'image, presser ESC pour sortir
#    if show :
#        cv2.imshow("visage",img)
#        if cv2.waitKey(5000) == 27: cv2.destroyWindow("visage")
     
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
            
##############################

#### VISAGE OBLIGATOIREMENT CENTREE #########################################

# Lecture image initiale
im = io.imread(imName)
height, width, depth = im.shape

print "height, width", height, width

im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

ratio = float(width)/float(height)
ratio_f = float(finalx) / float(finaly)

# Affichage profil intensite colonne milieu 1ere composante

x0_tab = []
y0_tab = []

for i in range (int(width * marge), int(width - width * marge), 10):
    x0_tab.append(i)

seuil_verti = []
seuil_horiz = []

x1_tab = []
x2_tab = []
y1_tab = []

x1_tab_rec = []
x2_tab_rec = []
y1_tab_rec = []

###################### SEUIL VERTICAL ######################

for i in range(len(x0_tab)):
    profil_vert = im[:,x0_tab[i]]
    seuil_verti.append(0)
    for pixelY in range(0, height):
        seuil_verti[i] = seuil_verti[i] + profil_vert[pixelY]
        
    seuil_verti[i] = float(seuil_verti[i]) / float(height)
    #print "Seuil vertical:", seuil_verti[i]
    
    y1_tab.append(0)
    while profil_vert[y1_tab[i]] > seuil_verti[i]:
        y1_tab[i] = y1_tab[i] + 1
        
    if ((y1_tab[i] - y1_tab[i-1])/(x0_tab[i] - x0_tab[i-1]) == 0) & (y1_tab[i] > height * marge):
        y1_tab_rec.append(y1_tab[i])
        #print "y1", y1_tab[i]

y1 = min(y1_tab_rec)

for i in range (y1, int(height - height * 3 * marge), 10):
    y0_tab.append(i)

###################### SEUIL HORIZONTAL ######################
    
for i in range(len(y0_tab)):
    profil_hori = im[y0_tab[i],:]
    seuil_horiz.append(0)
    for pixelX in range(0, width):
        seuil_horiz[i] = seuil_horiz[i] + profil_hori[pixelX]
        
    seuil_horiz[i] = float(seuil_horiz[i])/ float(width)
    #print "Seuil horizontal:", seuil_horiz[i]
    
    x1_tab.append(0)
    while profil_hori[x1_tab[i]] > seuil_horiz[i]:
        x1_tab[i] = x1_tab[i] + 1
        
    x2_tab.append(width - 1)
    while profil_hori[x2_tab[i]] > seuil_horiz[i]:
        x2_tab[i] = x2_tab[i] - 1
        
    if ((x1_tab[i] - x1_tab[i-1])/(y0_tab[i] - y0_tab[i-1]) == 0) & (x1_tab[i] > width * marge) & (i < haut):
        x1_tab_rec.append(x1_tab[i])
        #print "x1     ", x1_tab[i]
        
    if ((x2_tab[i] - x2_tab[i-1])/(y0_tab[i] - y0_tab[i-1]) == 0) & (x2_tab[i] < width - (width * marge)):
        x2_tab_rec.append(x2_tab[i])
        #print "x2          ", x2_tab[i]
        
#plt.plot(seuil_horiz)
#plt.ylabel('seuil_horiz')
#plt.show()
#
#plt.plot(seuil_verti)
#plt.ylabel('seuil_verti')
#plt.show()

print width * marge, width - width * marge, height * marge

plt.plot(x2_tab)
plt.ylabel('x2')
plt.show()

plt.plot(x1_tab)
plt.ylabel('x1')
plt.show()

plt.plot(y1_tab)
plt.ylabel('y1')
plt.show()

somme = 0
for i in range(len(x1_tab_rec)):
    somme = somme + x1_tab_rec[i]

moy_x1 = (somme - somme * marge) / len(x1_tab_rec)

for i in range(len(x1_tab_rec)):
    if x1_tab_rec[i] < moy_x1 :
        x1_tab_rec[i] = moy_x1

somme = 0
for i in range(len(x2_tab_rec)):
    somme = somme + x2_tab_rec[i]

moy_x2 = (somme + somme * marge) / len(x2_tab_rec)

for i in range(len(x2_tab_rec)):
    if x2_tab_rec[i] > moy_x2 :
        x2_tab_rec[i] = moy_x2

print moy_x1, moy_x2

x1 = min(x1_tab_rec)
x2 = max(x2_tab_rec)

y2 = ((x2 - x1) / ratio_f) + y1

#ratio_inter = float(x2-x1)/float(y2-y1)

print "x1", x1
print "x2", x2
print "y1", y1
print "y2", y2

if x2 > width :
    print ""
    print "UNESPECTED ERROR, l'image peut être déformé : x2 out of bound =", x2
    print "New x2 value =", (width -1)
    print ""
    x2 = width - 1
    y2 = ((x2 - x1) / ratio_f) + y1

if y2 > height :
    print ""
    print "SCALING ERROR BY RATIO, l'image peut être déformé : y2 out of bound =", y2
    print "New y2 value =", (height -1)
    print ""
    y2 = height -1
    x2 = ((y2 - y1) * ratio_f) + x1

#lignes

#fig=plt.figure("profil vertical")
#plt.plot(profil_vert)
#plt.ylabel('profil vertical')
#plt.show()
#
#fig=plt.figure("profil horizontal")
#plt.plot(profil_hori)
#plt.ylabel('profil horizontal')
#plt.show()

im_cropped_a = im[y1:y2, x1:x2]
im_resize_a = cv2.resize(im_cropped_a, (finalx, finaly), interpolation = cv2.INTER_CUBIC)
fig=plt.figure("image")
io.imshow(im_resize_a)
io.show()
cv2.imwrite(final_name, im_resize_a)

##############################          
#-----------------------------
# Programme pcp        
# append tinyfacerec to module search path
sys.path.append ("..")

# 1 - read images ATTENTION, chemin a changer - ATTENTION au \\
[X,y] = read_images (chemin_base)
#n = 0
##cv2.imshow('image+str(n)',X[0])
#titre = "image "+ str(n)
#fig=plt.figure(titre)
#io.imshow(X[0])
#io.show()

# perform a full pca
[D, W, mu] = pca ( asRowMatrix(X), y)

# 2 images ( note : eigenvectors are stored by column !)
E = []
for i in xrange ( min( len (X), 16)):
    e = W[:,i].reshape(X [0].shape)
    E.append( normalize (e ,0 ,255) )
# plot them and store the plot to " python_eigenfaces.pdf"
#subplot ( title =" Eigenfaces AT&T Facedatabase ", images = E, rows =4, cols =4, sptitle =" Eigenface", colormap =cm.jet , filename ="python_pca_eigenfaces.png")

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

ouverture = open(chemin_base + csv_filename, "rb")

csvFileName = csv.reader(ouverture)
name = []
for row in csvFileName:
    name = row

ouverture.close()

#name = ["adrien","enzo","florian","najwa","pierre"]
final_name = name[model.predict(test)]
print final_name