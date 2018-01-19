import numpy as np
import itertools
import math



#shifts the subjets towards the 1st PFJet based on the highest pt subjet
#eta'_i = eta_i - eta_0
#phi'_i = phi_i - phi_0
def translate(etas, phis, pts):
    xShift = []
    yShift = []
    
    #center based on 1st hardest subPFJet
    index = np.argmax(pts)
    xCenter = etas[index]
    yCenter = phis[index]
    
    #etas and phis from subjets from 1st PFJet
    for ix, iy in zip(etas, phis):
        xShift.append( ix - xCenter )
        yShift.append( iy - yCenter )
    
    return np.array(xShift), np.array(yShift)
    
    
    
    
    
#rotation not applied directly on eta-phi plane to preserve Lorentz invariances
#the rotation is likely done around px
def rotate(etas, phis, pts):
    #finds a second hardest subPFJet to rotate towards the 1st subPFjet
    index = np.argmax(pts)
    xCenter = etas[index]
    yCenter = phis[index]
    maxPt = -1
    v = 0
    for ix, iy, ip in zip(etas, phis, pts):
        dv = np.matrix([[ix],[iy]])-np.matrix([[xCenter],[yCenter]])
        dR = np.linalg.norm(dv)
        if(dR > 0.2 and ip > maxPt):
            maxPt = ip
            py = ip*math.sin(iy)
            pz = ip*math.sinh(ix)
            #computes the rotation angle
            v = np.arctan2(py,pz) + np.radians(90)

    rot_x = []
    rot_y = []
    #creates a rotation matrix
    c, s = np.cos(v), np.sin(v)
    R = np.matrix('{} {}; {} {}'.format(c, -s, s, c))

    for ix, iy, iw in zip(etas, phis, pts):
        #original components
        py = iw*math.sin(iy)
        pz = iw*math.sinh(ix)

        #transforms components
        rot = R*np.matrix([[py],[pz]])
        rix, riy = np.arcsinh(rot[1,0]/iw), np.arcsin(rot[0,0]/iw)
        
        rot_x.append(rix)
        rot_y.append(riy)
        
    return np.array(rot_x), np.array(rot_y)






#accounts for assymetry on energy distribution
#keeps the jet average pt always in eta' positive range
def reflect(etas, pts):
    leftSum = 0
    rightSum = 0
    for ix, iw in zip(etas, pts):
        if ix > 0: 
            rightSum += iw
        elif ix < 0:
            leftSum += iw
                        
    if(leftSum > rightSum):
        ref_x = [e*(-1) for e in etas]
        return np.array(ref_x)
    else:
        return np.array(etas)
    
    
    
    
    
    
#nx size of image in eta
#ny size of image in phi
def prepareImages(jets, nx, xmin, xmax, ny, ymin, ymax, pre_process):
    njets = len(jets)
    
    list_x = []
    list_y = []
    list_w = []

    #determines the pixel of picture
    xbins = np.linspace(xmin,xmax,nx+1)
    ybins = np.linspace(ymin,ymax,ny+1)
    
    #creates an empty 'th2'
    jet_images = np.zeros((njets, 1, nx, ny))

    #loop over jets
    for ijet in range(njets):
        #get vars
        pts = jets[ijet][1]
        etas = jets[ijet][2]
        phis = jets[ijet][3]
        energies = jets[ijet][4]
        charges = jets[ijet][5]
        pdgIds = jets[ijet][6]
        
        #shifts the jet image to the center in (eta,phi) = (0,0)
        if(pre_process[0]):
            etas, phis = translate(etas, phis, pts)

        #applies the rotation processing
        if(pre_process[1]):
            etas, phis = rotate(etas, phis, pts)
                        
        #applies the reflection processing
        if(pre_process[2]):
            etas = reflect(etas, pts)

        #applies normalization
        if(pre_process[3]):
            sumPt = sum(pts)
            pts = [ip/sumPt for ip in pts]

        x = etas
        y = phis
        weights = pts
        list_x.append(x)
        list_y.append(y)
        list_w.append(weights)
        hist, xedges, yedges = np.histogram2d(x, y, weights=weights, bins=(xbins, ybins))
        for ix in range(0,nx):
            for iy in range(0,ny):
                jet_images[ijet,0,ix,iy] = hist[ix,iy]

                                        
    return jet_images, list_x, list_y, list_w, xbins, ybins
