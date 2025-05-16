####### latitude-longitude power plot 
####### for InGaP semiconductor 
####### written by Lucas Esson

# INSTRUCTIONS
# Once plotted (takes two and a half hours for resolution of 1)
# save figure as 'heatcopy.png'
# then open 'overlaycode.py' and run code
# this will plot the land map over the data plot



import math
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import integrate
from scipy.interpolate import CubicSpline
from PIL import Image




to_rad = math.pi/180





class solar():
        def __init__(self,day_of_year,lat,temp):         # indent all the arrays and self.each
                self.wvraw = [] #um     # this 'def' is within the class
                self.wv = []
                self.Horaw = []
                self.Ho = []
                self.n1 = 1
                self.n2 = []
                self.n2s_ = []
                self.n2f_ = []
                self.Rs_ = []
                self.Rs_abs_ = []
                self.Rp_ = []
                self.Rp_abs_ = []
                self.Reff_ = []
                self.Teff_ = []
                self.a_wraw = []
                self.a_w = []
                self.a_oraw = []
                self.a_o = []
                self.a_uraw = []
                self.a_u = []
                self.Tr_ = []
                self.Tw_ = []
                self.To_ = []
                self.Tu_ = []
                self.Id_ = []
                self.w_a_ = []
                self.Taa_ = []
                self.Cs_ = []
                self.Ir_ = []
                self.Tas_ = []
                self.Ia_ = []
                self.tau_ = []                                           
                self.Ta_ = []
                #Ta_ = 1
                # (alt transmittance functions)
                self.TrZ_ = []
                self.TwZ_ = []
                self.ToZ_ = []
                self.TaaZ_ = [] 
                self.TasZ_ = []
                self.rs_ = []
                self.Ig_ = []
                self.Is_ = []
                self.I_T_ = []
                self.Iglobal_ = []        # Iglobal in nm 
                self.wvn = []                           #wv in nm once global calculated
                self.Iwater_ = []

                self.wva = []                   # need wv of absorp data to interpolate with 
                self.abso = []
                self.wva3 = []  #coastal3
                self.abso3 = [] #coastal3
                self.wvaIII = []
                self.absoIII = []
                self.wvaII = []
                self.absoII = []
                self.wvaI = []
                self.absoI = []
                self.wvaIA = []
                self.absoIA = []
                self.wvaIB = []
                self.absoIB = []
                self.location = []
                self.j_3 = []    #coastal3
                self.j_III = []
                self.j_II = []
                self.j_I = []
                self.j_IA = []
                self.j_IB = []
                self.arr2 = []
                self.I_x_ = []

                self.d =day_of_year
                self.lat = lat

                self.lat = to_rad*self.lat                      # Convert to rad
                #hour = input("Hour of the day, 0 to 24: ")
                self.hour = 12

                # General
                #t = input("Solar Cell Tilt Angle, deg: ")
                self.t = 0
                self.t = self.t*(math.pi/180)
                #P = input("Measured Surface Pressure, mbar: ")                   # Measured Surface Pressure, 980 is a value taken from Met Office (Atlantic)
                self.P = 1013
                self.Po = 1013                                                                                             # Average Atmospheric Pressure at sea level, milli-bar
                self.rg = 0.2                                                                                                   # ground albedo, user input, but in pdf only 0.2 is the only rg value ever stated
                #B = input("Aerosol Optical Depth: ")                                           # 0.27 usually
                self.B = 0.27
                self.Temp = temp
                #self.x = input("Depth: ")
                #self.x = float(self.x)
                

                # Diode Calcs
                self.E_ = []
                self.flux_inc_ = []
                self.wv_eqe = []
                self.EQE_raw = []
                self.flux_abs_ = []
                self.V = []
                self.Idiode_ = []
                self.Vadj = []                                     # V adjusted to plot V where I < 0
                self.P_ = []
                self.Iabs_ = []
        def load_data(self):                            # should load data into the indexes/arrays
                with open("Ho.txt") as file:
                        for line in file:
                                s=line.rstrip().split()                  #(1,2,3) to ['1','2','3']
                                if len(s)==2:
                                        self.wvraw.append(float(s[0]))
                                        self.Horaw.append(float(s[1]))                                 
                with open("a_w.txt") as file:
                        for line in file:
                                s=line.rstrip().split()
                                if len(s)==2:
                                        self.a_wraw.append(float(s[1]))     
                with open("a_o.txt") as file:
                        for line in file:
                                s=line.rstrip().split()
                                if len(s)==2:
                                        self.a_oraw.append(float(s[1]))      
                with open("a_u.txt") as file:
                        for line in file:
                                s=line.rstrip().split()
                                if len(s)==2:
                                        self.a_uraw.append(float(s[1]))
                #replace below with desired locational data
                with open("southpacific.txt") as file:
                        for line in file:
                                s=line.rstrip().split()             #(1,2,3) to ['1','2','3']
                                if len(s)==2:
                                        self.wva.append(float(s[0]))
                                        self.abso.append(float(s[1]))
                with open("jerlov-III.txt") as file:
                        for line in file:
                                s=line.rstrip().split()             #(1,2,3) to ['1','2','3']
                                if len(s)==2:
                                        self.wvaIII.append(float(s[0]))
                                        self.absoIII.append(float(s[1]))
                with open("jerlov-II.txt") as file:
                        for line in file:
                                s=line.rstrip().split()             #(1,2,3) to ['1','2','3']
                                if len(s)==2:
                                        self.wvaII.append(float(s[0]))
                                        self.absoII.append(float(s[1]))
                with open("jerlov-I.txt") as file:
                        for line in file:
                                s=line.rstrip().split()             #(1,2,3) to ['1','2','3']
                                if len(s)==2:
                                        self.wvaI.append(float(s[0]))
                                        self.absoI.append(float(s[1]))             
                with open("jerlov-IA.txt") as file:
                        for line in file:
                                s=line.rstrip().split()             #(1,2,3) to ['1','2','3']
                                if len(s)==2:
                                        self.wvaIA.append(float(s[0]))
                                        self.absoIA.append(float(s[1]))
                with open("jerlov-IB.txt") as file:
                        for line in file:
                                s=line.rstrip().split()             #(1,2,3) to ['1','2','3']
                                if len(s)==2:
                                        self.wvaIB.append(float(s[0]))
                                        self.absoIB.append(float(s[1]))
                with open("jerlov-3.txt") as file:
                        for line in file:
                                s=line.rstrip().split()
                                if len(s)==2:
                                        self.wva3.append(float(s[0]))
                                        self.abso3.append(float(s[1]))
                with open("EQE_InGap.txt") as file:
                        for line in file:
                                s=line.rstrip().split()                  #(1,2,3) to ['1','2','3']
                                if len(s)==2:
                                        self.wv_eqe.append(float(s[0]))
                                        self.EQE_raw.append(float(s[1])/100)     # converts from % to 
                with open("V40.txt") as file:
                        for line in file:
                                s=line.rstrip().split()                  #(1,2,3) to ['1','2','3']
                                if len(s)==1:
                                        self.V.append(float(s[0]))                              
        ###########
        ### SSM ###
        ###########
        def Interp(self):
                #Ho and wv
                self.ho = CubicSpline(self.wvraw, self.Horaw)
                self.wv = np.arange(0.300, 2.001, 0.001)     # sets up better resolution of wv
                for i in range(0,len(self.wv)):
                        self.wv_ = self.wv[i]*1000
                        self.wvn.append(self.wv_)                # wvn is the wavelength data converted to nm
                self.Ho = self.ho(self.wv)
                # a_w
                self.aw = CubicSpline(self.wvraw, self.a_wraw)     
                self.a_w = (self.aw(self.wv))                           
                self.a_w[self.a_w<0] = 0                        # any -ve values =0
                # a_o
                self.ao = CubicSpline(self.wvraw, self.a_oraw)
                self.a_o = (self.ao(self.wv))                           
                self.a_o[self.a_o<0] = 0
                # a_u
                self.au = CubicSpline(self.wvraw, self.a_uraw)
                self.a_u = (self.au(self.wv))                           
                self.a_u[self.a_u<0] = 0
                ### water types
                # estonia
                self.location = np.interp(self.wvn, self.wva, self.abso)
                # jerlov types
                self.j_III = np.interp(self.wvn, self.wvaIII, self.absoIII)
                self.j_II = np.interp(self.wvn, self.wvaII, self.absoII)
                self.j_I = np.interp(self.wvn, self.wvaI, self.absoI)
                self.j_IA = np.interp(self.wvn, self.wvaIA, self.absoIA)
                self.j_IB = np.interp(self.wvn, self.wvaIB, self.absoIB)
                self.j_3 = np.interp(self.wvn, self.wva3, self.abso3)
                

        def Zenith(self):        # a calc section
                # Declination Angle                                              https://en.wikipedia.org/wiki/Position_of_the_Sun#Declination_of_the_Sun_as_seen_from_Earth
                self.dec = -23.44*math.cos(to_rad*((360/365)*(self.d + 10)))
                self.dec_rad = to_rad*self.dec
                # Hour Angle                                                            S.C. Bhatia, in Advanced Renewable Energy Systems, 2014
                self.hr_ang = (self.hour - 12)*15
                self.hr_ang = self.hr_ang*to_rad          # Convert to rad
                # cos(Z)
                self.cosz = math.cos(self.lat)*math.cos(self.dec_rad)*math.cos(self.hr_ang) + math.sin(self.dec_rad)*math.sin(self.lat)
                # Zenith Angle
                self.Z = math.acos(self.cosz)                     # cos^-1()
                if self.Z>1.561:
                    self.Z = 1.56
                self.Zdeg = self.Z*180/math.pi          # Just for reference so I can see Z in degrees 
                #print("Zdeg :", self.Zdeg)

        def DirectIrr(self):   # another calc section (you get the idea)
                # Correction Factor for Earth-Sun distance
                self.phi = (2*math.pi*(self.d-1)/365)     # phi = day angle, rad
                self.D = 1.00011 + 0.034221*math.cos(self.phi) + 0.00128*math.sin(self.phi) + 0.000719*math.cos(2*self.phi) + 0.000077*math.sin(2*self.phi)
                #D = 1

                # Tr:  Transmittance function of Rayleigh Scattering
                self.M = (math.cos(self.Z) + 0.15*(93.885 - self.Z)**-1.253)**-1                           # Air Mass
                self.Mcor = self.M*self.P/self.Po                                                                                                  # Pressure Corrected Air Mass

                for i in range(0,len(self.wv)):
                        self.Tr = math.exp(-self.Mcor/((self.wv[i]**4)*(115.6406 - (1.335/(self.wv[i]**2)))))      # Transmittance function of Rayleigh scattering
                        self.Tr_.append(self.Tr)

                # Ta:  Transmittance function of Aerosol Scattering and Absorption
                self.asc = 1.140                                                                                         # Single value for rural aerosol model
                for i in range(0,len(self.wv)):
                        self.tau = self.B*(self.wv[i]/0.5)**(-self.asc)                                    # 'turbidity' calc from B and asc inputs
                        self.tau_.append(self.tau)
                        self.Ta = math.exp(-self.M*self.tau_[i])                                                                   # Ta from using tau value
                        self.Ta_.append(self.Ta)

                # Tw:  Transimttance function of Water Vapour Absorption
                self.W = 1.42                                                                                                           # This should be a user input, I'll keep it here for now
                for i in range(0,len(self.wv)):
                        self.Tw = math.exp((-0.2385*(self.a_w[i])*self.W*self.M)/((1 + 20.07*self.a_w[i]*self.W*self.M)**0.45))
                        self.Tw_.append(self.Tw)

                # To:  Transmittance function of Ozone 
                self.O3 = 0.344                                                                 # 'typically 0.3 atm.cm'
                self.hozone = 22                                                                                                                 # height of max. ozone concentration (km)
                self.Mo = (1 + (self.hozone/6370))/(((math.cos(self.Z))**2) + 2*self.hozone/6370)**0.5    # Ozone Mass

                for i in range(0,len(self.wv)):
                        self.To = math.exp(-self.a_o[i]*self.O3*self.Mo)
                        self.To_.append(self.To)

                # Tu:  Transmittance function of Uniformly Mixed Gas Absorption

                for i in range(0,len(self.wv)):
                        self.Tu = math.exp(-1.41*self.a_u[i]*self.Mcor/(1 + 118.93*self.a_u[i]*self.Mcor)**0.45)
                        self.Tu_.append(self.Tu)

                # Direct Irradiance                     Combined T ends up still being around 0.999947, the only significant factor is cosZ (cos60 = 0.5)
                #Tcomb = Tr[i]*Ta*Tw[i]*To[i]*Tu[i]
                for i in range(0,len(self.wv)):
                        self.Id = self.D*(self.Ho[i]*self.Tr_[i]*self.Ta_[i]*self.Tw_[i]*self.To_[i]*self.Tu_[i])               #*math.cos(Z)   Why is this not applied? I get more or less the correct results w/o cosZ
                        self.Id_.append(self.Id)                                                                                # Direct Irradiance stored 

        def DiffuseIrr(self):
                ## Rayleigh Scattering Component
                self.w_400 = 0.945         
                self.w_vf = 0.095               # wavelength variation factor
                for i in range(0,len(self.wv)):
                        self.w_a = self.w_400*math.exp(-self.w_vf*(math.log(self.wv[i]/0.4))**2)        # aerosol single scattering albedo
                        self.w_a_.append(self.w_a)
                for i in range(0,len(self.wv)):
                        self.Taa = math.exp(-(1 - self.w_a_[i])*self.tau_[i]*self.M)                                    # aerosol absorption transmittance
                        self.Taa_.append(self.Taa)
                        #Taa_ = 1.0   
                for i in range(0,len(self.wv)):
                        if self.wv[i] < 0.451: self.Cs = (self.wv[i] + 0.55)**1.8
                        else: self.Cs = 1.0
                        self.Cs_.append(self.Cs)
                        #Cs_ = 1.0
                for i in range(0,len(self.wv)):
                        self.Ir = self.Ho[i]*self.D*math.cos(self.Z)*self.To_[i]*self.Tu_[i]*self.Tw_[i]*self.Taa_[i]*(1-self.Tr_[i]**0.45)*0.5*self.Cs_[i]  # Diffuse Irradiance from Rayleigh Scattering 
                        self.Ir_.append(self.Ir)
                ## Aerosol Scattering Component
                for i in range(0,len(self.wv)):
                        self.Tas = math.exp(-self.w_a_[i]*self.tau_[i]*self.M)                                            # aerosol scattering transmittance
                        self.Tas_.append(self.Tas)
                self.cos_ = 0.65                                  # aerosol asymmetry value   
                self.ALG = math.log(1 - self.cos_)
                self.AFS = self.ALG*(1.459 + self.ALG*(0.1595 + self.ALG*(0.4129)))
                self.BFS = self.ALG*(0.0783 + self.ALG*(-0.3824 - self.ALG*0.5874))
                self.Fs = 1 - 0.5*math.exp((self.AFS + self.BFS*math.cos(self.Z))*math.cos(self.Z))       # downward fraction of aerosol scatter
                for i in range(0,len(self.wv)):
                        self.Ia = self.Ho[i]*self.D*math.cos(self.Z)*self.To_[i]*self.Tu_[i]*self.Tw_[i]*self.Taa_[i]*(self.Tr_[i]**1.5)*(1 - self.Tas_[i])*self.Fs*self.Cs_[i] # Diffuse Irradiance from Aerosol Scattering
                        self.Ia_.append(self.Ia)
                ## Reflection Component
                #T values are transmittance values for M = 1.8
                for i in range(0,len(self.wv)):
                        self.TrZ = math.exp(-1.8/((self.wv[i]**4)*(115.6406 - (1.335/(self.wv[i]**2)))))
                        self.TrZ_.append(self.TrZ)
                        self.TwZ = math.exp(-0.2385*self.a_w[i]*self.W*1.8/(1 + 20.07*self.a_w[i]*self.W*1.8)**0.45)
                        self.TwZ_.append(self.TwZ)
                        self.ToZ = math.exp(-self.a_o[i]*self.O3*1.8)
                        self.ToZ_.append(self.ToZ)
                        self.TaaZ = math.exp(-(1 - self.w_a_[i])*self.tau_[i]*1.8)
                        self.TaaZ_.append(self.TaaZ)
                        self.TasZ = math.exp(-self.w_a_[i]*self.tau_[i]*1.8)
                        self.TasZ_.append(self.TasZ)
                        self.Fs_ = 1 - 0.5*math.exp((self.AFS + self.BFS/1.8)/1.8)
                        self.rs = self.ToZ_[i]*self.TwZ_[i]*self.TaaZ_[i]*(0.5*(1 - self.TrZ_[i]) + (1 - self.Fs_)*self.TrZ_[i]*(1 - self.TasZ_[i]))    # sky reflectivity
                        self.rs_.append(self.rs)        
                        self.Ig = (self.Id_[i]*math.cos(self.Z) + self.Ir_[i] + self.Ia_[i])*self.rs_[i]*self.rg*self.Cs_[i]/(1 - self.rs_[i]*self.rg)                  # Diffuse Irradiance from Reflectivity
                        self.Ig_.append(self.Ig)
                ## Diffuse Irradiance ##
                for i in range(0,len(self.wv)):
                        self.Is = self.Ir_[i] + self.Ia_[i] + self.Ig_[i]                  # Total Diffuse Irradiance
                        self.Is_.append(self.Is)

        def GlobalIrr(self):
                ### Global Irradiance ###
                for i in range(0,len(self.wv)):
                        self.I_T = self.Id_[i]*math.cos(self.Z) + self.Is_[i]              # Global Irradiance on Horizontal Surface
                        self.I_T_.append(self.I_T)
                        #print('Global Irradiance on Horizontal Surface =', I_T)
                self.theta = self.Z - self.t                                       # Angle of Incidence of Direct Sun Ray on Inclined Surface
                for i in range(0,len(self.wv)):
                        self.Iglobal = self.Id_[i]*math.cos(self.theta) + self.Is_[i]*((self.Id_[i]*math.cos(self.theta)/(self.Ho[i]*self.D*math.cos(self.Z))) + 0.5*(1 + math.cos(self.t))*(1 - self.Id_[i]/(self.Ho[i]*self.D))) + 0.5*self.I_T_[i]*self.rg*(1 - math.cos(self.t))   # Global Irradiance on Tilted Surface
                        self.Iglobal_.append(self.Iglobal*10**-3)        # w/adjustment of irrad from um-1 to nm-1                                 
                ## Allows me to easily see transmittance values to observe code behaviour
                self.T30 = list(['Tr =', self.Tr_[30], 'Ta =', self.Ta_[30], 'Tw =', self.Tw_[30], 'To =', self.To_[30], 'Tu =', self.Tu_[30], 'Tcomb =', self.Tr_[30]*self.Ta_[30]*self.Tw_[30]*self.To_[30]*self.Tu_[30]])
                ## Plot
                #fig, ax = plt.subplots()
                #ax.plot(self.wvn, self.Iglobal_)
                #ax.set(xlabel='Wavelength (nm)', ylabel='Global Irradiance (W m-2 nm-1)', title='Irradiance Spectra')
                #ax.grid()
                #fig.savefig("spectra.png")
                #plt.show()

        def RefIndex(self,salt_or_fresh):                                # this most definitely needs to be changed 
                # Seawater (salinity = 35%)      # is for 300-700nm and 0to30deg -> not accurate 
                self.a1 = -1.50156*10**-6
                self.b1 = 1.07085*10**-7
                self.c1 = -4.27594*10**-5
                self.d1 = -1.60476*10**-4
                self.e1 = 1.39807
                # Freshwater (salinity = 0%)
                self.a2 = -1.97812*10**-6
                self.b2 = 1.03223*10**-7
                self.c2 = -8.58123*10**-6
                self.d2 = -1.54834*10**-4
                self.e2 = 1.38919
                # Seawater n ->>> wv should be in nm!!! 
                for i in range(0,len(self.wvn)):
                        self.n2s = self.a1*self.Temp**2 + self.b1*self.wvn[i]**2 + self.c1*self.Temp + self.d1*self.wvn[i] + self.e1
                        self.n2s_.append(self.n2s)
                # Freshwater n
                for i in range(0,len(self.wvn)):
                        self.n2f = self.a2*self.Temp**2 + self.b2*self.wvn[i]**2 + self.c2*self.Temp + self.d2*self.wvn[i] + self.e2
                        self.n2f_.append(self.n2f)
                # Sea or Fresh
                self.choice = salt_or_fresh
                for i in range(0,len(self.wvn)):
                        if str(self.choice) == "Salt":
                                self.n2 = self.n2s_
                        if str(self.choice) == "Fresh":
                                self.n2 = self.n2f_

        def WaterTrans(self):
                #print(self.n1)
                #print(self.n2)
                for i in range(0,len(self.wvn)):
                        self.Rs = (self.n1*math.cos(self.Z) - self.n2[i]*(1 - ((self.n1/self.n2[i])*math.sin(self.Z))**2)**0.5)/(self.n1*math.cos(self.Z) + self.n2[i]*(1 - ((self.n1/self.n2[i])*math.sin(self.Z))**2)**0.5)
                        self.Rs_.append(self.Rs)
                        self.Rs_abs = (abs(self.Rs_[i]))**2
                        self.Rs_abs_.append(self.Rs_abs)
                for i in range(0,len(self.wvn)):
                        self.Rp = (self.n1*(1 - ((self.n1/self.n2[i])*math.sin(self.Z))**2)**0.5 - self.n2[i]*math.cos(self.Z))/(self.n1*(1 - ((self.n1/self.n2[i])*math.sin(self.Z))**2)**0.5 + self.n2[i]*math.cos(self.Z))
                        self.Rp_.append(self.Rp)
                        self.Rp_abs = (abs(self.Rp_[i]))**2
                        self.Rp_abs_.append(self.Rp_abs)
                for i in range(0,len(self.wvn)):
                        self.Reff = 0.5*(self.Rs_abs_[i] + self.Rp_abs_[i])
                        self.Reff_.append(self.Reff)
                        self.Teff = 1 - self.Reff_[i]
                        self.Teff_.append(self.Teff)
                for i in range(0,len(self.wvn)):
                        self.Iwater = self.Iglobal_[i]*self.Teff_[i]               # Irradiance that transmits through water boundary
                        self.Iwater_.append(self.Iwater)
                self.Isurf = scipy.integrate.trapezoid(self.Iglobal_, self.wvn)
                #print("Surface Power Density: ", self.Isurf)

        def ConditionalCoeff(self, lat, long):
                ##############################################
                ######                                 #######
                #### CONDITIONAL ATTENUATION COEFFICIENTS ####
                ######                                 #######
                ##############################################

# coastal layers at top, so that as you go further into ocean the layers
# plot ontop of the darker coastal layers 
                
                ####   PACIFIC   ####

                ## North Pacific
                # East Japan
                if (30 <= lat <= 45) and (135 <= long <= 150):
                        self.arr2 = self.j_IB
                else: self.arr2 = self.j_3
                if (10 <= lat <= 30) and (125 <= long <= 135):
                        self.arr2 = self.j_II
                
                # Rest of North Pac.
                if (30 <= lat <= 47.5) and (-135 <= long <= -127.5):
                        self.arr2 = self.j_III
                if (10 <= lat <= 40) and (-135 <= long <= -120):
                        self.arr2 = self.j_III
                if (30 <= lat <= 50) and (150 <= long <= 180):
                        self.arr2 = self.j_II
                if (30 <= lat <= 50) and (-180 <= long <= -135):
                        self.arr2 = self.j_II
                if (45 <= lat <= 50) and (135 <= long <= 150):
                        self.arr2 = self.j_II
                        
                ## Mid-North Pacific Band, Type I
                if (10 <=lat <= 20) and (-125 <= long <= -107.5):
                        self.arr2 = self.j_III
                if (7.5 <= lat <= 12.5) and (-125 <= long <= -110):
                        self.arr2 = self.j_III
                # mexico corner
                if (17.5 <= lat <= 25) and (-125 <= long <= -115):
                        self.arr2 = self.j_III
                if (10 <= lat <= 17.5) and (-125 <= long <= -110):
                        self.arr2 = self.j_II
                if (10 <= lat <= 30) and (-180 <= long <= -125): 
                        self.arr2 = self.j_I
                if (10 <= lat <= 30) and (135 <= long <= 180):
                        self.arr2 = self.j_I
                        
                ## Mid-South Pacific Band, Type IB
                if (-10 <= lat <= 10) and (135 <= long <= 180):
                        self.arr2 = self.j_IB
                if (-10 <= lat <=7.5) and (-105 <= long <= -87.5):
                        self.arr2 = self.j_III
                if (5 <= lat <= 12.5) and (-105 <= long <= -95):
                        self.arr2 = self.j_III
                if (-10 <= lat <= 5) and (-105 <= long <= -90):
                        self.arr2 = self.j_II
                if (10 <= lat <= 15) and (110 <= long <= 115):
                        self.arr2 = self.j_II
                if (-10 <= lat <= 10) and (-180 <= long <= -105):
                        self.arr2 = self.j_IB
                  
                ## Australia 
                # n.e australia
                if (-17.5 <= lat <= -10) and (150 <= long <= 157.5):
                        self.arr2 = self.j_II # clear coasts, but still a coast so dont wanna give any higher than II
                # n.w australia 
                if (-17.5 <= lat <= -10) and (110 <= long <=117.5):
                        self.arr2 = self.j_II
                # s.w australia
                if (-40 <= lat <= -37.5) and (110 <= long <= 135):
                        self.arr2 = self.j_II
                # s.e australia
                if (-40 <= lat <= -35) and (152.5 <= long <= 157.5):
                        self.arr2 = self.j_II
                # e. aus
                if (-40 <= lat <= -17.5) and (155 <= long <= 157.5):
                        self.arr2 = self.j_II
                ## South Pacific
                if (-45 <= lat <= -17.5) and (-80 <= long <= -75):
                        self.arr2 = self.j_III
                if (-45 <= lat <= -10) and (-90 <= long <= -80):
                        self.arr2 = self.j_IB
                if (-45 <= lat <= -10) and (-180 <= long <= -90):
                        self.arr2 = self.j_I
                if (-45 <= lat <= -10) and (157.5 <= long <= 180):
                        self.arr2 = self.j_I
                ## Antartic
                # whole ant. band assumed as III, apart from where stated otherwise 
                if (-60 <= lat <= -45) and (-180 <= long <= 180):
                        self.arr2 = self.j_III
                # (stated 'otherwise')
                if (-60 <= lat <= -45) and (165 <= long <= 180):
                        self.arr2 = self.j_II
                ## Indonesia/Phillipines
                if (-10 <= lat <= 10) and (100 <= long <= 135):
                        self.arr2 = self.j_IB
                #Malacca strait
                if (-5 <= lat <= 0) and (100 <= long <= 115):
                        self.arr2 = self.j_III

                        
                
                ####   INDIAN OCEAN   ####
                # Arabian Gulf
                if (5 <= lat <= 17.5) and (57.5 <= long <= 70):
                        self.arr2 = self.j_III
                if (5 <= lat <= 10) and (52.5 <= long <= 57.5):
                        self.arr2 = self.j_III
                if (17.5 <= lat <= 22.5) and (60 <= long <= 65):
                        self.arr2 = self.j_III
                if (5 <= lat <= 15) and (60 <= long <= 67.5):
                        self.arr2 = self.j_II
                # Bengal Gulf
                if (5 <= lat <= 15) and (82.5 <= long <= 92.5):
                        self.arr2 = self.j_III
                if (15 <= lat <= 17.5) and (85 <= long <= 90):
                        self.arr2 = self.j_III
                if (5 <= lat <= 12.5) and (85 <= long <= 90):
                        self.arr2 = self.j_II
                ## East Africa
                if (-10 <= lat <= 2.5) and (47.5 <= long <= 60):
                        self.arr2 = self.j_III
                if (-10 <= lat <= 0) and (50 <= long <= 60):
                        self.arr2 = self.j_II
                
                ## Indian 'band'
                if (0 <= lat <= 7.5) and (55 <= long <= 100):
                        self.arr2 = self.j_II
                if (-10 <= lat <= 2.5) and (60 <= long <= 100):
                        self.arr2 = self.j_IB
                
                #e. SA
                if (-32.5 <= lat <= -27.5) and (35 <= long <= 55):
                        self.arr2 = self.j_III
                if (-27.5 <= lat <= -22.5) and (52.5 <= long <= 55):
                        self.arr2 = self.j_III
                ## Indian Bloc
                if (-45 <= lat <= -10) and (42.5 <= long <= 110):
                        self.arr2 = self.j_II
                # underneath Australia
                if (-45 <= lat <= -40) and (110 <= long <= 157.5):
                        self.arr2 = self.j_II
                
                
                        



                ####   ATLANTIC OCEAN   ####

                ## North Sea
                if (60 <= lat <= 70) and (-10 <= long <= 5):
                        self.arr2 = self.j_III
                if (70 <= lat <= 80) and (-20 <= long <= 20):
                        self.arr2 = self.j_III
                ## North Atlantic
                # Sargasso Sea
                
                if (27.5 <= lat <= 32.5) and (-77.5 <= long <= -70):
                        self.arr2 = self.j_II #horiz
                if (27.5 <= lat <= 37.5) and (-72.5 <= long <= -70):
                        self.arr2 = self.j_II #vert
                if (15 <= lat <= 35) and (-70 <= long <= -25):
                        self.arr2 = self.j_IA
                # north of sargasso
                if (35 <= lat <= 37.5) and (-72.5 <= long <= -60):
                        self.arr2 = self.j_II #horiz
                if (35 <= lat <= 45) and (-62.5 <= long <= -60):
                        self.arr2 = self.j_II #vert
                if (35 <= lat <= 42.5) and (-60 <= long <= -25):
                        self.arr2 = self.j_IA
                # north north
                if (42.5 <= lat <= 45) and (-62.5 <= long <= -50):
                        self.arr2 = self.j_II #horiz
                if (42.5 <= lat <= 57.5) and (-50 <= long <= -12.5):
                        self.arr2 = self.j_II
                # north north north
                if (57.5 <= lat <= 62.5) and (-40 <= long <= -10):
                        self.arr2 = self.j_II
                # Carribean
                if (20 <= lat <= 27.5) and (-95 <= long <= -90):
                        self.arr2 = self.j_II
                if (15 <= lat <= 27.5) and (-90 <= long <= -60):
                        self.arr2 = self.j_IB
                if (5 <= lat <= 15) and (-75 <= long <= -60):
                        self.arr2 = self.j_IB
                # Key West - after carribean to ensure it overlays 
                if (25 <= lat <= 30) and (-85 <= long <= -80):
                        self.arr2 = self.j_3
                # Iberia/Africa
                # iberia
                if (40 <= lat <= 45) and (-25 <= long <= -12.5):
                        self.arr2 = self.j_II
                # iberia/africa crossover
                if (35 <= lat <= 40) and (-30 <= long <= -12.5):
                        self.arr2 = self.j_IB
                if (31 <= lat <= 35) and (-30 <= long <= -15):
                        self.arr2 = self.j_IB
                ## west africa III
                if (2.5 <= lat <= 30) and (-29 <= long <= -20):
                        self.arr2 = self.j_III
                if (2.5 <= lat <= 5) and (-20 <= long <= -17.5):
                        self.arr2 = self.j_III
                if (2.5 <= lat <= 30) and (-30 <= long <= -22.5):
                        self.arr2 = self.j_II
                if (2.5 <= lat <= 15) and (-40 <= long <= -30):
                        self.arr2 = self.j_IA #lil west african IA pocket
                if (5 <= lat <= 15) and (-50 <= long <= -40):
                        self.arr2 = self.j_II
                # north brazil
                if (0 <= lat <= 5) and (-52.5 <= long <= -32.5):
                        self.arr2 = self.j_III
                if (5 <= lat <= 15) and (-60 <= long <= -50):
                        self.arr2 = self.j_III
                # east brazil
                if (-37.5 <= long <= -32.5) and (-25 <= lat <= -10):
                        self.arr2 = self.j_III
                ## argentina
                # low
                if (-40 <= lat <= -37.5) and (-55 <= long <= -47.5):
                        self.arr2 = self.j_III #horiz
                if (-40 <= lat <= -32.5) and (-50 <= long <= -47.5):
                        self.arr2 = self.j_III #vert
                if (-45 <= lat <= -40) and (-60 <= long <= -32.5):
                        self.arr2 = self.j_II
                # mid
                if (-35 <= lat <= -22.5) and (-47.5 <= long <= -40):
                        self.arr2 = self.j_III #vert
                if (-40 <= lat <= -35) and (-47.5 <= long <= -32.5):
                        self.arr2 = self.j_II
                # high
                if (-25 <= lat <= -22.5) and (-42.5 <= long <= -32.5):
                        self.arr2 = self.j_III #horiz
                if (-35 <= lat <= -25) and (-40 <= long <= -32.5):
                        self.arr2 = self.j_II

                #w. SA
                if (-32.5 <= lat <= -7.5) and (5 <= long <= 10): #
                        self.arr2 = self.j_III
                if (-32.5 <= lat <= -30) and (10 <= long <= 12.5):
                        self.arr2 = self.j_III
                        
                # Atlantic Bloc -> my attempt at 'filling the oceans'
                if (-45 <= lat <= 2.5) and (-32.5 <= long <= 5):
                        self.arr2 = self.j_II
                
                
                #s. SA
                if (-45 <= lat <= -32.5) and (0 <= long <= 60):
                        self.arr2 = self.j_II
                
                    
                

                #self.arr2 = self.j_I  # interpolates so len(wvn) = len(abs)

        def IncidentPower(self, depth):
                for i in range(0,len(self.wvn)):
                        self.I_x = (self.Iwater_[i])*math.exp(-(self.arr2[i])*depth)    # models exponential absorption of irradiance w/ depth 
                        self.I_x_.append(self.I_x)

                self.IncidentP = scipy.integrate.trapezoid(self.I_x_, self.wvn)
                return self.IncidentP
                #print('Incident Power Density:', self.IncidentP)   # incident Power Density at chosen depth, location and time

        ###################     
        ### DIODE CALCS ###
        ###################
                
        def PhotonFlux(self):
                # Photon Energy(wvn)
                self.h = 6.62607015*10**-34               # Plancks const
                self.c = 3*10**8                                         # Speed of light
                for i in range(0,len(self.wvn)):
                        self.E = self.h*self.c/((10**-9)*self.wvn[i])
                        self.E_.append(self.E)            # Photon Energy(wvn)
                # Incident Photon Flux 
                for i in range(0,len(self.wvn)):
                        self.flux_inc = self.I_x_[i]/self.E_[i]         # incident flux of irradiance(x)
                        self.flux_inc_.append(self.flux_inc)
                # Absorbed Photon Flux
                self.EQE = np.interp(self.wvn, self.wv_eqe, self.EQE_raw)
                for i in range(0,len(self.wvn)):
                        self.flux_abs = self.EQE[i]*self.flux_inc_[i]
                        self.flux_abs_.append(self.flux_abs)
                # Number of photons absorbed by the solar cell
                self.photon_num = scipy.integrate.trapezoid(self.flux_abs_, self.wvn)
                self.q = 1.60217663*10**-19
                # Induced current density
                self.I_ph_den = self.photon_num*self.q
                return self.I_ph_den
                
        def CellSpecs(self):            # these specs would be different if i changed solar cell tech, type or manufacturer
                ###### m-Si ######
                self.Ar = 0.024336               # Cell area of m-Si, m^2 
                # ? should i now adjust I_ph by the factor it previously took to
                # change I_ph to graph Isc value on mSi graph ?  That's what I've done 
                self.f_adjust = 1.03592
                self.I_ph = self.I_ph_den*self.Ar*self.f_adjust                   # Modelled induced current and adjusted 
                self.Rsh = 10**6                                         # Shunt resistance
                self.I_dark = 10**-7.6                                 # Saturation current 
                self.n = 3                                          # Ideality factor 
                self.k = 1.380649*10**-23
                self.T = self.Temp + 273.15                        # Convert water temp 'Temp' into Kelvin
                
        def IVCurve(self):
                # Itot in diode
                for i in range(0, len(self.V)):
                        self.Idiode = - self.I_ph + self.I_dark*(math.exp(self.q*self.V[i]/(self.n*self.k*self.T))) + self.V[i]/self.Rsh
                        self.Idiode_.append(self.Idiode)
                        if self.Idiode > 0:
                                break
                for i in range(0, len(self.Idiode_)):
                        self.Vadj.append(self.V[i])     # records corresponding V to -ve I vals 
                # Plots IV curve 
                #fig, ax = plt.subplots()
                #ax.plot(self.Vadj, self.Idiode_)
                #ax.set(xlabel='V', ylabel='I', title='IV')
                #ax.grid()
                #plt.show()
                return self.Vadj
                
        def Pmax(self):
                for i in range(0, len(self.Idiode_)):
                        self.P = self.Vadj[i]*abs(self.Idiode_[i])
                        self.P_.append(self.P)
                self.PMAX = max(self.P_)
                self.Pden_out = (self.PMAX*1000)/(self.Ar*10000) # mW/cm^2 output
                if self.Pden_out < 0.7:
                        self.Pden_out = 0
                #print("Pmax =",(max(self.P_)))
                self.mp = self.P_.index(max(self.P_))  # mp is the index position of Pmax, mp denotes 'max power'
                # Imax and Vmax found using index of Pmax 
                self.Imp = self.Idiode_[self.mp]
                self.Vmp = self.Vadj[self.mp]
                #print("Imp =", self.Imp)                   
                #print("Vmp =", self.Vmp)
                # finding Voc for tuning purposes
                for i in range(0, len(self.Idiode_)):
                        self.Iabs = abs(self.Idiode_[i])
                        self.Iabs_.append(self.Iabs)
                self.Vocp = self.Iabs_.index(min(self.Iabs_))  # index position of Voc
                self.Voc = self.Vadj[self.Vocp]
                #print("Voc =", self.Voc)
                return self.PMAX
                return self.Imp
                return self.Vmp
                return self.Pden_out

                







# put instructions to trigger code




day_of_year = input("Day Number of the Year, 1 to 365: ")
if day_of_year=="":
        day_of_year = 79
        print("using default")
else:
        day_of_year = float(day_of_year)

#lat = input("Latitude, degrees: ")
#if lat=="":
#        lat = 0
#        print("using default")
#else:
#        lat = float(lat)

Temp = input("Water temp: ")
if Temp=="":
        Temp = 17
        print("using default")
else:
        Temp = float(Temp)

salt_or_fresh=input("Input Water Type: Fresh or Salt: ")
if salt_or_fresh=="":
        salt_or_fresh = "Salt"
        print("using default")

depth=input("Depth: ")
depth=float(depth)

res = input("Resolution: ")
res= float(res)

dataplot = []
lat = 90

while (lat>-90):
        build = []
        pdens = []
        pincid = []
        long = -180
        while(long<180):    
            a=solar(day_of_year,lat,Temp)                      # this is now a list of instructions which triggers each
            a.load_data()
            a.Interp()
            a.Zenith()
            a.DirectIrr()
            a.DiffuseIrr()
            a.GlobalIrr()
            a.RefIndex(salt_or_fresh)
            a.WaterTrans()
            a.ConditionalCoeff(lat,long)
            a.IncidentPower(depth)
            a.PhotonFlux()
            a.CellSpecs()
            a.IVCurve()
            a.Pmax()
            build.append(a.PMAX)
            pdens.append(a.Pden_out)    # Converted Power output 
            pincid.append(a.IncidentP)
            #print(a.PMAX)
            #print(a.Pden_out)
            #print(a.IncidentP)         # Irradiance incident on cell at depth
            #print('Imp ', a.Imp)
            #print('Vmp ', a.Vmp)
            print('Long =', long)
            print('Lat =', lat)
            long = long+res               # using poor resolution for now just to speed up results for adjustments
        lat = lat-res
        #print(lat)
        #dataplot.append(build)
        dataplot.append(pdens)

        
#print('Output dens =', max(pdens))
#print('Incident dens =', max(pincid)/10)        # converted to mW/cm^2
#print('Efficiency =', (100*max(pdens))/(max(pincid)/10)) #pincident /10 because it's in W/m^2, pdens in mW/cm^2
        

#plt.imshow(dataplot, cmap='hot', interpolation='nearest')
#plt.savefig('heat.jpeg')
#plt.show()

# no legend whilst i configure coords
fig, ax = plt.subplots()
im = ax.imshow(dataplot, cmap='hot', interpolation='nearest')
#cbar1 = ax.figure.colorbar(im, ax = ax, orientation='horizontal')
cbar2 = ax.figure.colorbar(im, ax = ax, orientation='vertical')
#cbar.ax.set_ylabel("Color bar", rotation = -90, va = "bottom")
#cbar1.ax.set_xlabel("Power Output", rotation = 0, va = "top") # label for color bar
cbar2.ax.set_ylabel("Power Output (mW/cm2)", rotation = -90, va = "bottom")
plt.show()
#plt.savefig("heatbackup.png")
### both vertical and horiz. color bar included, which one?




#background = Image.open("heat.png")
#foreground = Image.open("WorldMap.png").convert("RGBA")

#background.paste(foreground, (0, 0), foreground)
#background.save("out.png")


#print(">>>>>>>>>>>>>>>>>:X")
#print(x)
#print(">>>>>>>>>>>>>>>>>:Y")
#print(y)

#fig, ax = plt.subplots()
#ax.plot(x, y)            
#ax.set(xlabel='Depth(m)', ylabel='Pmax(W)', title='Power(depth)')
#ax.grid()
#plt.show()





# How to write data to a file, may need later 
# wv and I_x_ written to a text file "spectra_data.txt", with one space in between columns 

#OUT_FILE= 'spectra_data.txt'
 
#def writeDataToFile(x, y):
#       with open(OUT_FILE, 'w') as f:
#               for i in range(len(x)):
#                       f.write(f"{x[i]} {y[i]}\n")
#       print(f"Data written to {OUT_FILE}")

#writeDataToFile(x, y)





