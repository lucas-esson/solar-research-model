### overlay map plot ###

########################################
############ INSTRUCTIONS ##############

# 1. Produce Heat Map, crop w/ legend
# 2. Save as heatcopy
# 3. Trigger this code






import math
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import integrate
from scipy.interpolate import CubicSpline
from PIL import Image

heat = Image.open('heatcopy.png').convert("RGBA")   # have to crop before use 
heat = heat.resize((1920,960))
world = Image.open('WorldFinal.png').convert("RGBA") 
#world = world.resize((360,180))

background = heat
foreground = world

background.paste(foreground, (0, 0), foreground)
background.save("out.png")
background.show()


