### Refractive Index Model ###


wv = []
n2 = []
n2s_ = []
n2f_ = []


with open("Ho.txt") as file:
    for line in file:
        s=line.rstrip().split()             #(1,2,3) to ['1','2','3']
        if len(s)==2:
            wv.append(float(s[0]))


# Temperature Input
Temp = input("Water temperature: ")
Temp = float(Temp)


        
            
# Seawater (salinity = 35%)
a1 = -1.50156*10**-6
b1 = 1.07085*10**-7
c1 = -4.27594*10**-5
d1 = -1.60476*10**-4
e1 = 1.39807

# Freshwater (salinity = 0%)
a2 = -1.97812*10**-6
b2 = 1.03223*10**-7
c2 = -8.58123*10**-6
d2 = -1.54834*10**-4
e2 = 1.38919





# Seawater n
for i in range(0,len(wv)):
    n2s = a1*Temp**2 + b1*wv[i]**2 + c1*Temp + d1*wv[i] + e1
    n2s_.append(n2s)

# Freshwater n
for i in range(0,len(wv)):
    n2f = a2*Temp**2 + b2*wv[i]**2 + c2*Temp + d2*wv[i] + e2
    n2f_.append(n2f)


# Sea or Fresh
choice = input("Input Water Type: Fresh or Salt: ")
for i in range(0,len(wv)):
    if str(choice) == "Salt":
        n2 = n2s_

    if str(choice) == "Fresh":
        n2 = n2f_

