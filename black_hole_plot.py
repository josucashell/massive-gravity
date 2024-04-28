#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 19:21:59 2024

@author: josucashell
"""

import numpy as np
import matplotlib.pyplot as plt
F1 = 1
G1 = 1
F2 = -2/13 * (19 + 5 * np.sqrt(3))
G2a = -1/13 * (5 - 11 * np.sqrt(3))
G2b = -3/13 * (11 + 7 * np.sqrt(3))
F3a = (17367 + 7862 * np.sqrt(3))/1690
F3b = (27883 + 18662 * np.sqrt(3))/5070
G3a = (14305 - 3586 * np.sqrt(3))/5070
G3b = (21893 + 15278 * np.sqrt(3))/1690
F4a = -8 * (20212216 + 10761615 * np.sqrt(3))/3987555
F4b = -4 * (86474551 + 52074576 * np.sqrt(3))/3987555
G4a = -((13859147 - 13307623 * np.sqrt(3))/7975110)
G4b = -7 * (18003911 + 11074631 * np.sqrt(3))/1329185
G4c = -((245168455 + 136955569 * np.sqrt(3))/7975110)
F5a = (2358595986147 + 1341907450280 * np.sqrt(3))/14514700200
F5b = (6207302851481 + 3597373424072 * np.sqrt(3))/7257350100
F5c = (791752137769 + 452552246456 * np.sqrt(3))/4838233400
G5a = (58623861651 - 29657041376 * np.sqrt(3))/14514700200
G5b = (32883045033 + 19373615360 * np.sqrt(3))/59978100
G5c = (830194303889 + 475959378368 * np.sqrt(3))/1319518200

def rhoext(t,r,rg, beta):
    rhoseriesext=[G1, G2a+G2b*np.tanh(t/(2*rg))**2, G3a+G3b*np.tanh(t/(2*rg))**2, G4a+G4b*np.tanh(t/(2*rg))**2+G4c*np.tanh(t/(2*rg))**4, G5a+G5b*np.tanh(t/(2*rg))**2+G5c*np.tanh(t/(2*rg))**4]
    rho = np.sqrt(3)*beta*rg*(rhoseriesext[0]*np.cosh(t/(2*rg))*np.sqrt(r/rg-1)\
                              + rhoseriesext[1]*(np.cosh(t/(2*rg))**2)*(r/rg-1)\
                              + rhoseriesext[2]*(np.cosh(t/(2*rg))**3)*(np.sqrt(r/rg-1)**3)\
                              + rhoseriesext[3]*(np.cosh(t/(2*rg))**4)*((r/rg-1)**2)\
                              + rhoseriesext[4]*(np.cosh(t/(2*rg))**5)*(np.sqrt(r/rg-1)**5))
    return rho

def tauext(t,r,rg,beta):
    tauseriesext=[F1, F2, F3a+F3b*np.tanh(t/(2*rg))**2, F4a+F4b*np.tanh(t/(2*rg))**2, F5a+F5b*np.tanh(t/(2*rg))**2+F5c*np.tanh(t/(2*rg))**4]
    tau = np.sqrt(3)*beta*rg*np.tanh(t/(2*rg))*(tauseriesext[0]*np.cosh(t/(2*rg))*np.sqrt(r/rg-1)\
                              + tauseriesext[1]*(np.cosh(t/(2*rg))**2)*(r/rg-1)\
                              + tauseriesext[2]*(np.cosh(t/(2*rg))**3)*(np.sqrt(r/rg-1)**3)\
                              + tauseriesext[3]*(np.cosh(t/(2*rg))**4)*((r/rg-1)**2)\
                              + tauseriesext[4]*(np.cosh(t/(2*rg))**5)*(np.sqrt(r/rg-1)**5))
    return tau

def rhoint(t,r,rg, beta):
    rhoseriesint=[G1, G2a*np.tanh(t/(2*rg))+(G2b/np.tanh(t/(2*rg))), G3b+G3a*np.tanh(t/(2*rg))**2, (G4c/np.tanh(t/(2*rg)))+G4b*np.tanh(t/(2*rg))+G4a*np.tanh(t/(2*rg))**3, G5c+G5b*np.tanh(t/(2*rg))**2+G5a*np.tanh(t/(2*rg))**4]
    rho = np.sqrt(3)*beta*rg*np.tanh(t/(2*rg))*(rhoseriesint[0]*np.cosh(t/(2*rg))*np.sqrt(1-r/rg)\
                              + rhoseriesint[1]*(np.cosh(t/(2*rg))**2)*(1-r/rg)\
                              + rhoseriesint[2]*(np.cosh(t/(2*rg))**3)*(np.sqrt(1-r/rg)**3)\
                              + rhoseriesint[3]*(np.cosh(t/(2*rg))**4)*((1-r/rg)**2)\
                              + rhoseriesint[4]*(np.cosh(t/(2*rg))**5)*(np.sqrt(1-r/rg)**5))
    return rho

def tauint(t,r,rg,beta):
    tauseriesint=[F1, F2*np.tanh(t/(2*rg)), F3b+F3a*np.tanh(t/(2*rg))**2, F4b*np.tanh(t/(2*rg))+F4a*np.tanh(t/(2*rg))**3, F5c+F5b*np.tanh(t/(2*rg))**2+F5a*np.tanh(t/(2*rg))**4]
    tau = np.sqrt(3)*beta*rg*(tauseriesint[0]*np.cosh(t/(2*rg))*np.sqrt(1-r/rg)\
                              + tauseriesint[1]*(np.cosh(t/(2*rg))**2)*(1-r/rg)\
                              + tauseriesint[2]*(np.cosh(t/(2*rg))**3)*(np.sqrt(1-r/rg)**3)\
                              + tauseriesint[3]*(np.cosh(t/(2*rg))**4)*((1-r/rg)**2)\
                              + tauseriesint[4]*(np.cosh(t/(2*rg))**5)*(np.sqrt(1-r/rg)**5))
    return tau

r = np.linspace(1.0001,2,1000)
tconst=[0, 0.5, 1, 1.5]
rconst=[1.01,1.035,1.06,1.085]
rmax=[1.5,1.6,1.5,1.5]
tmax=[10,10,10,10]
rconstint = [0.986,0.974,0.962,0.95]
tconstint=[0.0001, 0.5, 1, 2]
plt.subplot(1,2,1)
alphavals=[1, 0.8, 0.6, 0.4]
alphavalsred=[1, 0.8, 0.6, 0.4]

rint=np.linspace(0.001,0.99999, 1000)
t = np.linspace(-tmax[0],tmax[0], 1000)
plt.plot(rhoint(t, rconstint[0], 1, 1), tauint(t, rconstint[0], 1, 1), color="red", label="Const r", alpha=alphavalsred[0])
plt.plot(rhoint(tconstint[0], rint, 1, 1), tauint(tconstint[0], rint, 1, 1), color="blue", label="Const t", alpha=alphavals[0])
for i in range(4):
    r = np.linspace(1,rmax[i],1000)
    t = np.linspace(-tmax[i],tmax[i], 1000)
    plt.plot(rhoext(tconst[i], r, 1, 1), tauext(tconst[i], r, 1, 1), color="blue", alpha=alphavals[i])
    plt.plot(rhoext(-tconst[i], r, 1, 1), tauext(-tconst[i], r, 1, 1), color="blue", alpha=alphavals[i])
    plt.plot(rhoext(t, rconst[i], 1, 1), tauext(t, rconst[i], 1, 1), color="red", alpha=alphavalsred[i])
    rint=np.linspace(0.001,0.99999, 1000)
    plt.plot(rhoint(t, rconstint[i], 1, 1), tauint(t, rconstint[i], 1, 1), color="red", alpha=alphavalsred[i])
    plt.plot(rhoint(tconstint[i], rint, 1, 1), tauint(tconstint[i], rint, 1, 1), color="blue", alpha=alphavals[i])
a=np.linspace(-4,4,1000)
plt.plot(a,a, color="black")
plt.plot(a,-a, color="black")
plt.text(0.25,0.35,r"$r=r_g$, $t\to \infty$", rotation=45)
plt.text(0.76,0.025,r"$t=0$")
plt.axis('square')
plt.ylim(-1,1)
plt.xlim(-1,1)
plt.xticks((-1.,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1),(-1,"",-0.5,"",0,"",0.5,"",1))
plt.yticks((-1.,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1),(-1,"",-0.5,"",0,"",0.5,"",1))
plt.xlabel(r"$\rho$(t,r)")
plt.ylabel(r"$\tau$(t,r)")
plt.title("Schwarzschild Coordinates")
plt.tight_layout()
plt.legend(loc=3)
plt.grid()


import scipy.special as sp

def lambertwfunc(X,T,n, order):
    lambert = 0
    x=(X**2-T**2)/np.e
    for i in range(1,n):
        lambert+=((-i)**(i-1)*x**(i-1))/sp.factorial(i)
    res = (X*np.sqrt(lambert/np.e))**order
    return res
    
order=13
def rhoTXext(T,X,beta):
    rhoseriesTXext=[G1, G2a+G2b*(T/X)**2, G3a+G3b*(T/X)**2, G4a+G4b*(T/X)**2+G4c*(T/X)**4, G5a+G5b*(T/X)**2+G5c*(T/X)**4]
    rho = np.sqrt(3)*beta*(rhoseriesTXext[0]*lambertwfunc(X, T, order,1)\
                               + rhoseriesTXext[1]*lambertwfunc(X, T, order,2)\
                               + rhoseriesTXext[2]*lambertwfunc(X, T, order,3)\
                               + rhoseriesTXext[3]*lambertwfunc(X, T, order,4)\
                               + rhoseriesTXext[4]*lambertwfunc(X, T, order,5))
    return rho

def tauTXext(T,X, beta):
    tauseriesext=[F1, F2, F3a+F3b*(T/X)**2, F4a+F4b*(T/X)**2, F5a+F5b*(T/X)**2+F5c*(T/X)**4]
    tau = np.sqrt(3)*beta*(T/X)*(tauseriesext[0]*lambertwfunc(X, T, order,1)\
                               + tauseriesext[1]*lambertwfunc(X, T, order,2)\
                               + tauseriesext[2]*lambertwfunc(X, T, order,3)\
                               + tauseriesext[3]*lambertwfunc(X, T, order,4)\
                               + tauseriesext[4]*lambertwfunc(X, T, order,5))
    return tau

# def rhoTXint(T,X,beta):
#     rhoseriesint=[G1, G2a*(X/T)+(G2b/(X/T)), G3b+G3a*(X/T)**2, (G4c/(X/T))+G4b*(X/T)+G4a*(X/T)**3, G5c+G5b*(X/T)**2+G5a*(X/T)**4]
#     rho = np.sqrt(3)*beta*np.tanh(t/(2*rg))*(rhoseriesint[0]*np.cosh(t/(2*rg))*np.sqrt(1-r/rg)\
#                               + rhoseriesint[1]*(np.cosh(t/(2*rg))**2)*(1-r/rg)\
#                               + rhoseriesint[2]*(np.cosh(t/(2*rg))**3)*(np.sqrt(1-r/rg)**3)\
#                               + rhoseriesint[3]*(np.cosh(t/(2*rg))**4)*((1-r/rg)**2)\
#                               + rhoseriesint[4]*(np.cosh(t/(2*rg))**5)*(np.sqrt(1-r/rg)**5))
#     return rho

# def tauTXint(T,X, beta):
#     tauseriesint=[F1, F2*(X/T), F3b+F3a*(X/T)**2, F4b*(X/T)+F4a*(X/T)**3, F5c+F5b*(X/T)**2+F5a*(X/T)**4]
#     tau = np.sqrt(3)*beta*(tauseriesint[0]*np.cosh(t/(2*rg))*np.sqrt(1-r/rg)\
#                               + tauseriesint[1]*(np.cosh(t/(2*rg))**2)*(1-r/rg)\
#                               + tauseriesint[2]*(np.cosh(t/(2*rg))**3)*(np.sqrt(1-r/rg)**3)\
#                               + tauseriesint[3]*(np.cosh(t/(2*rg))**4)*((1-r/rg)**2)\
#                               + tauseriesint[4]*(np.cosh(t/(2*rg))**5)*(np.sqrt(1-r/rg)**5))
#     return tau

plt.subplot(1,2,2)
Tconst=[0.0001, 0.1,0.2,0.3, 0.36, 0.4]
Xconst=[0.01, 0.1,0.2,0.3,0.4, 0.5]
alphavals=[1,0.9,0.8,0.7,0.6,0.5]
alphavalsblue=[1,0.85,0.7,0.55,0.4,0.25]
T=np.linspace(-Xconst[0],0.651,1000)
X=np.linspace(-Tconst[0], 0.651, 1000)
plt.plot(rhoTXext(T, Xconst[0], 1), tauTXext(T, Xconst[0], 1), color="red", label="Const R")
plt.plot(rhoTXext(Tconst[0], X, 1), tauTXext(Tconst[0], X, 1), color="blue", label="Const T")
for i in range(len(Tconst)):
    T=np.linspace(-Xconst[i],0.651,1000)
    X=np.linspace(-Tconst[i], 0.651, 1000)
    Xneg=np.linspace(Tconst[i], 0.61, 1000)
    plt.plot(rhoTXext(T, Xconst[i], 1), tauTXext(T, Xconst[i], 1), color="red", alpha=alphavals[i])
    plt.plot(rhoTXext(Tconst[i], X, 1), tauTXext(Tconst[i], X, 1), color="blue", alpha=alphavalsblue[i])
    plt.plot(rhoTXext(-Tconst[i], Xneg, 1), tauTXext(-Tconst[i], Xneg, 1), color="blue", alpha=alphavalsblue[i])
a=np.linspace(-4,4,1000)
plt.plot(a,a, color="black")
plt.plot(a,-a, color="black")
plt.text(0.225,0.325,r"$T^2=X^2$", rotation=45)
plt.text(0.74,0.025,r"$T=0$")
plt.axis('square')
plt.ylim(-1,1)
plt.xlim(-1,1)
plt.legend(loc=3)
plt.xticks((-1.,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1),(-1,"",-0.5,"",0,"",0.5,"",1))
plt.yticks((-1.,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1),(-1,"",-0.5,"",0,"",0.5,"",1))
plt.xlabel(r"$\rho$(T,R)")
plt.ylabel(r"$\tau$(T,R)")
plt.grid()
plt.title("Kruskal-Szekeres Coordinates")
plt.tight_layout()
plt.show()
