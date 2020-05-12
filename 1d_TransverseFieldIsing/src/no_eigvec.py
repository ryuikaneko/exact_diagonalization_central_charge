#!/usr/bin/env python

# coding:utf-8
from __future__ import print_function
import math
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import argparse
import time
from numba import jit

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-L',metavar='L',dest='L',type=int,default=16,help='set L')
    parser.add_argument('-momk',metavar='momk',dest='momk',type=int,default=0,help='set momk')
    return parser.parse_args()

def num2bit(state,L):
    return np.binary_repr(state,L)

## https://stackoverflow.com/questions/8928240/convert-base-2-binary-number-string-to-int
def bit2num(bit):
    return int(bit,2)

## http://lptms.u-psud.fr/membres/groux/Test/ED/ED_Lecture1.html
def show_state01(state,L): # show spins from left to right
    return "|"+"".join([i for i in num2bit(state,L)[::-1]])+">" # use 0,1 with ket
#    return "".join([i for i in num2bit(state,L)[::-1]]) # use 0,1

def show_state(state,L): # show spins from left to right
    return "|"+"".join([ str('+') if i==str(0) else str('-') for i in num2bit(state,L)[::-1]])+">" # use +,- with ket
#    return "".join([ str('+') if i==str(0) else str('-') for i in num2bit(state,L)[::-1]]) # use +,-

## https://github.com/alexwie/ed_basics/blob/master/hamiltonian_hb_staggered.py
@jit(nopython=True)
def get_spin(state,site):
    return (state>>site)&1

## http://lptms.u-psud.fr/membres/groux/Test/ED/ED_Lecture1.html
def get_spin_alternative(state,site):
    return (state&(1<<site))>>site

@jit(nopython=True)
def shift_1spin(state,L):
    return ((state<<1)&(1<<L)-2)|((state>>(L-1))&1)

@jit(nopython=True)
def shift_1spin_inv(state,L):
    return ((state<<(L-1))&(1<<(L-1)))|((state>>1)&((1<<(L-1))-1))

def init_parameters(L):
    Nhilbert = 1 << L
    ihfbit = 1 << (L//2)
    irght = ihfbit-1
    ilft = ((1<<L)-1) ^ irght
    return Nhilbert, ihfbit, irght, ilft

## http://physics.bu.edu/~sandvik/vietri/dia.pdf
@jit(nopython=True)
def find_state_2(state,list_1,maxind):
    imin = 0
    imax = maxind-1
    while True:
        i = (imin+imax)//2
#        print(i,imin,imax,maxind,state,list_1[i])
        if (state < list_1[i]):
            imax = i-1
        elif (state > list_1[i]):
            imin = i+1
        else:
            break
        if (imin > imax):
            return -1
    return i

@jit(nopython=True)
def check_state(state,momk,L):
    t = state
    for i in range(L):
        t = shift_1spin(t,L)
        if (t < state):
            return -1
        elif (t == state):
            if (np.mod(momk,L//(i+1)) != 0):
                return -1
            else:
                return i+1

@jit(nopython=True)
def find_representative(state,L):
    rep = state
    tmp = state
    exponent = 0
    for i in range(L):
        tmp = shift_1spin(tmp,L)
        if (tmp < rep):
            rep = tmp
            exponent = i+1
    return rep, exponent

@jit(nopython=True)
def flip_1spin(state,i1):
    return state^(1<<i1)
#    return state^(2**i1)

@jit(nopython=True)
def flip_2spins(state,i1,i2):
    return state^((1<<i1)+(1<<i2))
#    return state^(2**i1+2**i2)

@jit(nopython=True)
def make_basis(L,momk):
    list_state = []
    list_R = []
    first = 0
    last = 1<<L
#    print("# first:",first,num2bit(first,L))
#    print("# last:",last,num2bit(last,L))
    Nrep = 0
    for state in range(first,last+1):
        R = check_state(state,momk,L)
        if (R>=0):
            list_state.append(state)
            list_R.append(R)
            Nrep += 1
    return list_state, list_R, Nrep

def calc_exp(L,momk):
    return np.array([np.exp(-1j*exponent*2.0*np.pi*momk/L) for exponent in range(L)])

@jit(nopython=True)
def make_hamiltonian_child(Nbond,list_site1,list_site2,Nrep,list_state,list_sqrtR,L,momk,expk):
#
#---- FM TFIsing model (spin: \sigma)
## sx.sx + sy.sy: #elements = Nrep*Nbond (not used in TFIsing)
## sz.sz:         #elements = Nrep*1 (diagonal elements)
## sx:            #elements = Nrep*L
##
    listki = np.array([i for k in range(Nbond+1+L) for i in range(Nrep)],dtype=np.int64)
    loc = np.zeros((Nbond+1+L)*Nrep,dtype=np.int64)
    elemnt = np.zeros((Nbond+1+L)*Nrep,dtype=np.complex128)
#    Ham = np.zeros((Nrep,Nrep),dtype=complex)
    for a in range(Nrep):
        sa = list_state[a]
        for i in range(Nbond): ## Ising (- \sigma^z \sigma^z)
            i1 = list_site1[i]
            i2 = list_site2[i]
            loc[Nbond*Nrep+a] = a
            if get_spin(sa,i1) == get_spin(sa,i2):
#                Ham[a,a] -= 1.0
                elemnt[Nbond*Nrep+a] -= 1.0
            else:
#                Ham[a,a] += 1.0
                elemnt[Nbond*Nrep+a] += 1.0
        for i in range(L): ## Transverse field (- \sigma^x = -2 S^x = - S^+ - S^-)
            bb = flip_1spin(sa,i)
            sb, exponent = find_representative(bb,L)
            b = find_state_2(sb,list_state,Nrep)
            if b>=0:
#                Ham[a,b] -= list_sqrtR[a]/list_sqrtR[b]*expk[exponent]
                elemnt[(Nbond+1+i)*Nrep+a] -= list_sqrtR[a]/list_sqrtR[b]*expk[exponent]
                loc[(Nbond+1+i)*Nrep+a] = b
#---- end of FM TFIsing model (spin: \sigma)
#
##---- AF Heisenberg model (spin: S = \sigma / 2)
#    listki = np.array([i for k in range(Nbond+1) for i in range(Nrep)],dtype=np.int64)
#    loc = np.zeros((Nbond+1)*Nrep,dtype=np.int64)
#    elemnt = np.zeros((Nbond+1)*Nrep,dtype=np.complex128)
##    Ham = np.zeros((Nrep,Nrep),dtype=complex)
#    for a in range(Nrep):
#        sa = list_state[a]
#        for i in range(Nbond):
#            i1 = list_site1[i]
#            i2 = list_site2[i]
#            loc[Nbond*Nrep+a] = a
#            if get_spin(sa,i1) == get_spin(sa,i2):
##                Ham[a,a] += 0.25
#                elemnt[Nbond*Nrep+a] += 0.25
#            else:
##                Ham[a,a] -= 0.25
#                elemnt[Nbond*Nrep+a] -= 0.25
#                bb = flip_2spins(sa,i1,i2)
#                sb, exponent = find_representative(bb,L)
#                b = find_state_2(sb,list_state,Nrep)
#                if b>=0:
##                    Ham[a,b] += 0.5*list_sqrtR[a]/list_sqrtR[b]*expk[exponent]
#                    elemnt[i*Nrep+a] += 0.5*list_sqrtR[a]/list_sqrtR[b]*expk[exponent]
#                    loc[i*Nrep+a] = b
##---- end of AF Heisenberg model (spin: S = \sigma / 2)
#
## https://stackoverflow.com/questions/19420171/sparse-matrix-in-numba
## Unknown attribute 'csr_matrix' of type Module
#    Ham = scipy.sparse.csr_matrix((elemnt,(listki,loc)),shape=(Nrep,Nrep),dtype=np.complex128)
#    return Ham
    return elemnt, listki, loc

def make_hamiltonian(Nrep,elemnt,listki,loc):
    return scipy.sparse.csr_matrix((elemnt,(listki,loc)),shape=(Nrep,Nrep),dtype=np.complex128)


def main():
    args = parse_args()
    L = args.L
    momk = args.momk

    start = time.time()
    print("# make basis: sz not conserved")
    list_state, list_R, Nrep = make_basis(L,momk)
    list_state = np.array(list_state,dtype=np.int64)
    list_R = np.array(list_R,dtype=np.int64)
    list_sqrtR = np.sqrt(list_R)
    print("# L=",L,", momk=",momk,", Nrep=",Nrep)
    print("# show first and last bases")
    print("# ind state_num state_bit period_R period_sqrtR")
#    for i in range(Nrep):
    for i in range(0,Nrep,Nrep-1):
        print(i,list_state[i],num2bit(list_state[i],L),list_R[i],list_sqrtR[i])
    end = time.time()
    print("# time:",end-start)
    print()

    start = time.time()
    end = time.time()
    print("# make interactions")
    Nbond = L
    list_site1 = np.array([i for i in range(Nbond)],dtype=np.int64)
    list_site2 = np.array([(i+1)%L for i in range(Nbond)],dtype=np.int64)
    print("list_site1=",list_site1)
    print("list_site2=",list_site2)
    end = time.time()
    print("# time:",end-start)
    print()

    start = time.time()
    end = time.time()
    print("# make Hamiltonian")
    expk = calc_exp(L,momk)
#    Ham = make_hamiltonian(Nbond,list_site1,list_site2,Nrep,list_state,list_sqrtR,L,momk,expk)
    elemnt, listki, loc = make_hamiltonian_child(Nbond,list_site1,list_site2,Nrep,list_state,list_sqrtR,L,momk,expk)
    Ham = make_hamiltonian(Nrep,elemnt,listki,loc)
#    print(Ham)
    end = time.time()
    print("# time:",end-start)
    print()

    start = time.time()
    end = time.time()
    print("# diag Hamiltonian")
    Neig = 5
#    ene,vec = scipy.linalg.eigh(Ham,eigvals=(0,min(Neig,Nrep-1)))
#    ene,vec = scipy.sparse.linalg.eigsh(Ham,which='SA',k=min(Neig,Nrep-1))
    ene = scipy.sparse.linalg.eigsh(Ham,which='SA',k=min(Neig,Nrep-1),return_eigenvectors=False)
#    ene4 = 4.0*ene
#    ene = np.sort(ene4)
    ene = np.sort(ene)
    end = time.time()
    print ("L k energy:",L,momk,ene[0],ene[1],ene[2],ene[3],ene[4])
    print("# time:",end-start)
    print()

if __name__ == "__main__":
    main()
