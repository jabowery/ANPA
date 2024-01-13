# Generate the Combinatorial Hierarchy
# Reference example equations 2.3 and 2.4 of"On the Physical 
# Interpretation and the Mathematical Structure of the Combinatorial Hierarchy"
# by Ted Bastin, H. Pierre Noyes, John Amson and Clive Kilmister
# 1979

import itertools
import functools
import numpy as np
from sympy import Matrix
import sys

def isin(npar, setofnpar):
    for n in range(len(setofnpar)):
        if np.array_equal(setofnpar[n],npar):
            return n # return index
    return -1 # failed == illegal index

def generate_bool_matrices(mdcs):
    n = len(mdcs[0])
    for mrows in itertools.combinations(mdcs, n): # PITCH cols (col major) are numpy rows (row major)
        for pmrows in itertools.permutations(mrows): # so order of rows matters
            mat = np.array(pmrows).reshape(n, n).T
            yield mat

def closedUnderCross(dcss, m):
    for element in dcss:
        #        print(to01s((element,m)))
        if isin(np.dot(m,element) % 2, dcss) == -1:
            return False
    return True

def combs(things,n):
        if n == 1:
                return [x[0] for x in itertools.combinations(things, 1)]
        else:
#                return [x for x in itertools.combinations(things,n)]+combs(things, n-1)
                return [functools.reduce(np.logical_xor,x).astype(int) for x in itertools.combinations(things,n)]+combs(things, n-1)


def Sprout(seeds):
    for natat in range(1,len(seeds)+1):
        myseeds = [x for x in itertools.combinations(seeds, natat)]
        if natat==1:
            DCsSs=[[x[0]] for x in myseeds]
        else:
            for seed in myseeds:
                DCsSs.append(combs(seed, natat))
    return DCsSs

def to01s(matvs): # convert vector matrix,vecs pair to 01s string for print
    n = matvs[1].shape[0] # all dimensions are the same n-rows n-columns
    mat = matvs[1]
    vecs = matvs[0]
    st = ""
    for r in range(n):
        for c in range(n):
            st += str(mat[r,c]) # next column of matrix row
        st += ' '    # space out in prep for vecs
        for vec in vecs:
            st += str(vec[r]) + ' '
        st += "\n"
    return st

def get_independent_vectors(vectors):
    dim = len(vectors[0])
    result = []
    for col in range(dim):
        pivot = None
        for row in range(len(vectors)):
            if vectors[row][col] == 1:
                pivot = row
                break
        if pivot is not None:
            result.append(vectors[pivot])
            for row in range(len(vectors)):
                if row != pivot and vectors[row][col] == 1:
                    vectors[row] = [vectors[row][i]^vectors[pivot][i] for i in range(dim)]
    return result



def check_linear_indep(bool_vectors):
    # Convert list of boolean vectors Matrix
    M = Matrix(bool_vectors)

    # Reduce the matrix to Echelon form
    M = M.rref()[0]

    # Check rows for all zeros
    for row in M.tolist():
        if sum([abs(i) for i in row]) == 0:
            return False
    return True

def CHLevel(LevelSeed):
    assert check_linear_indep(LevelSeed.copy())
    DCsSs=Sprout(LevelSeed.copy())
    MDsS=DCsSs[-1] #the last in the list is the Maximal DCsSs
    print(len(MDsS))
    print("MDsS:")
    for element in MDsS:
        print(''.join(map(str, element)))
    ms = []
    cnt = 0
    for m in generate_bool_matrices(MDsS):
#        if cnt%100==0:
#            print(cnt,file=sys.stderr, end="\r")
        cnt += 1
#        print(''.join(map(str,list(m.flatten()))))
        for DCsS in DCsSs:
            StillOK = False
            if closedUnderCross(DCsS,m):
                StillOK = True # OK but it can't be closed under any others because "iff"
                # so continue to check it doesn't work for all other vectors (but null (first) vector)
                for element in [np.array(x) for x in itertools.product([0,1],repeat=len(DCsS[0]))][1:]: 
                        # if this element isn't int he DCsS being tested and the matrix is its identity
                        if isin(element, DCsS)==-1 and np.array_equal(np.dot(m,element) % 2, element):
                            StillOK = False # discard this matrix for this DCsS
                            break
                # if matrix is identify for this DCsS's vectors and no other vectors
                if StillOK:
                    break # this matrix can't be the automorphism for any other possible DCsS
        # if automorphism matrix has been identified
        if StillOK:
            ms.append((DCsS,m)) # use this matrix as the automorphism for this DCsS
            ind = isin(ms[-1][0],DCsSs) # We no longer need find an automorphism for this DCsS
            del DCsSs[ind] # so delete it from the set of all DCSsSs needing an automorphism matrix
            # if no more DCsSs need automorophisms
            if len(DCsSs)==0:
                break # we've finished this level
    return ms # return this level's DCsSs and their automorphim matrices
            
Seed = [[0,1],[1,0]] 
Level = 0
while True:
    Seed = [np.array(x) for x in Seed] # convert set of lists to set of vectors
    Level += 1
    print(f'Level{Level}')
    ms = CHLevel(Seed)
    print(f'{len(ms)} matrices:')
    Seed = [] # build Seed for next Level
    for m in ms:
        print(to01s(m))
        mcols = list(m[1])
        Seed.append(m[1].flatten())
