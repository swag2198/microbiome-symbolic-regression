from gplearn.functions import make_function

def _presence(x1):
    return (x1 > 0).astype(x1.dtype)

def _absence(x1):
    # this species is absent
    return (x1 <= 0).astype(x1.dtype)

def _presence2(x1, x2):
    # both are present
    return ((x1 > 0) * (x2 > 0)).astype(x1.dtype)

def _absence2(x1, x2):
    # both are absent
    return ((x1 <= 0) * (x2 <= 0)).astype(x1.dtype)

def _presence3(x1, x2, x3):
    # all 3 are present
    return ((x1 > 0) * (x2 > 0) * (x3 > 0)).astype(x1.dtype)

def _absence3(x1, x2, x3):
    # all 3 are absent
    return ((x1 <= 0) * (x2 <= 0) * (x3 <= 0)).astype(x1.dtype)


def _pres_abs(x1, x2):
    # xor
    x1p = x1 > 0
    x1a = x1 <= 0
    
    x2p = x2 > 0
    x2a = x2 <= 0
    
    return ((x1p * x2a) + (x1a * x2p)).astype(x1.dtype)


def _add3(x1, x2, x3):
    return x1 + x2 + x3

def _add10(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    return x1 + x2 + x3 + x4 + x5 +x6 + x7 + x8 + x9 + x10

def _ifelse(x1, x2):
    return x1*(x1 > x2).astype(x1.dtype)

def _ifelseless(x1, x2):
    return x1*(x1 <= x2).astype(x1.dtype)


presence  = make_function(function=_presence, name='presence', arity=1)    
absence   = make_function(function=_absence, name='absence', arity=1)
presence2 = make_function(function=_presence2, name='presence_both', arity=2)
absence2  = make_function(function=_absence2, name='absence_both', arity=2)
presabs   = make_function(function=_pres_abs, name='presence_absence', arity=2)
presence3 = make_function(function=_presence3, name='presence_3', arity=3)
absence3  = make_function(function=_absence3, name='absence_3', arity=3)
add3 = make_function(function=_add3, name='add3', arity=3)
add10 = make_function(function=_add10, name='add10', arity=10)
ifelse = make_function(function=_ifelse, name='ifelse >', arity=2)
ifelseless = make_function(function=_ifelseless, name='ifelse <=', arity=2)