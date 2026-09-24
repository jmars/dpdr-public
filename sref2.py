import sys
sys.setrecursionlimit(300000)

def V(n): return ('v', n)
def A(n): return ('a', n)
def C(f, *args): return ('c', f, args)

def walk(t, s):
    """RECURSIVE deref - canonicalises nested terms (the bug in v1)."""
    if t[0] == 'v':
        return walk(s[t[1]], s) if t[1] in s else t
    if t[0] == 'c':
        return ('c', t[1], tuple(walk(x, s) for x in t[2]))
    return t

def unify(a, b, s):
    a = walk(a, s); b = walk(b, s)
    if a == b: return s
    if a[0] == 'v':
        s2 = dict(s); s2[a[1]] = b; return s2
    if b[0] == 'v':
        s2 = dict(s); s2[b[1]] = a; return s2
    if a[0] == 'c' and b[0] == 'c' and a[1] == b[1] and len(a[2]) == len(b[2]):
        for x, y in zip(a[2], b[2]):
            s = unify(x, y, s)
            if s is None: return None
        return s
    return None

STEPS=[0]; LIMIT=60000

def solve(goals, subst, clauses, table, K):
    if not goals: return subst
    STEPS[0]+=1
    if STEPS[0]>LIMIT: raise RuntimeError("STUCK")
    goal, rest = goals[0], goals[1:]
    g = walk(goal, subst)
    key = repr(g)
    if key in table: return None
    if len(table) < K: table[key]=True
    for head, body in clauses:
        s = unify(g, head, subst)
        if s is None: continue
        r = solve(list(body)+rest, s, clauses, table, K)
        if r is not None: return r
    return None

def build(N):
    cl=[]
    for i in range(N-1):
        cl.append((C('d%d'%i,V('x')), [C('d%d'%(i+1),V('x'))]))
    cl.append((C('d%d'%(N-1),V('x')), [C('d0',V('x'))]))
    cl.append((C('d%d'%(N-1),V('x')), [C('fact',V('x'))]))
    cl.append((C('fact',A('f0')), []))
    return cl

def run(N,K):
    STEPS[0]=0
    try:
        r=solve([C('d0',A('f0'))],{},build(N),{},K)
        return STEPS[0], r is not None
    except RuntimeError:
        return STEPS[0], False

if __name__=="__main__":
    print("FIXED engine. cache capacity K needed vs self-reference depth N")
    print("       N | K=0  K=1  K=2  K=3  K=4  K=8")
    for N in [2,4,8,16,32,64,128]:
        row=[]
        for K in [0,1,2,3,4,8]:
            _,ok=run(N,K); row.append(' ok ' if ok else ' -- ')
        print(f"N={N:5d} | "+"".join(row))
    print()
    print("steps to resolve (unbounded cache) vs depth:")
    for N in [2,4,8,16,32,64]:
        st,ok=run(N,10**6); print(f"  N={N:4d}: steps={st:5d} resolved={ok}")
