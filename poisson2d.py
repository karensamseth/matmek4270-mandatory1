import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange

x, y = sp.symbols('x,y')

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), in [0, L]^2

    where L is the length of the domain in both x and y directions.
    Dirichlet boundary conditions are used for the entire boundary.
    The Dirichlet values depend on the chosen manufactured solution.

    """

    def __init__(self, L, ue): #Michael
        """Initialize Poisson solver for the method of manufactured solutions

        Parameters
        ----------
        L : number
            The length of the domain in both x and y directions
        ue : Sympy function
            The analytical solution used with the method of manufactured solutions.
            ue is used to compute the right hand side function f.
        """
        self.L = L
        self.ue = ue
        self.f = sp.diff(self.ue, x, 2)+sp.diff(self.ue, y, 2)

    def create_mesh(self, N):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.N = N
        self.h = self.L/N
        x = self.x = np.linspace(0, self.L, self.N+1)
        y = self.y = np.linspace(0, self.L, self.N+1)
        self.xij, self.yij = np.meshgrid(x,y,indexing='ij')
        return x,y

    def D2(self):
        """Return second order differentiation matrix"""
        D = sparse.diags([1,-2,1], [-1,0,1], (self.N+1, self.N+1),'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        return D

    def laplace(self):
        """Return vectorized Laplace operator"""
        D2x = (1./self.h**2)*self.D2()
        D2y = (1./self.h**2)*self.D2()
        return sparse.kron(D2x, sparse.eye(self.N+1)) + sparse.kron(sparse.eye(self.N+1),  D2y)

    def get_boundary_indices(self):
        """Return indices of vectorized matrix that belongs to the boundary"""
        B = np.ones((self.N+1, self.N+1), dtype=bool) #enere
        B[1:-1,1:-1] = 0 #null i midten
        bnds = np.where(B.ravel() == 1)[0] #hvor enerne er i B
        return bnds

    def assemble(self):
        """Return assembled matrix A and right hand side vector b"""
        #f = ue.diff(x, 2) + ue.diff(y, 2)
        F = sp.lambdify((x, y), self.f)(self.xij,self.yij)
        A = self.laplace()
        A = A.tolil()
        bnds = self.get_boundary_indices()
        for i in bnds:
            A[i] = 0 #rundt kanten
            A[i,i] = 1 #oppe til venstre og nede til høyre
        b = F.ravel()
        ue_func = sp.lambdify((x,y),ue)(self.xij, self.yij).ravel()
        b[bnds] = ue_func[bnds]
        return A.tocsr(), b

    def l2_error(self, u):
        """Return l2-error norm"""
        uj = sp.lambdify([x,y], self.ue)(self.x, self.y) #evaluerer ue i meshpoints
        return np.sqrt(self.h**2*np.sum((uj-u)**2))        

    def __call__(self, N): #Michael
        """Solve Poisson's equation.

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction

        Returns
        -------
        The solution as a Numpy array

        """
        self.create_mesh(N)
        A, b = self.assemble()
        self.U = sparse.linalg.spsolve(A, b.flatten()).reshape((N+1, N+1))
        return self.U

    def convergence_rates(self, m=6): #Michael
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretization levels to use

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            u = self(N0)
            E.append(self.l2_error(u))
            h.append(self.h)
            N0 *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)
    
    def Lagrangebasis(self, xj, x=x):
        """Construct Lagrange basis function for points in xj
        
        Parameters
        --------
        xj : array
            Interpolation points
        x : sympy symbol
        
        Returns
        --------
        Lagrange basis functions
        """
        n = len(xj)
        ell = []
        numert = sp.Mul(*[x-xj[i] for i in range(n)])
        
        for i in range(n):
            numer = numert/(x-xj[i])
            denom = sp.Mul(*[(xj[i]-xj[j]) for j in range(n) if i!=j])
            ell.append(numer/denom)
        return ell
    
    def Lagrangefunction2D(self, u, basisx, basisy):
        """Return Lagrange polynomial for 2D case
        
        Parameters
        --------
        u : array
            Mesh function values
        basisx : tuple of Lagrange basis functions
            Output from Lagrangebasis
        basisy : same as over.
        """
        N, M = u.shape
        f = 0
        for i in range(N):
            for j in range(M):
                f += basisx[i]*basisy[j]*u[i,j]
        return f

    def eval(self, x_coord, y_coord):
        """Return u(x, y)

        Parameters
        ----------
        x, y : numbers
            The coordinates for evaluation

        Returns
        -------
        The value of u(x, y)

        """
        # Finner de 4 punktene som er rundt (x_coord,y_coord): 
        print("h=",self.h)
        print("evalueringspunkt:",x_coord,y_coord)
        for xi in self.x: #x-koord.
            if x_coord-xi==0:
                break
            elif x_coord-xi<self.h and x_coord-xi>0:
                x_lower = xi
            elif -(x_coord-xi)<self.h and -(x_coord-xi)>0:
                x_upper = xi
        for yi in self.y: #y-koord
            if y_coord-yi==0:
                break
            elif y_coord-yi<self.h and y_coord-yi>0:
                y_lower = yi
            elif -(y_coord-yi)<self.h and -(y_coord-yi)>0:
                y_upper = yi
        if 'x_upper' in locals():    
            print("x ligger mellom punktene ",x_lower,x_upper)
            print("y ligger mellom punktene ",y_lower,y_upper)
            ix_lower = int(np.where(self.x == x_lower)[0])
            ix_upper = int(np.where(self.x == x_upper)[0])
            x_interppunkter = slice(ix_lower,ix_upper+1)
            iy_lower = int(np.where(self.y == y_lower)[0])
            iy_upper = int(np.where(self.y == y_upper)[0])
            y_interppunkter = slice(iy_lower,iy_upper+1)
            #u_interppunkter = [[self.U[ix_lower,iy_upper], self.U[ix_upper,iy_upper]],\
            #                   [self.U[ix_lower,iy_lower], self.U[ix_upper,iy_lower]]]
            u_interppunkter = self.U[x_interppunkter, y_interppunkter]
            print("u-verdier:", u_interppunkter)
            # Lagrange basis: 
            lx = self.Lagrangebasis([x_lower,x_upper], x=x)
            ly = self.Lagrangebasis([y_lower,y_upper], x=y)
            # Lagrange function:
            f = self.Lagrangefunction2D(u_interppunkter, lx, ly)
            f_eval = f.subs({x: x_coord, y: y_coord})
            print ("u poly: " ,sp.simplify(f))
            print("u-verdi i punktet:", f_eval)
        else:
            ix_coord = int(np.where(self.x == x_coord)[0])
            iy_coord = int(np.where(self.y == y_coord)[0])
            f_eval = self.U[ix_coord,iy_coord]
            print("u-verdi i punktet: (eksakt) ", f_eval)
        return f_eval

def test_convergence_poisson2d(): #Michael
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    r, E, h = sol.convergence_rates()
    assert abs(r[-1]-2) < 1e-2

def test_interpolation(): #Michael
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    U = sol(100)
    assert abs(sol.eval(0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    assert abs(sol.eval(sol.h/2, 1-sol.h/2) - ue.subs({x: sol.h, y: 1-sol.h/2}).n()) < 1e-3

if __name__ == '__main__':
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    r, E, h = sol.convergence_rates()
    print("siste r: ",r[-1])
    K = abs(r[-1]-2) < 1e-2
    print("test convergence:",K,"\n")
    
    print("Test interpolation:")
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    U = sol(100)
    K1 = abs(sol.eval(0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    print("K1=", K1,"\n")
    K2 = abs(sol.eval(sol.h/2, 1-sol.h/2) - ue.subs({x: sol.h, y: 1-sol.h/2}).n()) < 1e-3
    print("K2: ", K2)
    