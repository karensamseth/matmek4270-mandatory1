import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm

x, y, t = sp.symbols('x,y,t')

class Wave2D:
    """
    Solves the wave equation for 2D domain [0,1]**2 and time [0,T].
    Dirichlet initial condition.
    """
    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.L = 1
        self.N = N
        self.h = self.L/self.N #dx = dy =h
        x = self.x = np.linspace(0, self.L, self.N+1)
        y = self.y = np.linspace(0, self.L, self.N+1)
        self.xij, self.yij = np.meshgrid(x,y,indexing='ij')
        return self.h

    def D2(self):
        """Return second order differentiation matrix"""
        D = sparse.diags([1,-2,1], [-1,0,1], (self.N+1, self.N+1),'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        return D

    @property
    def w(self):
        """Return the dispersion coefficient"""
        kx = self.mx*sp.pi
        ky = self.my*sp.pi
        return self.c*sp.sqrt(kx**2+ky**2)

    def ue(self, mx, my, x, y, t): #Mikael
        """Return the exact standing wave"""
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        self.N = N
        self.mx = mx
        self.my = my
        
    @property
    def dt(self):
        """Return the time step"""
        return self.cfl*self.h/self.c

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        """
        Esum = 0
        for n in range(self.N+1):
            Un = self.plotdata(n)
            en = Un-self.ue(self.mx,self.my)  #dette er nok feil...
            Esum = en**2
        return np.sqrt(self.h**2*Esum)
        """
        ue_f = sp.lambdify((x,y),self.ue(self.mx,self.my,x,y,t0))(self.xij, self.yij)
        ut = ue_f-u
        return np.sqrt(self.h**2*np.sum(ut**2))

    def apply_bcs(self, u=None):
        """Apply Dirichlet boundary conditions to solution vector

        Parameters
        ----------
        u : array, optional
            The solution array to fix at boundaries.
            If not probided, use self.unp1

        """
        u[0] = 0
        u[-1] = 0
        u[:, 0] = 0 
        u[:, -1] = 0 
        

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.N = N
        self.Nt = Nt
        self.cfl = cfl
        self.c = c
        self.mx = mx
        self.my = my
        self.h = self.create_mesh(self.N)
        Unp1, Un, Unm1 = np.zeros((3, self.N+1, self.N+1))
        D = (1/self.h**2)*self.D2()
        Unm1[:] =sp.lambdify((x, y), self.ue(mx, my, x, y, 0))(self.xij, self.yij)
        #Unm1[:] = self.ue(self.mx, self.my, self.xij, self.yij, t=0)    #u0 initial condition, ie. ue at t=0
        Un[:] =sp.lambdify((x, y), self.ue(mx, my, x, y, self.dt))(self.xij, self.yij)
        #Un[:] = Unm1[:] + 0.5*(self.c/self.dt)**2* (D@Un + Un@D.transpose())
        
        plotdata = {0: Unm1.copy()}
        for n in range(1, self.Nt+1):
            Unp1[:] = 2*Un - Unm1 + (self.c*self.dt)**2* (D@Un + Un@D.transpose())
            # Boundary conditions:
            self.apply_bcs(Unp1)
            # Swap solutions:
            Unm1[:] = Un
            Un[:] = Unp1
            if n % store_data == 0:
                plotdata[n] = Unm1.copy() #which is what Un was earlier
            #print(np.linalg.norm(Un), np.linalg.norm(Unm1))
        if store_data == -1:
            print("h:",self.h,"l2 error:",self.l2_error(plotdata[0], 0))
            return (self.h, [self.l2_error(Un, (self.Nt+1)*self.dt)])
        elif store_data > 0:
            print("dt", self.dt, "solu",self.plotdata)
            return {"dt": self.dt, "solu":self.plotdata}
        return self.xij, self.yij, self.plotdata

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3): #Mikael
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

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
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):

    def D2(self):
        """Return second order differentiation matrix with Neumann bc"""
        D = sparse.diags([1,-2,1], [-1,0,1], (self.N+1, self.N+1),'lil')
        D[0, :2] = -2, 2
        D[-1, -2:] = 2, -2
        return D

    def ue(self, mx, my, x, y, t):
        """Return the exact standing wave"""
        return sp.cos(mx*sp.pi*x)*sp.cos(my*sp.pi*y)*sp.cos(self.w*t)

    def apply_bcs(self,u=None):
        """Apply Neumann boundary conditions to solution vector"""
        return "BC already implemented in the new D2."

def test_convergence_wave2d(): #Mikael
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann(): #Mikael
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    solD = Wave2D()
    solN = Wave2D_Neumann()
    tol = 1E-12
    hD,errD = solD(N=10, Nt=10, cfl = 1/np.sqrt(2)) #by default, mx = my
    hN,errN = solN(N=10, Nt=10, cfl = 1/np.sqrt(2))
    assert abs(errD[-1]) < tol
    assert abs(errN[-1]) < tol
    
if __name__ == "__main__":
    sol = Wave2D()
    #dx, err = sol(10,2,cfl = 1/np.sqrt(2))
    r, E, h = sol.convergence_rates(mx=2, my=3)
    print(r, E)
    K = abs(r[-1]-2) < 1e-2  
    print("Test konvergens Dirichlet:", K)
    
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    K2= abs(r[-1]-2) < 0.05
    print("Test konvergens Neumann:", K2)
    
    solD = Wave2D()
    solN = Wave2D_Neumann()
    tol = 1E-12
    hD,errD = solD(N=8, Nt=10, cfl = 1/np.sqrt(2)) #by default, mx = my
    hN,errN = solN(N=8, Nt=10, cfl = 1/np.sqrt(2))
    print("err",errD)
    K3 = abs(errD[-1]) < tol
    print("Test l2-error 1:",K3)
    K4 = abs(errN[-1]) < tol
    print("Test l2-error 2:",K4)