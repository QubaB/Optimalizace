import numpy as np
from scipy.integrate import solve_ivp

class Dipol:
    mu0=4*np.pi*1e-7
    muOver4pi=1e-7  # mu0/4pi
    name=""

    def __init__(self,r,m,m_a):
        """
            r = poloha
            m = dipolovy moment
            m_a = smerovy uhel
        """
        self._m=m   
        self._m_a=m_a
        self._r=r
        self.__set_mVec() # aktualizuj vektor m

    def __set_mVec(self):
        self._mvec=np.array([
            self._m*np.cos(self._m_a),
            self._m*np.sin(self._m_a)
        ])


    def setDipol_ma(self,m_a):
        """
            Nastav dipolovy moment
        """
        self._m_a=m_a
        self.__set_mVec()

    def getDipol_m(self):
        """
            vrat dipolovy moment
        """
        return(self._m)


    def getDipol_B(self,x):
        """
            Magneticka indukce v miste x
        """
        R=x-self._r
        Rsize = np.linalg.norm(R)
        mdotR = np.sum(R*self._mvec)         # m*R
        term1 = (3.0 * R * mdotR / Rsize**5)
        term2 = self._mvec / Rsize**3
        B = Dipol.muOver4pi * (term1 - term2)
        return(B)
    
    def getDipol_Silocara(self,x0,max_l):
        def fun(t,x):
            B=self.getDipol_B(x)
            B=B/np.linalg.norm(B)  # normalize to 1
            return B
        
        sol = solve_ivp(fun, [0, max_l], x0, method='RK45', t_eval=np.linspace(0, max_l, 100),rtol=1e-12,atol=1e-7)
        return([sol.y[0],sol.y[1]])



    
