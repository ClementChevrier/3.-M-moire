import numpy as np
from numba import njit, prange
from scipy.optimize import minimize

from .helper import *


#-----------------------------------------------------------------------------------------------------"Volatilité"-----------------------------------------------------------------------------------------------------#
class VISurface(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._spline_equation = np.array([np.array(CubicSpline(pd.DataFrame(self.loc[mat, :])).calculate_spline_coefficients()) for mat in self.index])

    def get_vol(self, Maturité: float, Strike: float) -> float:
        """
        Retrieves the volatility for the given maturity and strike.

        Parameters
        ----------
        Maturité : float
            Maturity for which the volatility is to be retrieved.
        Strike : float
            Strike for which the volatility is to be retrieved.

        Returns
        -------
        float
            Volatility for the given maturity and strike.

        Notes
        -----
        If volatility data is stored as a single float value, it returns that value.
        If volatility data is stored as a VISurface object, it retrieves volatility from the surface.
        """
        if Maturité in self.index and Strike in self.columns:                
            return self.loc[Maturité, Strike]
        if Maturité in self.index:
            return self._interpolate_VI_from_Smile(Strike, Maturité)
        if Strike in self.columns:
            return self._interpolate_VI_from_Term_struc(Strike, Maturité)
        return self._interpolate_VI_from_Smile_n_Term(Strike, Maturité)

    def _interpolate_VI_from_Smile(self, K: int|float, mat: int|float) -> float:
        spline_number = self._get_strike_bucket(K)
        mat_number = list(self.index).index(mat)
        if spline_number == -1: K, spline_number = self.columns[0], 0
        if K > self.columns[-1]: K = self.columns[-1]
        return calculate_spline(K, spline_number, self._spline_equation[mat_number])

    def _interpolate_VI_from_Term_struc(self, K: int|float, mat: int|float) -> float:
        """
        Interpolate une VI via un interpolation linéaire sur la variance lorsque la maturité n'est pas observable sur le marché.

        Parameters
        ----------
        K : int | float
            Strike pour lequel on souhaite connaitre la VI
        mat : int | float
            Maturité pour laquel on souhaite connaitre la VI

        Returns
        -------
        float
            VI interpolé de la term_structure
        """
        mat1, mat2 = self._get_maturity_bucket(mat, True)[1]
        VI1 = self.loc[mat2, K] if mat1 == 0 else self.loc[mat1, K]
        VI2 = self.loc[mat1, K] if mat2 == 0 else self.loc[mat2, K]
        return np.sqrt(interpolation_lineaire([mat1, VI1**2], [mat2, VI2**2], mat))

    def _interpolate_VI_from_Smile_n_Term(self, K: int|float, mat: int|float) -> float:
        """
        Interpolate la VI lorsque ni la maturité, ni le strike ne sont observable sur le marché.
        Sélectionne les 2 smiles connus pour la maturité avant et après celle recherchée. 
        Puis interpolate les 2 VI pour les 2 smiles sélectionnés.
        On interpole linéairement entre les 2 VI.

        Parameters
        ----------
        K : int | float
            Strike pour lequel on souhaite connaître la VI
        mat : int | float
            Maturité pour laquelle on souhaite connaître la VI

        Returns
        -------
        float
            Return la VI.
        """
        mat1, mat2 = self._get_maturity_bucket(mat, True)[1]
        if mat1 == 0: # je retourne la valeur pour la maturité la plus proche ainsi vol constante pour les mat courtes
            return self._interpolate_VI_from_Smile(K, mat2)
        else:
            VI1 = self._interpolate_VI_from_Smile(K, mat1)
            VI2 = self._interpolate_VI_from_Smile(K, mat2)
            return np.sqrt(interpolation_lineaire([mat1, VI1**2], [mat2, VI2**2], mat))
    
    def _get_closest_mat(self, mat: int|float) -> int:
        return self.index[self.index.get_indexer([mat], method='nearest')[0]]
    
    def _get_closest_strike(self, strike: int|float) -> int:
        return self.columns[self.columns.get_indexer([strike], method='nearest')[0]]
    
    def _get_strike_bucket(self, strike: int | float, get_bucket:bool=False) -> int:
        """
        Determines the bucket index for a given strike.

        Parameters
        ----------
        strike : int | float
            Strike value for which the bucket index is to be determined.
        get_bucket : bool, optional
            If True, returns the bucket index along with the strike range, by default False.

        Returns
        -------
        int
            Bucket index corresponding to the given strike.

        Raises
        ------
        ValueError
            If the strike value exceeds the observed range of strikes.
        """
        for i, K in enumerate(self.columns):
            if strike < K:
                if i == 0:
                    return i-1 if not get_bucket else (i,[0,K])
                else:
                    return i-1 if not get_bucket else (i-1,[last_K, K])
            last_K = K
        return i-1 if not get_bucket else(i-1,[last_K,0])

    def _get_maturity_bucket(self, Maturity: int | float, get_bucket:bool=False) -> int:
        """
        Determines the bucket index for a given maturity.

        Parameters
        ----------
        Maturity : int | float
            Maturity value for which the bucket index is to be determined.
        get_bucket : bool, optional
            If True, returns the bucket index along with the maturity range, by default False.

        Returns
        -------
        int
            Bucket index corresponding to the given maturity.

        Raises
        ------
        ValueError
            If the maturity value exceeds the observed range of maturities.
        """
        for i, mat in enumerate(self.index):
            if Maturity < mat:
                if i == 0:
                    return i if not get_bucket else(i,[0, mat])
                else:
                    return i if not get_bucket else (i,[last_mat, mat])
            last_mat = mat
        return i if not get_bucket else (i,[last_mat, 0]) #ainsi si jamais une des mes bornes et 0 je sais que je suis en dehors de ma nap de Vol
        


#-----------------------------------------------------------------------------------------------------"Market Data"----------------------------------------------------------------------------------------------------#
class MarketData:
    def __init__(self, Spot: float, Vol: float | VISurface, Taux: float | list[str, float], Div, Date: np.datetime64 = np.datetime64('today')) -> None:
        """
        Objet pour représenter un actif.

        Paramètres
        ----------
        Spot : float
            La valeur actuelle de l'actif.
        Vol : float
            La volatilité de l'actif en %.
        Taux : float
            Le taux d'intérêt de l'actif en %.
        Div : float
            Le taux de dividende de l'actif en %.
        Date : np.datetime64, optionnel
            La date pour les données de marché (par défaut, la date d'aujourd'hui).

        Méthodes
        ----------
        get_rate_curve(self, Maturité: float) -> float:
            Renvoie le taux pour une maturité spécifique.

        get_div_curve(self, Maturité: float) -> float:
            Renvoie les dividendes pour une maturité spécifique.

        get_vol(self, Maturité: float, Vol: float) -> float:
            Renvoie la volatilité implicite du sous-jacent.
        """
        self.Spot = float(Spot)
        self.Taux = Taux
        self.Div = Div
        self.Vol = Vol
        self.Date = Date

    #----------------------------------------------"Setting Up Variable"-----------------------------------------------#    
    @property
    def Taux(self) -> float | list[str, float]:
        return self._Taux
    @Taux.setter
    def Taux(self, value: float | list[str, float] | tuple | np.ndarray) -> float | list[str, float]:
        """
        Permet d'uniformiser le format de la courbe de taux afin que ce soit toujours
        une liste (important car minimize de spicy ne fonctionne pas avec des tupples)
        """
        if isinstance(value, float): self._Taux = value
        elif isinstance(value, (tuple, np.ndarray, list)): self._Taux = list(value)
        else: raise TypeError(f"Format de données des taux d'intérêt non reconnu: {type(value)}")
        
    @property 
    def Vol(self) -> float | VISurface:
        return self._Vol
    @Vol.setter
    def Vol(self, value: float | pd.DataFrame) -> float | VISurface:
        """
        Sets the volatility surface with the correct format.

        Parameters
        ----------
        value : float | pd.DataFrame
            Volatility surface data. Can be either a float or a pandas DataFrame.

        Raises
        ------
        TypeError
            If the format of the volatility surface data is not recognized.
        """
        if isinstance(value, float):
            self._Vol = value
            self.Vol_Local = None
        elif isinstance(value, pd.DataFrame):
            if value.index[0] < 1:
                value.index = [int(mat * 252) for mat in value.index]
            value.columns = [float(strike)/100 * self.Spot for strike in value.columns]
            self._Vol = VISurface(value.copy())
            self.Vol_Local = self._get_local_vol_surface()
        else: raise TypeError(f"Format de données de VI non reconnu : {type(value)}")
    
    @property
    def Div(self) -> float:
        return self._Div
    @Div.setter
    def Div(self, value: float | list[str, float] | tuple | np.ndarray) -> float | list[str, float]:
        if isinstance(value, float): self._Div = value
        elif isinstance(value, (tuple, np.ndarray, list)): self._Div = list(value)
        else: TypeError(f"Format de données de div non reconnu : {type(value)}")
        
    def _get_local_vol_surface(self):
        df_local = pd.DataFrame(index=self.Vol.index, columns=self.Vol.columns)
        for M in self.Vol.index[1:]:
            T = M/365
            r = self.get_rate(T)
            q = self.get_div(T)
            prev_M = self.Vol.index[self.Vol.index.get_loc(M) - 1]
            for K in self.Vol.columns[1:-1]:
                sigma = self.Vol.get_vol(M,K)
                d = (np.log(self.Spot/K) + ((r-q)+(sigma**2)/2)*T) / (sigma*np.sqrt(T))
                prev_K = self.Vol.columns[self.Vol.columns.get_loc(K) - 1]
                next_K = self.Vol.columns[self.Vol.columns.get_loc(K) + 1]

                #Différence Finit
                partial_T = (self.Vol.get_vol(prev_M,K) - sigma) / ((M-prev_M)/365)
                partial_K = (self.Vol.get_vol(M,prev_K) - sigma) / (K-prev_K)
                second_partial_K = (self.Vol.get_vol(M,prev_K) - 2*sigma + self.Vol.get_vol(M,next_K)) / ((next_K-K) * (K-prev_K))

                #Formule
                numérateur= sigma**2 + 2*sigma*T*(partial_T + (r-q)*K*partial_K)
                dénominateur1 = (1+K*d*partial_K*np.sqrt(T))**2
                dénominateur2 = sigma * (K**2)*T*((second_partial_K - d*partial_K)**2)*np.sqrt(T)
                df_local.loc[M,K] = np.sqrt(numérateur / (dénominateur1+dénominateur2))
        self.Vol_Local = VISurface(df_local.iloc[1:,1:-1])
        return self.Vol_Local
    
        
    def __str__(self) -> str:
        formatted_Spot = int(self.Spot) if self.Spot.is_integer() else self.Spot
        formatted_Vol = f"{self.get_vol(365,100) * 100:.0f}%" if (self.get_vol(365,100)*100).is_integer() else f"{self.get_vol(365,100) * 100:.2f}%"
        formatted_Div = f"{self.get_div(1) * 100:.0f}%" if (self.get_div(1)*100).is_integer() else f"{self.get_div(1) * 100:.2f}%"
        formatted_Taux = f"{self.get_rate(1) * 100:.0f}%" if (self.get_rate(1)*100).is_integer() else f"{self.get_rate(1) * 100:.2f}%"
        return f"Le spot est de {formatted_Spot}€ la volatilité est de {formatted_Vol}, le taux de dividende est de {formatted_Div} et le taux sans risque est de {formatted_Taux}."
    
    

    #--------------------------------------------"Acces to timedepend data"--------------------------------------------#
    def get_rate(self, Maturité: float) -> float:
        """
        Returns the interest rate for the given maturity.

        Parameters
        ----------
        Maturité : float
            Maturity for which the interest rate is to be calculated.

        Returns
        -------
        float
            Interest rate for the given maturity.

        Raises
        ------
        ValueError
            If the interest rates have not been specified.
        """
        if isinstance(self.Taux, float): return self.Taux
        elif isinstance(self.Taux, list): return NNS(get_paramNNS(self.Taux), Maturité)
        else: raise TypeError(f"Les taux d'interets n'ont pas été renseigné ou le type de la courbe de taux :{type(self.Taux)} n'est pas surpporté")

    def get_vol(self, Maturité:float, Strike:float)->float:
        if isinstance(self.Vol, float): return self.Vol
        elif isinstance(self.Vol, VISurface): return self.Vol.get_vol(Maturité,Strike)
        else: raise TypeError(f"La volatilité n'a pas été renseigné ou le type de la nap de vol :{type(self.Vol)} n'est pas surpporté")

    def get_div(self, Maturité: float) -> float:
        if isinstance(self.Div, float): return self.Div
        elif isinstance(self.Div, list): return NNS(get_paramNNS(self.Div), Maturité)
        else: raise TypeError(f"Le div yield n'a pas été renseigné ou le type du div yield :{type(self.Div)} n'est pas surpporté")



    #---------------------------------------------------"Proba Spot"---------------------------------------------------#
    def get_proba_spot(self, ST: float, T:float) -> float:
        """
        Calcul la probabilité que le spot finisse au dessus d'une certaine valeur.

        Parameters
        ----------
        Underlying : MarketData
            Sous-jacent dont nous souhaitons connaitre la probabilité.
        Spot : float
            Valeur future du spot dont nous souhaitons connaître la probabilité.
        Maturité : float
            Temps, en année, pour lequel nous souhaitons connaître la probabilité.

        Returns
        -------
        float
            La probabilité
        """
        if T <= 1:
            x = (np.log(ST/self.Spot) - (self.get_rate(T) - self.get_div(T) - (self.get_vol(T,100)**2)/2)*T)/(self.get_vol(T,100)*np.sqrt(T))
            Proba = 1-norm_cdf(x)
        else:
            Proba = self.compute_probability_for_i_greater_than_1(ST, T)
        return Proba

    def compute_probability_for_i_greater_than_1(self, ST: float, T: float) -> float:
        raise NotImplementedError("W.I.P")
        # i = int(T)  # Récupère la partie entière de T
        # L = np.zeros(i)  # Initialise le vecteur L
        # U = np.zeros(i)  # Initialise le vecteur U
        # for t in range(1, i+1):  # Débute à 1
        #     L[t-1] = -np.inf if t < i else np.log(ST/self.Spot) - (self.get_rate(t) - self.get_div(T) - (self.get_vol(T,self.Spot)**2)/2)*t / (self.get_vol(T,self.Spot))
        #     U[t-1] = np.inf if t >= i else np.log(ST/self.Spot) - (self.get_rate(t) - self.get_div(T) - (self.get_vol(T,self.Spot)**2)/2)*t / (self.get_vol(T,self.Spot))
        # Proba = self.multivariate_normal_cdf(L, U)
        # return Proba

    def multivariate_normal_cdf(self, L, U):
        Liste_Proba = np.zeros(len(L))
        proba = 1

        for k in range(1,len(L)):
            std_dev_k = np.sqrt(k)
            proba_k = norm_cdf(U[k-1], std=std_dev_k) - norm_cdf(L[k-1], std=std_dev_k)

            # Mise à jour de la probabilité totale en multipliant par la probabilité de la variable aléatoire k+1
            proba *= proba_k
            Liste_Proba[k] = proba

        return proba

    def get_proba_spot_nodiv(self, ST: float, T:float) -> float:
        x = (np.log(ST/self.Spot) - (self.get_rate(T) -  (self.Vol**2)/2)*T)/(self.Vol*np.sqrt(T))
        Proba = 1-norm_cdf(x)
        return Proba     






#------------------------------------------------------------------------------------------------"Simulation MonteCarlo"-----------------------------------------------------------------------------------------------#
class MonteCarlo:
    def __init__(self, Udl: MarketData, T: float, NB_Pas: int, NB_Simu: int) -> None:
        """
        Classe pour effectuer des simulations de Monte Carlo sur un actif.
        Les Simulations sont multi-threadé afin d'aller le plus vite possible. 
        J'ai choisit d'utiliser une classe car c'est plus simple de manipuler
        le module njit avec une classe et ainsi de multi-threadé.

        Paramètres
        ----------
        Udl : MarketData
            Les données de marché nécessaires pour les simulations.
        T : float
            La maturité.
        NB_Pas : int
            Le nombre de pas pour la simulation.
        NB_Simu : int
            Le nombre de simulations.
        Seed : int
            Si spécifié alors la valeur est attribué comme seed du random generator

        Méthodes
        ----------
        Stantard(self) -> np.ndarray:
            Effectue une simulation Monte Carlo standard/

        RV_Correl(self) -> np.ndarray:
            Effectue une simulation Monte Carlo avec des variables aléatoires corrélées négativement.
        """
        self.S = float(Udl.Spot)
        self.r = float(Udl.get_rate(T))
        self.q = float(Udl.get_div(T))
        self.T = float(T)
        self.N = int(NB_Pas)
        self.M = int(NB_Simu)
        self.Vol = float(Udl.get_vol(self.T,self.S))
        self.Vol_Local = Udl.Vol_Local
           
    def vol_standard(self) -> np.ndarray:
        """
        Effectue une simulation Monte Carlo avec Vol B&S.

        Returns
        -------
        np.ndarray
            Array représentant les résultats de la simulation.
        """
        return _standard_vol(self.S, self.Vol, self.r, self.q, self.T, self.N, self.M)

    def RV_Correl(self) -> np.ndarray:
        """
        Effectue une simulation Monte Carlo avec des variables aléatoires corrélées négativement. 
        Le prix du produit doit être le même peut-importe mon mouvement brownien. 
        Ainsi si on génère une second spot qui a correl de mouvement brownien de -1,
        nous avons besoin de moins de simulation.  De plus cela permet de réduire la variance de mon modèle 
        et donc par la même occasion le Stantard Error de mon prix.

        Returns
        -------
        np.ndarray
            Array représentant les résultats de la simulation.
        """
        return _RV_Correl(self.S, self.Vol, self.r, self.q, self.T, self.N, self.M)

    def vol_local(self) -> np.ndarray:
        """
        Effectue une simulation Monte Carlo avec la Vol Local.

        Returns
        -------
        np.ndarray
            Array représentant les résultats de la simulation.

        Raises
        ------
        TypeError
            Si la vol local n'est pas disponible alors il retourne une erreur.
        """
        if isinstance(self.Vol_Local, VISurface):
            return _local_vol(self.S, self.r, self.q, self.T, self.N, self.M, np.array(self.Vol_Local.index), np.array(self.Vol_Local.columns), np.array(self.Vol_Local._spline_equation))
        else:
            raise TypeError("Nous avons besoin d'une nap de VI afin de réaliser une simulation en Volatilité Local.")

    def vol_sto(self, Correl: float, Variance: float, Kappa:float, Theta:float, vol_vol:float) -> np.ndarray:
        """
        Effectue une simulation Monte Carlo avec la Vol Sto.

        Parameters
        ----------
        Correl : float
            Correlation entre les 2 mouvements brownien
        Variance : float
            La variance intial
        Kappa : float
            La vitesse de retour à la variance moyenne
        Theta : float
            La variance moyenne
        vol_vol : float
            La Vol de Vol

        Returns
        -------
        np.ndarray
            Array représentant les résultats de la simulation.
        """
        return _vol_sto(self.S, self.r, self.q, self.T, self.N, self.M, Correl, Variance, Kappa, Theta, vol_vol)

@njit(parallel=True, fastmath=True, nogil=True)
def _standard_vol(S: float, Vol: float, r: float, q: float, T: float, N: int, M: int) -> np.ndarray:
    drift = ((r - q) - (Vol**2) / 2) * (1 / N)
    diffusion = Vol * np.sqrt(1 / N)
    Longeur = int((N * T) + 1)
    array = np.zeros((M, Longeur), dtype=float)
    for i in prange(M):
        array[i, 0] = S
        for J in range(1, Longeur):
            array[i, J] = array[i, J - 1] * np.exp(drift + np.random.normal() * diffusion)
    return array

@njit(parallel=True, fastmath=True, nogil=True)
def _RV_Correl(S: float, Vol: float, r: float, q: float, T: float, N: int, M: int) -> np.ndarray:
    M = int(M/2)
    drift = ((r-q) - (Vol**2)/2)*(1/N)
    diffusion = Vol*np.sqrt(1/N)
    Longeur = int((N * T) + 1)
    array3D = np.zeros((M, Longeur, 2), dtype=float)
    for i in prange(M):
        array3D[i, 0, 0] = S
        array3D[i, 0, 1] = S
        for J in range(1,Longeur):
            mvt_brownien = np.random.normal()
            array3D[i, J, 0] = array3D[i, J - 1, 0] * np.exp(drift + mvt_brownien * diffusion)
            array3D[i, J, 1] = array3D[i, J - 1, 1] * np.exp(drift - mvt_brownien * diffusion)
    return array3D

@njit(parallel=True, fastmath=True, nogil=True)
def _local_vol(S: float, r: float, q: float, T: float, N: int, M: int, index:np.ndarray, columns:np.ndarray, equa_spline:np.ndarray) -> np.ndarray:
    Longueur = int((N * T) + 1)
    dt = 1 / N
    array = np.zeros((M, Longueur), dtype=float)
    for i in prange(M):
        array[i, 0] = S
        for J in range(1, Longueur):
        #je dois récupéré la vol pour le spot précédent et la vol précedent
            for iloc_mat, mat in enumerate(index):
                if J == mat:
                    spline = equa_spline[iloc_mat]
                    for iloc_k, k in enumerate(columns):
                        if array[i, J-1] <= k: break
                    if iloc_k == 0:
                        coef = spline[iloc_k]
                        Vol = coef[0] * k ** 3 + coef[1] * k ** 2 + coef[2] * k + coef[3]
                    else:
                        iloc_k -= 1
                        coef = spline[iloc_k]
                        if iloc_k == len(spline)-1:
                            Vol = coef[0] * k ** 3 + coef[1] * k ** 2 + coef[2] * k + coef[3]
                        else:
                            Vol = coef[0] * array[i, J-1] ** 3 + coef[1] * array[i, J-1] ** 2 + coef[2] * array[i, J-1] + coef[3]
                    break
                elif J < mat:
                    # I need to calculate the spline above and below to interpolate linearly.
                    if iloc_mat != 0:
                        mat_basse = iloc_mat - 1
                        mat_hautte = iloc_mat
                        # Calculate the two smiles
                        spline_bas = equa_spline[mat_basse]
                        spline_haut = equa_spline[mat_hautte]
                        for iloc_k, k in enumerate(columns):
                            if array[i, J-1] == k or array[i, J-1] < k: break
                        if iloc_k == 0:
                            VI1 = spline_bas[iloc_k][0] * k ** 3 + spline_bas[iloc_k][1] * k ** 2 + spline_bas[iloc_k][2] * k + spline_bas[iloc_k][3]
                            VI2 = spline_haut[iloc_k][0] * k ** 3 + spline_haut[iloc_k][1] * k ** 2 + spline_haut[iloc_k][2] * k + spline_haut[iloc_k][3]
                        else:
                            iloc_k -= 1
                            if iloc_k == len(spline_bas)-1:
                                VI1 = spline_bas[iloc_k][0] * k ** 3 + spline_bas[iloc_k][1] * k ** 2 + spline_bas[iloc_k][2] * k + spline_bas[iloc_k][3]
                                VI2 = spline_haut[iloc_k][0] * k ** 3 + spline_haut[iloc_k][1] * k ** 2 + spline_haut[iloc_k][2] * k + spline_haut[iloc_k][3]
                            else:
                                VI1 = spline_bas[iloc_k][0] * array[i, J-1] ** 3 + spline_bas[iloc_k][1] * array[i, J-1] ** 2 + spline_bas[iloc_k][2] * array[i, J-1] + spline_bas[iloc_k][3]
                                VI2 = spline_haut[iloc_k][0] * array[i, J-1] ** 3 + spline_haut[iloc_k][1] * array[i, J-1] ** 2 + spline_haut[iloc_k][2] * array[i, J-1] + spline_haut[iloc_k][3]
                        Vol = np.sqrt(((VI2 ** 2 - VI1 ** 2) / (index[mat_hautte] - index[mat_basse])) * (J - index[mat_basse]) + VI1 ** 2)
                        break
                    else:
                        # Calculate the smile
                        spline = equa_spline[iloc_mat]
                        for iloc_k, k in enumerate(columns):
                            if array[i, J-1] <= k: break
                        if iloc_k == 0 :
                            coef = spline[iloc_k]
                            Vol = coef[0] * k ** 3 + coef[1] * k ** 2 + coef[2] * k + coef[3]
                        else:
                            iloc_k -= 1
                            coef = spline[iloc_k]
                            if iloc_k == len(spline)-1:
                                Vol = coef[0] * k ** 3 + coef[1] * k ** 2 + coef[2] * k + coef[3]
                            else:
                                Vol = coef[0] * array[i, J-1] ** 3 + coef[1] * array[i, J-1] ** 2 + coef[2] * array[i, J-1] + coef[3]
                        break
                #une fois que je sors de l'univers observable je considère ma vol comme constant.
            #maintenant que la vol est récupéré je continue comme avant
            drift = ((r - q) - (Vol**2) / 2) * (dt)
            diffusion = Vol * np.sqrt(dt)
            array[i, J] = array[i, J - 1] * np.exp(drift + np.random.normal() * diffusion)
    return array

@njit(parallel=True, fastmath=True, nogil=True)
def _vol_sto(S: float, r: float, q: float, T: float, N: int, M: int, rho:float, v0:float, kappa:float, theta:float, vol_vol:float) -> np.ndarray:
	Longeur = int((N * T) + 1)
	dt = 1/N
	s = np.zeros((M, Longeur), dtype=float)
	v = np.zeros((M, Longeur), dtype=float)

	for i in prange(M):
		s[i, 0] = S
		v[i, 0] = v0
		for J in range(1, Longeur):
			drift = ((r - q) - v[i, J-1]/2) * dt
			diffusion = np.sqrt(v[i, J-1]*dt)
			
			z1, z2 = np.random.normal(), np.random.normal()
			z2 = z1*rho + np.sqrt(1 - rho**2)*z2
			
			s[i, J] = s[i, J - 1] * np.exp(drift + diffusion*z1)
			v[i, J] = np.maximum(v[i, J-1] + kappa*(theta - v[i, J-1])*dt + vol_vol*np.sqrt(v[i, J-1]*dt)*z2,0)
	return s




#--------------------------------------------------------------------------------------------------"Taux Sans-Risque"--------------------------------------------------------------------------------------------------#
def NNS(param: list, t: np.ndarray | float ) -> np.ndarray | float:
    """
    Calcul les taux selon le modèle de Nelson-Siegel-Svensson. Modèle sert à interpoler
    les points manquants. Il peut savoir soit une seul maturité soit un array de maturité.   

    Parameters
    ----------
    param : list
        Liste contenant les facteurs nécessaires au fonctionnement du modèle:
        B0 = niveau absolue des taux
        B1 = la pente des taux
        B2 = le smile du taux
        B3 = si j'ai une bausse dans ma courbe de taux
        λ = 
        κ = 
    t : np.ndarray | float
        Maturité du taux voulu

    Returns
    -------
    np.ndarray | float
        Retourne un array ou un valeur suivant l'input

    Raises
    ------
    ValueError
        Si la maturité est nul alors une error est levé.
    """
    if np.any(t == 0):
        raise ValueError("Maturity must be non-zero")

    B0, B1, B2, B3, λ, κ = param
    F1 = (1 - np.exp(-t/λ)) / (t/λ)
    F2 = ((1 - np.exp(-t/λ)) / (t/λ)) - np.exp(-t/λ)
    F3 = ((1 - np.exp(-t/κ)) / (t/κ)) - np.exp(-t/κ)
    return B0 + B1*F1 + B2*F2 + B3*F3

def _least_square_NNS(param: list, Data: list) -> float:
    """
    Fonction servant à retourner la distance au carré entre la courbe calculé avec les facteurs 
    et les vrais données.

    Parameters
    ----------
    param : list
        Liste des paramètres nécessaires aux fonctionnement du modèle.
    Data : list
        Donnée du monde réel, organisé comme suit Maturité, Taux

    Returns
    -------
    float
        Retourne la somme des écarts au carrés.
    """
    Maturité, Taux = Data
    estimated_rates = []
    for t in Maturité:
        rate_esti = NNS(param, t)
        estimated_rates.append(rate_esti)
    estimated_rates = np.array(estimated_rates)
    Taux = np.array(Taux)
    return np.sum((estimated_rates - Taux)**2)

@cache
def get_paramNNS(Data:list) -> list:
    """
    Fonctionnement retournant la liste des paramètres du modèle NNS
    fittant avec les données renseigner.

    Parameters
    ----------
    Data : list
        Les données avec lesquelles nous souhaitons interpollé la courbe.

    Returns
    -------
    list
        Liste des paramètres
    """
    Liste_Ini = [0.0005, -0.0001, -0.0252, 0.0247, 2, 5]
    # params = gradient_descent(_least_square_NNS, Liste_Ini, Data)
    result = minimize(_least_square_NNS, Liste_Ini, args=(Data))
    return list(result.x)