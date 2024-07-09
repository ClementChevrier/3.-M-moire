import numpy as np


from ModuleFinance.helper import *
from ModuleFinance.finfunc import *



#========================================================================================"VANILLE"========================================================================================#
class CallOption:
    def __init__(self, Udl: MarketData, K: float, T: float) -> None:
        """
        Objet pour représenter une option d'achat (Call Option) sur un actif.

        Paramètres
        ----------
        Uderlying : MarketData
            Les données de marché de l'actif sous-jacent.
        Strike : float
            Le prix d'exercice (strike price).
        Maturité : float
            La durée en année jusqu'à l'expiration de l'option.

        Méthodes
        ----------
        update() -> bool:
            Vérifie si les paramètres de l'option ont été modifiés.
        
        price() -> float:
            Calcule le prix de l'option à l'aide du modèle de Black-Scholes.

        price_mc(pas: int = 252, simu: int = 262143, SE: bool = False) -> tuple[float, float]:
            Calcule le prix de l'option à l'aide de simulations de Monte Carlo.
        
        `Grecques`_Cash():
            Méthodes disponibles avec les principales grecques 
            permettant de connaître la sensi en Cash.
            
        `Grecques`_pct():
            Méthodes disponibles avec les principales grecques 
            permettant de connaître la sensi en pourcentage.
        """
        self.Udl = Udl
        self.K = float(K)
        self.T = float(T)
        self.Param = [self.Udl.Spot, self.Udl.Vol, self.Udl.Taux, self.Udl.Div, self.K, self.T]
        
    def __str__(self) -> str:
        formatted_S = int(self.Udl.Spot) if self.Udl.Spot.is_integer() else self.Udl.Spot
        formatted_K = int(self.K) if self.K.is_integer() else self.K
        formatted_T = int(self.T) if self.T.is_integer() else self.T
        formatted_Vol = f"{self.Udl.get_vol(self.T,self.K) * 100:.0f}%" if (self.Udl.get_vol(self.T,self.K)*100).is_integer() else f"{self.Udl.get_vol(self.T,self.K) * 100:.2f}%"
        formatted_q = f"{self.Udl.get_div(self.T) * 100:.0f}%" if (self.Udl.get_div(self.T)*100).is_integer() else f"{self.Udl.get_div(self.T) * 100:.2f}%"
        formatted_r = f"{self.Udl.get_rate(self.T) * 100:.0f}%" if (self.Udl.get_rate(self.T)*100).is_integer() else f"{self.Udl.get_rate(self.T) * 100:.2f}%"
        return f"Call Option de Strike {formatted_K}€ et Maturité {formatted_T} an(s), pour un spot de {formatted_S}€, une volatilité de {formatted_Vol}, un taux de dividende de {formatted_q} et pour un taux sans risque de {formatted_r}."
        
    def copy(self):
        new_Udl = MarketData(self.Udl.Spot, self.Udl.Vol, self.Udl.Taux, self.Udl.Div, self.Udl.Date) #ligne obligatoire pour ne pas modifer la marketdata dans le delta par exemple ou autre car sinon ça modifie la market data de base à voir pourquoi
        return self.__class__(new_Udl, self.K, self.T)
       
    def update(self) -> bool:
        """
        Vérifie si les caractéristique de l'option ont été modifiées depuis sa création.

        Returns
        -------
        bool
            Return True si l'option a été modifié et met les données à jour.
        """
        return [self.Udl.Spot, self.Udl.Vol, self.Udl.Taux, self.Udl.Div, self.K, self.T] != self.Param
    


    #---------------------------------------------------------------"Divers"---------------------------------------------------------------# 
    def payoff(self, ST) -> np.ndarray:
        return np.maximum(0 , ST - self.K)
    


    #-----------------------------------------------------------"Black & Scholes"----------------------------------------------------------#
    def calculate_d1_d2(self)-> tuple[float, float]:
        Vol = self.Udl.get_vol(self.T,self.K)
        d1 = (np.log(self.Udl.Spot/self.K) + (self.Udl.get_rate(self.T)-self.Udl.get_div(self.T) + ((Vol**2)/2))*self.T) / (Vol*np.sqrt(self.T))
        d2 = d1 - Vol*np.sqrt(self.T)
        return d1, d2
    
    def price(self) -> float:
        d1, d2 = self.calculate_d1_d2()
        return self.Udl.Spot* norm_cdf(d1)*np.exp(-self.Udl.get_div(self.T)*self.T) - np.exp((-self.Udl.get_rate(self.T))*self.T)*self.K*norm_cdf(d2)

    def delta(self) -> float:
        """
        Permet de calculer le Delta Cash de l'option.
            
        See Also
        --------
        CallOption.delta_pct() : Retourne le delta pour une variation de 1%.
        """
        Option_Up = self.copy()
        Option_Up.Udl.Spot *= 1.01
        Option_Down = self.copy()
        Option_Down.Udl.Spot *= 0.99
        return (Option_Up.price() - Option_Down.price())/2

    def gamma(self) -> float:
        """
        Permet de calculer le Gamma Cash de l'option pour une variation de 1%.

        See Also
        --------
        CallOption.gamma_pct() : Retourne le gamma pour une variation de 1%.
        """
        Option_Up = self.copy()
        Option_Up.Udl.Spot *= 1.01
        Option_Down = self.copy()
        Option_Down.Udl.Spot *= 0.99
        return (Option_Up.price() - self.price()) + (Option_Down.price() - self.price())

    def vega(self) -> float:
        """
        Permet de calculer le Vega Cash de l'option pour 
        une variation de 1% de la volatilité en Absolu.

        See Also
        --------
        CallOption.vega_rel() : Retourne le gamma pour une 
        variation de 1% de la volatilité en Relatif.
        """
        Option_Up = self.copy()
        Option_Up.Udl.Vol = self.Udl.get_vol(self.T,self.K) + 0.01
        Option_Down = self.copy()
        Option_Down.Udl.Vol = self.Udl.get_vol(self.T,self.K) - 0.01
        return (Option_Up.price() - Option_Down.price())/2 
    def vega_rel(self) -> float:
        """
        Permet de calculer le Vega Cash de l'option pour 
        une variation de 1% de la volatilité en Relatif.

        See Also
        --------
        CallOption.vega() : Retourne le gamma pour une 
        variation de 1% de la volatilité en Absolu.
        """
        Option_Up = self.copy()
        Option_Up.Udl.Vol = self.Udl.get_vol(self.T,self.K) * 1.01
        Option_Down = self.copy()
        Option_Down.Udl.Vol = self.Udl.get_vol(self.T,self.K) * 0.99
        return (Option_Up.price() - Option_Down.price())/2 

    def rho(self) -> float:
        """
        Permet de calculer le Rho Cash de l'option pour
        une variation de 0.01% des taux en Absolu,
        soit 10bps.
        
        SANS DOUTE FAUX A CHECK
        -----------------------
        """
        Option_Up = self.copy()
        Option_Up.Udl.Taux = self.Udl.get_rate(self.T) + 0.001
        Option_Down = self.copy()
        Option_Down.Udl.Taux = self.Udl.get_rate(self.T) - 0.001
        return (Option_Up.price() - Option_Down.price())/2 * 10


    def theta(self) -> float:
        """
        Permet de Calculer le Theta pour une variation 
        de 1 jour.
        """
        Option_J1 = self.copy()
        Option_J1.T -= 1/252
        return (Option_J1.price() - self.price())
    
    
    
    #------------------------------------------------------------"MonteCarlo"------------------------------------------------------------#
    def price_mc(self, pas: int=252, simu: int=262143, SE: bool=False) -> tuple[float, float]:
        Last = MonteCarlo(self.Udl, self.T, pas, simu).Stantard()[:,-1]
        Gain = self.payoff(Last)
        PrixMC = np.exp(-self.Udl.get_rate(self.T)*self.T) * Gain.mean()
        SE = (np.sqrt((np.sum(Gain - PrixMC)**2)/(simu-1))) / (np.sqrt(simu))
        return (PrixMC, SE) if SE else PrixMC




class PutOption(CallOption):
    def __init__(self, MarketData: MarketData, K: float, T: float) -> None:
        """
        Objet pour représenter une option de Vente (Put Option) sur un actif.

        Paramètres
        ----------
        Uderlying : MarketData
            Les données de marché de l'actif sous-jacent.
        Strike : float
            Le prix d'exercice (strike price).
        Maturité : float
            La durée en année jusqu'à l'expiration de l'option.

        Méthodes
        ----------
        update() -> bool:
        Vérifie si les paramètres de l'option ont été modifié.
        
        price() -> float:
            Calcule le prix de l'option à l'aide du modèle de Black-Scholes.

        price_mc(pas: int = 252, simu: int = 262143, SE: bool = False) -> tuple[float, float]:
            Calcule le prix de l'option à l'aide de simulations de Monte Carlo.
        
        `Grecques`_Cash():
            Méthodes disponibles avec les principales grecques 
            permettant de connaître la sensi en Cash.
            
        `Grecques`_pct():
            Méthodes disponibles avec les principales grecques 
            permettant de connaître la sensi en pourcentage.
        """
        super().__init__(MarketData, K, T)
        
    def __str__(self) -> str:
        formatted_S = int(self.Udl.Spot) if self.Udl.Spot.is_integer() else self.Udl.Spot
        formatted_K = int(self.K) if self.K.is_integer() else self.K
        formatted_T = int(self.T) if self.T.is_integer() else self.T
        formatted_Vol = f"{self.Udl.get_vol(self.T,self.K) * 100:.0f}%" if (self.Udl.get_vol(self.T,self.K)*100).is_integer() else f"{self.Udl.get_vol(self.T,self.K) * 100:.2f}%"
        formatted_q = f"{self.Udl.get_div(self.T) * 100:.0f}%" if (self.Udl.get_div(self.T)*100).is_integer() else f"{self.Udl.get_div(self.T) * 100:.2f}%"
        formatted_r = f"{self.Udl.get_rate(self.T) * 100:.0f}%" if (self.Udl.get_rate(self.T)*100).is_integer() else f"{self.Udl.get_rate(self.T) * 100:.2f}%"
        return f"Put Option de Strike {formatted_K}€ et Maturité {formatted_T} an(s), pour un spot de {formatted_S}€, une volatilité de {formatted_Vol}, un taux de dividende de {formatted_q} et pour un taux sans risque de {formatted_r}."

    def payoff(self, ST) -> np.ndarray:
        return np.maximum(0,self.K - ST)
    
    def price(self) -> float:
        d1, d2 = self.calculate_d1_d2()
        return -self.Udl.Spot* norm_cdf(d1)*np.exp(-self.Udl.get_div(self.T)*self.T) + np.exp((-self.Udl.get_rate(self.T))*self.T)*self.K*norm_cdf(d2)




class CallSpread:
    def __init__(self, Udl: MarketData, K: int, T: float, Largeur: float, Coupon: float, Positionnement: str="Neutre"):
        self.Udl = Udl
        self.CallUp, self.CallDown = self._CreationCS(Udl,K,T,Largeur, Positionnement)
        self.NB_CS = Coupon/Largeur
    
    def _CreationCS(self,Udl: MarketData, Strike: int, T: float, Largeur: float, positionnement: str="Neutre"):
        Largeur = Largeur*100
        if positionnement == "Conservateur":
            StrikeUp = Strike
            StrikeDown = Strike - Largeur
        elif positionnement == "Neutre":
            StrikeUp = Strike + Largeur/2
            StrikeDown = Strike - Largeur/2    
        elif positionnement == "Agressif":
            StrikeUp = Strike + Largeur
            StrikeDown = Strike
        CallUp = CallOption(Udl, StrikeUp, T)
        CallDown = CallOption(Udl, StrikeDown, T)
        return CallUp, CallDown
    
    def _calculate_value(self, method_name: str, *args, **kwargs):
        Resultat = -getattr(self.CallUp, method_name)(*args, **kwargs) + getattr(self.CallDown, method_name)(*args, **kwargs)
        return Resultat * self.NB_CS
    
    def payoff(self, ST):
        return self._calculate_value("payoff", ST)
    
    def price(self):
        return self._calculate_value("price")
    
    def delta(self):
        return self._calculate_value("delta")
    
    def gamma(self):
        return self._calculate_value("gamma")

    def vega(self):
        return self._calculate_value("vega")

    def rho(self):
        return self._calculate_value("rho")

    def theta(self):
        return self._calculate_value("theta")
    
    def price_mc(self, *args):
        return self._calculate_value("price_mc", *args)
    

class Forward:
    def __init__(self, Udl: MarketData, T: float):
        self.Udl = Udl
        self.T = float(T)

    def price(self):
        return self.Udl.Spot * np.exp((self.Udl.get_rate(self.T) - self.Udl.get_div(self.T)) * (self.T))

    def delta(self):
        return np.exp((self.Udl.get_rate(self.T) - self.Udl.get_div(self.T)) * (self.T))
    
    def epsilon(self):
        raise NotImplementedError("W.I.P")
    
    

        






#========================================================================================"EXOTIC"========================================================================================#
class Autocall:
    def __init__(self, MarketData: MarketData, Coupon: float, B_Recall: float, B_Coupon: float, B_PDI: float,K: float, Obs: int, T: float, vol_model:str = "vol_standard") -> None:
        """
        Créer un Autocall avec les Paramètres spécifié.

        Paramètres :
        ------------
        MarketData : MarketData
            Les données de marché de l'actif sous-jacent.
        Coupon : float
            Le taux de coupon annuelle.
        B_Recall : float
            La barrière de rappel.
        B_Coupon : float
            La barrière de coupon.
        B_PDI : float
            La barrière du PDI.
        K : float
            Le strike du PDI.
        Obs : int
            La péridocité des coupons en mois (Obs=6 signifie qu'il y aurat des coupons tous les 6 mois.)
        T : float
            La maturité en année.
            
        Méthodes :
        -----------
        update() -> bool
            Vérifie si les paramètres de l'autocall ont été modifiés.

        obs_date() -> list
            Retourne les dates d'observation des coupons.

        `price`(simu: int = 262143, SE: bool = False,...) -> float:
            Plusieurs fonctions permettant de calculer le prix de l'Autocall
        
        `Grecques`(quick: bool = False) -> float
            Permet de calculer les sensis de l'Autocall
        """
        self.Udl = MarketData
        self.Coupon = Coupon
        self.B_Recall = B_Recall
        self.B_Coupon = B_Coupon
        self.B_PDI = float(B_PDI)
        self.K = float(K)
        self.Maturité = float(T)
        self._T = float(T)
        self.Obs = int(Obs)
        assert self.T % (self.Obs/12) == 0, "La périodicité des coupons doit matcher la maturité du produit"
        self.vol_model = str(vol_model)
        self.Param = [self.Udl.Spot, self.Udl.Vol, self.Udl.Taux, self.Udl.Div, self.K]
        self.MC = None
    
    @property
    def T(self) -> float:
        return self._T

    @T.setter
    def T(self, value: float) -> None:
        """
        Modifie la valeur de la maturité (T) en s'assurant qu'elle ne dépasse 
        pas la maturité définie à l'initialisation du produit.

        Paramètres :
        ------------
        value : float
            La nouvelle valeur de la maturité.
        """
        assert value <= self.Maturité, "La valeur de T ne peut pas être supérieure à la maturité définit lors de l'initialisation du produit."
        assert value > 0, "La valeur de T ne pas être égale ou inférieur à 0"
        self._T = value
        
    def __str__(self) -> str:
        Barrière_Recall = int(self.B_Recall) if self.B_Recall.is_integer() else self.B_Recall
        Barrière_Coupon = int(self.B_Coupon) if self.B_Coupon.is_integer() else self.B_Coupon
        Barrière_PDI = int(self.B_PDI) if self.B_PDI.is_integer() else self.B_PDI
        Maturité = self.T
        Coupon = f"{self.Coupon * (self.Obs/12) * 100:.0f}%" if (self.Coupon * (self.Obs/12)*100).is_integer() else f"{self.Coupon * (self.Obs/12) * 100:.2f}%"
        return f"Autocall de Maturité {Maturité} an(s), avec un coupon de {Coupon} tous les {self.Obs} mois, la barrière de recall est {Barrière_Recall}, la barrière de coupon est {Barrière_Coupon} et la barrière de PDI est {Barrière_PDI}"





    #--------------------------------------------------------------"Divers"--------------------------------------------------------------# 
    def copy(self) -> "Autocall":
        new_Udl = MarketData(self.Udl.Spot, self.Udl.Vol, self.Udl.Taux, self.Udl.Div, self.Udl.Date) #ligne obligatoire pour ne pas modifer la marketdata dans le delta par exemple ou autre car sinon ça modifie la market data de base
        return self.__class__(new_Udl, self.Coupon, self.B_Recall, self.B_Coupon, self.B_PDI, self.K, self.Obs, self.T)
    
    def update(self) -> bool:
        """
        Vérifie si les caractéristique de l'Autocall ont été modifiées depuis sa création.

        Returns
        -------
        bool
            Return True si l'Autocall a été modifié et met les données à jour.
        """
        return [self.Udl.Spot, self.Udl.Vol, self.Udl.Taux, self.Udl.Div, self.K] != self.Param
        
    def obs_date(self) -> list:
        """
        Calcule les dates d'observation pour les coupons.

        Returns:
        ---------
        list:
            Liste des dates d'observation et retoune les dates sous formes:
            dd/mm/YYYY.
        """
        Ajd = np.datetime64('today')
        Date_Obs = [Ajd]
        Last_Date = add_year(Ajd,self.T)
        while Date_Obs[-1] <= Last_Date:
            Next_Obs = add_month(Date_Obs[-1], self.Obs)
            Date_Obs.append(Next_Obs)
        #Sanity check    
        if Date_Obs[-1] != Last_Date:
            Date_Obs.pop(-1)
        return Date_Obs

    def CalculPayoff(self, MC_selected: np.ndarray) -> np.ndarray:
        """
        Calcule le prix de l'Autocall.

        Paramètres :
        ------------
        MC_selected : np.ndarray
            Matrice des simulations Monte Carlo.

        Returns:
        ---------
        np.ndarray:
            Array 1 colonne avec le gain du Produit pour chaque simulation.
        """
        Coupon = self.Coupon * (self.Obs/12)

        Recall_M = np.where(MC_selected>self.B_Recall, 1, 0)
        Recall_M[:, 0] = 0                                                                      #met un 0 pour car je ne peux pas recevoir de coupon ou être recall à l'initailisation
        Temp = np.argmax(Recall_M == 1, axis=1)                                                 #indique la position de mon premier 1 soit lorsque je suis recall 
        Maturity_V = np.where(Temp == 0, Recall_M.shape[1]-1, Temp)                             #si jamais je n'ai pas de 1 (c'est égale à 0) alors il est recall à maturité (Recall_M.shape[1]-1)
        Recall_M = np.where(np.arange(Recall_M.shape[1]) > Maturity_V[:, np.newaxis], 0, 1)     #permet de fill de 1 une fois que je suis au dessus de l'index determiné dans la ligne du dessus
        Mat_V = np.full(Recall_M.shape[1], np.arange(Recall_M.shape[1])*(self.Obs/12))[1:]      #créer un vecteur avec les dates de recall
        DC_Factor_V = np.insert(np.exp(-self.Udl.get_rate(Mat_V) * Mat_V),0,0)                  #convertit le vecteur de mat en discount factor

        #calcul Coupon
        Coupon_M = np.where(MC_selected>self.B_Coupon, Coupon * DC_Factor_V, 0)
        Coupon_M = np.where(Recall_M == 0, 0 , Coupon_M)
        Coupon_M[:, 0] = 0                                                                      #met un 0 pour car je ne peux pas recevoir de coupon ou être recall à l'initailisation
        Payoff_Coupon = np.sum(Coupon_M, axis=1)

        #calcul PDI + Capital
        Last = MC_selected[np.arange(len(MC_selected)), Maturity_V]                             #récupère le spot du recall (premier 1)
        Payoff_Capital = np.where(Last < self.B_PDI, Last/self.K, 1) * np.exp(-self.Udl.get_rate(Maturity_V * (self.Obs/12)) * (Maturity_V * (self.Obs/12)))
        return Payoff_Coupon + Payoff_Capital

        
        
        
    #------------------------------------------------------------"Price"------------------------------------------------------------#        
    def price(self,simu: int = 262143, SE: bool=False, *args, **kwargs) -> tuple[float, float] | float:
        """
        Calcule rapidement le prix du produit Autocall. Pour cela,
        le nombre de pas de mes simulations de Monte-Carlo est
        équivalent aux nombres de dates d'observation. Suivant le type de modèle
        choisit lors de l'initialisation du produit ou passé dans *args le modèle
        de diffustion sera différents

        Paramètres :
        ------------
        simu : int, optionnel
            Nombre de simulations de Monte Carlo. Par défaut, 262143.
        SE : bool, optionnel
            Indique si l'écart type doit être retourné. Par défaut, False.
        *args: str, optionnel
            "vol_local": la diffusion sera faite avec une vol local
            "vol_sto": la diffusion sera faite avec un vol sto. Il faut passer
            les arguments dans **kwargs.

        Returns:
        ---------
        tuple[float, float] | float:
            Tuple contenant le prix et l'écart type si SE=True, sinon le prix seul.
        """
        MC = getattr(MonteCarlo(self.Udl, self.Maturité, 12 / self.Obs, simu), self.vol_model)(*args, **kwargs)
        Payoff = self.CalculPayoff(MC)
        
        Prix = Payoff.mean()
        Std = Payoff.std()
        Stand_Error = Std / (np.sqrt(simu))
        return (Prix, Stand_Error) if SE else Prix

    def full_price(self, simu: int = 262143, SE: bool=False, force: bool=False, *args, **kwargs) -> tuple[float, float] | float:
        """
        Calcule le prix du produit de l'Autocall avec 252 pas.

        Paramètres :
        ------------
        simu : int, optionnel
            Nombre de simulations de Monte Carlo. Par défaut, 262143.
        SE : bool, optionnel
            Indique si l'écart type doit être retourné. Par défaut, False.
        force : bool, optionnel
            Force la génération de simulation de MonteCarlo Par défaut, False.
            Si le produit à déjà été pricé, il utilise les simus utilisés
            lors du premier priçage.

        Returns:
        ---------
        tuple[float, float] | float:
            Tuple contenant le prix et l'écart type si SE=True, sinon le prix seul.
        """
        updated = self.update()
        if (self.MC is None) or updated or force:
            self.MC = getattr(MonteCarlo(self.Udl, self.Maturité, 252, simu), self.vol_model)(*args, **kwargs)
            
        Jour = round(self.Maturité*252) - round(self.T * 252)
        mask_values = np.arange(0, self.MC.shape[1], int(252 * (self.Obs / 12)))
        mask = mask_values[mask_values >= Jour]
        MC_selected = self.MC[:, mask]
        
        Payoff = self.CalculPayoff(MC_selected)
        
        Prix = Payoff.mean()
        Std = Payoff.std()
        Stand_Error = (Std) / (np.sqrt(simu))
        return (Prix, Stand_Error) if SE else Prix


    def full_price_RV(self, simu: int = 262143, SE: bool=False, force: bool=False) -> tuple[float, float] | float:
        """
        Calcule le prix du produit de l'Autocall avec 252 pas et en réduisant
        la variance de la simulation de MC via un variable corrélée.

        Paramètres :
        ------------
        simu : int, optionnel
            Nombre de simulations de Monte Carlo. Par défaut, 262143.
        SE : bool, optionnel
            Indique si l'écart type doit être retourné. Par défaut, False.
        force : bool, optionnel
            Force la génération de simulation de MonteCarlo Par défaut, False.
            Si le produit à déjà été pricé, il utilise les simus utilisés
            lors du premier priçage.

        Returns:
        ---------
        tuple[float, float] | float:
            Tuple contenant le prix et l'écart type si SE=True, sinon le prix seul.
        """
        updated = self.update()
        updated = True if self.MC.ndim != 3 else False 
        if self.MC is None or updated or force:
            Array3D = MonteCarlo(self.Udl, self.T, 252, simu).RV_Correl()
            
        Liste_Prix = []
        for i in range(Array3D.shape[2]):
            Array2D = Array3D[:,:,i]
            Jour = round(self.Maturité*252) - round(self.T * 252)
            mask_values = np.arange(0, Array2D.shape[1], int(252 * (self.Obs / 12)))
            mask = mask_values[mask_values >= Jour]
            MC_selected = Array2D[:, mask]
            Payoff = self.CalculPayoff(MC_selected)
            Liste_Prix.append(Payoff)
        Liste_Prix = np.array(Liste_Prix)    
        Prix = Payoff.mean()
        Std = Payoff.std()
        Stand_Error = (Std) / (np.sqrt(simu))
        return (Prix, Stand_Error) if SE else Prix


    
    #-------------------------------------------------------------"Grecques"-------------------------------------------------------------#          
    def delta(self, quick: bool = True) -> float: 
        """
        Permet de calculer le Delta Cash de l'Autocall pour une 
        vairation du spot de 1%.
            
        Paramètres :
        ------------
        quick : bool, optionnel
            Indique si le calcul doit utilser price() au lieu de full_price().
            Par défaut, True.
        """
        Autocall_Up = self.copy()
        Autocall_Up.Udl.Spot *= 1.01
        Autocall_Down = self.copy()
        Autocall_Down.Udl.Spot *= 0.99
        if quick:
            Prix_Up = Autocall_Up.price()
            Prix_Down = Autocall_Down.price()
        else:
            Prix_Up = Autocall_Up.full_price()
            Prix_Down = Autocall_Down.full_price()
        Changement_Prix = (Prix_Up - Prix_Down)/2
        return Changement_Prix*100
    
    def vega(self, quick: bool = True) -> float:
        """
        Permet de calculer le Vega Cash de l'Autocall pour 
        une variation de 2.5% de la volatilité en Absolu.

        Paramètres :
        ------------
        quick : bool, optionnel
            Indique si le calcul doit utilser price() au lieu de full_price().
            Par défaut, True.
        """
        Autocall_Up = self.copy()
        Autocall_Up.Udl.Vol = Autocall_Up.Udl.get_vol(self.T, self.K) + 0.01
        Autocall_Down = self.copy()
        Autocall_Down.Udl.Vol = Autocall_Down.Udl.get_vol(self.T, self.K) - 0.01
        if quick:
            Prix_Up = Autocall_Up.price()
            Prix_Down = Autocall_Down.price()
        else:
            Prix_Up = Autocall_Up.full_price()
            Prix_Down = Autocall_Down.full_price()   
        Changement_Prix = (Prix_Up - Prix_Down)/2
        return Changement_Prix*100
    
    def rho(self, quick: bool = True) -> float:
        """
        Permet de calculer le Rho Cash de l'Autocall pour
        une variation de 0.001% des taux en Absolu,
        soit 10bps.
        
        Paramètres :
        ------------
        quick : bool, optionnel
            Indique si le calcul doit utilser price() au lieu de full_price().
            Par défaut, True.
            
        SANS DOUTE FAUX A CHECK
        -----------------------
        """
        Autocall_Up = self.copy()
        Autocall_Up.Udl.Taux = Autocall_Up.Udl.get_rate(self.T) + 0.001
        Autocall_Down = self.copy()
        Autocall_Down.Udl.Taux -= Autocall_Down.Udl.get_rate(self.T) - 0.001
        if quick:
            Prix_Up = Autocall_Up.price()
            Prix_Down = Autocall_Down.price()
        else:
            Prix_Up = Autocall_Up.full_price()
            Prix_Down = Autocall_Down.full_price()
        return (Prix_Up - Prix_Down)/2 * 10
    
    def theta(self, quick: bool = False) -> float:
        """
        Permet de calculer le theta de l'Autocall pour
        1 jour.
        
        Paramètres :
        ------------
        quick : bool, optionnel
            quick n'est pas accepté car j'ai besoin de 252 pas 
            de la simu de MC pour calculer le theta.
        """
        if quick:
            raise ValueError("La fonction Theta n'a pas d'attribut quick car nous avons besoin de tous les pas de simulation de MC.")
        Option_J1 = self.copy()
        Option_J1.T -= 1/252
        prix = self.full_price()
        return prix - Option_J1.full_price() 