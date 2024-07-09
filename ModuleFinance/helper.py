import pandas as pd
import numpy as np
import json
import time
import sys
from numba import njit



#-----------------------------------------------------------------------------------------------------"Manip Date"-----------------------------------------------------------------------------------------------------#
def add_month(date: np.datetime64, month: int) -> np.datetime64:
    day = np.datetime_as_string(date, unit="D")[8:10]
    mois = int(np.datetime_as_string(date, unit="M")[5:7])
    year = int(np.datetime_as_string(date, unit="Y"))
    new_mois = (mois - 1 + month) % 12 + 1 #permet de réindexé et donc d'avoir mes mois commençant à 0 pour janvier et finissant à 12 pour décembre sinon pour décembre j'obtiens 0 puis je rajoute 1 au moins janvier est le 1 et décembre le 12
    Nb_Année = (mois - 1 + month) // 12
    new_year = year + Nb_Année
    new_date = np.datetime64(f"{new_year}-{new_mois:02d}-{day}")
    return new_date

def add_year(date: np.datetime64, years: int) -> np.datetime64:
    day = np.datetime_as_string(date, unit="D")[8:10]
    mois = np.datetime_as_string(date, unit="M")[5:7]
    year = int(np.datetime_as_string(date, unit="Y"))
    new_year = int(year + years)
    new_date = np.datetime64(f"{new_year}-{mois}-{day}")
    return new_date

def format_time(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{int(hours):02d}h{int(minutes):02d}"
    elif minutes > 0:
        return f"{int(minutes):02d}min{int(seconds):02d}s"
    else:
        return f"{seconds:.2f}s"



#-----------------------------------------------------------------------------------------------------"Decorateur"-----------------------------------------------------------------------------------------------------#
def timer(func):
    """Décorateur pour mesurer le temps d'exécution d'une fonction."""
    def wrapper(*args, **kwargs):
        temps_debut = time.time()
        result = func(*args, **kwargs)
        temps_fin = time.time()
        sys.stdout.write(f"La function: {func.__name__} à prit {temps_fin - temps_debut:.6f} secondes\n")
        sys.stdout.flush()
        return result
    return wrapper

def cache(func):
    """Décorateur pour mettre en cache les résultats d'une fonction."""
    Dico = {}
    def wrapper(*args, **kwargs):
        key = str(func.__name__) + str(args) + str(**kwargs)
        if key not in Dico:
            Dico[key] = func(*args, **kwargs)
        return Dico[key]
    return wrapper

class ProgressBar:
    _Nb_ProgressBar = 0
    def __init__(self, sequence, nom: str="") -> None:
        """
        Class qui affiche une barre de progréssion de la boucle en indiquant, le pourcentage de progression, le temps écoulé, le temps restant, et le temps moyen.
        A utiliser comme suit::
        
            for i in ProgressBar(range(10)):
                #YOUR CODE
        
            for i in ProgressBar(range(5,len(list)), "Nom de la Fonction"):
                #YOUR CODE
        
        
        Paramètres
        ----------
        sequence : iterable
            La séquence à parcourir.
        nom : str, optionnel
            Nom pour la barre de progression (par défaut, une chaîne vide), peut être remplacé par le nom d'une fonction. Utile notamment lorsqu'il y a des boucles dans des boucles.

        Returns
        -------
        int
            La prochaine valeur de la boucle.

        Raises
        ------
        StopIteration
            Est levée lorsque la boucle est terminée.
        """
        self.sequence = sequence
        self.total: int = len(sequence)
        self.index: int = 0
        self.time: float = time.time()
        self.nom: str = nom
        ProgressBar._Nb_ProgressBar += 1

    def __iter__(self)  -> "ProgressBar":
        return self

    def __next__(self) -> int:
        if self.index < self.total:
            value = self.sequence[self.index]
            self.index += 1
            if value > 0:
                Temps_Passé = time.time() - self.time
                Temps_Moyen = Temps_Passé/value
                Temps_Restant = Temps_Moyen * (self.total - value)
                
                NB_BlockTotal =  40
                Pourcentage = (self.index / self.total) * 100
                Block = int((Pourcentage / 100) * NB_BlockTotal)
                Bar = "|" + "█" * Block + " " * (NB_BlockTotal-Block) + "|"
                Message = f"Progression de {self.nom}: " if self.nom else ""
                

                sys.stdout.write(f"\r{Message} {Bar} {self.index}/{self.total} [{Pourcentage:.0f}%] in {format_time(Temps_Passé)} (~{format_time(Temps_Restant)}, {format_time(Temps_Moyen)})                      ")
                sys.stdout.flush()
            else:
                sys.stdout.write("\n"* (ProgressBar._Nb_ProgressBar-1))
                sys.stdout.write("\rInitialisation...")
                sys.stdout.flush()
                sys.stdout.write("\r")
            return value
        else:
            sys.stdout.write(f"\r" + " "*1000)
            sys.stdout.write("\r")
            ProgressBar._Nb_ProgressBar -= 1
            sys.stdout.write("\033[F")
            sys.stdout.flush()
            raise StopIteration

#--------------------------------------------------------------------------------------------------------"Math"--------------------------------------------------------------------------------------------------------#
def _erfcc(x:float) -> float:
    # https://stackoverflow.com/questions/809362/how-to-calculate-cumulative-normal-distribution
    z = abs(x)
    t = 1. / (1. + 0.5*z)
    r = t * np.exp(-z*z-1.26551223+t*(1.00002368+t*(.37409196+
        t*(.09678418+t*(-.18628806+t*(.27886807+
        t*(-1.13520398+t*(1.48851587+t*(-.82215223+
        t*.17087277)))))))))
    if (x >= 0.):
        return r
    else:
        return 2. - r

def norm_cdf(x: float, mu: float=0, std: float=1) -> float:
    """
    Cumulative Distribution Function

    Parameters
    ----------
    x : float
        Value for which cdf(x)
    mu : float, optional
        Average of the standard repartition, by default 0
    std : float, optional
        Std. dev of the standard repartition, by default 1

    Returns
    -------
    float
        The probability that the value will be inferior to the input
    """
    t = x-mu
    y = 0.5*_erfcc(-t/(std*np.sqrt(2.0)))
    if y>1.0:
        y = 1.0
    return y

def norm_pdf(x: float, mu: float=0, std: float=1) -> float:
    """
    Probability Density Function

    Parameters
    ----------
    x : float
        Value for which pdf(x)
    mu : float, optional
        Average of the standard distribution, by default 0
    std : float, optional
        Std. dev of the standard distribution, by default 1

    Returns
    -------
    float
        The probability density at the given value
    """
    pdf = (1.0 / (std * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu) / std)**2)
    return pdf





class CubicSpline():
    def __init__(self, df:pd.DataFrame):
        self.df = df
            
    def calculate_spline_coefficients(self):
        """
        Calculates the coefficients of the cubic spline.

        Returns
        -------
        list
            Coefficients of the cubic spline.
        """
        __coef_matrix = self._get_coefficient_matrix(self.df).to_numpy()
        _result_vector = self._get_result_vector(self.df)
        _coef = gaussian_elimination(__coef_matrix, _result_vector)
        return [_coef[i:i+4] for i in range(0, len(_coef), 4)]

    def _get_coefficient_matrix(self, _df: pd.DataFrame)-> pd.DataFrame:
        """
        Generates the coefficient matrix for solving cubic spline equations.

        Parameters
        ----------
        _df : pd.DataFrame
            DataFrame containing the data points for interpolation.

        Returns
        -------
        pd.DataFrame
            Coefficient matrix for solving cubic spline equations.
        """
        _nb_spline = len(_df)-1
        _coef = pd.DataFrame(np.zeros((_nb_spline*4,_nb_spline*4), dtype=float))
        _coef = self._fill_known_points_coefficients(_df, _coef)
        _coef = self._fill_first_derivatives_coefficients(_df, _coef)
        _coef = self._fill_second_derivatives_coefficients(_df, _coef)
        _coef = self._fill_remaining_equations(_df, _coef)
        return _coef

    def _get_result_vector(self, _df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates the result vector for solving cubic spline equations.

        Parameters
        ----------
        _df : pd.DataFrame
            DataFrame containing the data points for interpolation.

        Returns
        -------
        pd.DataFrame
            Result vector for solving cubic spline equations.
        """
        _rslt = self._get_rslt_for_know_points(_df.iloc[:,0].values.tolist())
        i = len(_rslt)
        while i < ((len(_df)-1)*4):
            _rslt.append(0)
            i += 1
        return np.array([_rslt]).transpose()

    def _cubic(self, x:float)->list:
        return x**3, x**2, x, 1

    def _cubic_first_derivatives(self, x:float) ->list:
        return (3*x**2, 2*x,1), (-3*x**2, -2*x,-1)
    
    def _cubic_second_derivatives(self, x:float) ->list:
        return (6*x, 2), (-6*x, -2)

    def _get_rslt_for_know_points(self, points: list) ->list:
        _rslt = []
        _rslt.append(points[0])
        for i in range(1,len(points) -1):
            _rslt.append(points[i])
            _rslt.append(points[i])
        _rslt.append(points[-1])
        return _rslt

    def _fill_known_points_coefficients(self, _df: pd.DataFrame, _coef:pd.DataFrame) -> pd.DataFrame:
        """
        Fills the coefficients of the equations corresponding to known points in the coefficient matrix.

        Parameters
        ----------
        _df : pd.DataFrame
            DataFrame containing the data points for interpolation.
        _coef : pd.DataFrame
            Coefficient matrix to be filled.

        Returns
        -------
        pd.DataFrame
            Coefficient matrix with coefficient for known points filled.
        """
        col = -4
        for i,x in enumerate(self._get_rslt_for_know_points(_df.index)):
            if i % 2 == 0:
                col += 4
            _coef.iloc[i,[col, col+1, col+2, col+3]] = self._cubic(x)
        return _coef

    def _fill_first_derivatives_coefficients(self, _df: pd.DataFrame, _coef:pd.DataFrame)-> pd.DataFrame:
        """
        Fills the coefficients of the matching first derivatives in the coefficient matrix.

        Parameters
        ----------
        _df : pd.DataFrame
            DataFrame containing the data points for interpolation.
        _coef : pd.DataFrame
            Coefficient matrix to be filled.

        Returns
        -------
        pd.DataFrame
            Coefficient matrix with first derivative coefficients filled.
        """
        row = (len(_df)-1)*2
        col = 0
        for x in _df.index[1:-1]:
            _coef.iloc[row, [col, col+1, col+2]] = self._cubic_first_derivatives(x)[0]
            col += 4
            _coef.iloc[row, [col, col+1, col+2]] = self._cubic_first_derivatives(x)[1]
            row += 1
        return _coef
    
    def _fill_second_derivatives_coefficients(self, _df: pd.DataFrame, _coef:pd.DataFrame)-> pd.DataFrame:
        """
        Fills the coefficients of the matching second derivatives in the coefficient matrix.

        Parameters
        ----------
        _df : pd.DataFrame
            DataFrame containing the data points for interpolation.
        _coef : pd.DataFrame
            Coefficient matrix to be filled.

        Returns
        -------
        pd.DataFrame
            Coefficient matrix with second derivative coefficients filled.
        """
        nb_spline = len(_df)-1
        row_know_points = nb_spline*2
        row_fist_derivatives = nb_spline-1
        
        current_row = row_know_points + row_fist_derivatives
        col = 0        
        for x in _df.index[1:-1]:
            _coef.iloc[current_row, [col, col+1]] = self._cubic_second_derivatives(x)[0]
            col += 4
            _coef.iloc[current_row, [col, col+1]] = self._cubic_second_derivatives(x)[1]
            current_row += 1
        return _coef

    def _fill_remaining_equations(self, _df: pd.DataFrame, _coef: pd.DataFrame) -> pd.DataFrame:
        """
        Fills the coefficients of remaining equations in the coefficient matrix, following selected hypothesis.

        Parameters
        ----------
        _df : pd.DataFrame
            DataFrame containing the data points for interpolation.
        _coef : pd.DataFrame
            Coefficient matrix to be filled.

        Returns
        -------
        pd.DataFrame
            Coefficient matrix with remaining equations filled.
        """
        #Fisrt possibility, the second derivtives for our end points is 0
        nb_spline = len(_df)-1
        row_know_points = nb_spline*2
        row_fist_derivatives = nb_spline-1
        row_second_derivatives = nb_spline-1
        
        current_row = row_know_points + row_fist_derivatives + row_second_derivatives
        #first end point 
        x = _df.index[0]
        _coef.iloc[current_row, [0, 1]] = self._cubic_second_derivatives(x)[0]
        current_row += 1
        #last end point 
        x = _df.index[-1]
        col = len(_coef)-4
        _coef.iloc[current_row, [col, col+1]] = self._cubic_second_derivatives(x)[0]
        current_row += 1
        return _coef
    
    
    

@njit(fastmath=True, nogil=True)
def gaussian_elimination(_coef_matrice: np.ndarray, _rslt_matrice: np.ndarray):
    augmented_matrice = np.concatenate((_coef_matrice, _rslt_matrice), axis=1)

    n = len(_rslt_matrice)
    m = n - 1
    i = 0
    x = np.zeros(n)
    while i < n:
        # Partial pivoting
        max_index = i
        for p in range(i+1, n):
            if abs(augmented_matrice[p, i]) > abs(augmented_matrice[max_index, i]):
                max_index = p
        if augmented_matrice[max_index, i] == 0:
            raise ValueError("Cannot divide by zero")
        # Swap rows
        for k in range(i, n+1):
            temp = augmented_matrice[i, k]
            augmented_matrice[i, k] = augmented_matrice[max_index, k]
            augmented_matrice[max_index, k] = temp

        for j in range(i+1, n):
            scaling_factor = augmented_matrice[j, i] / augmented_matrice[i, i]
            for k in range(i, n+1):
                augmented_matrice[j, k] = augmented_matrice[j, k] - scaling_factor * augmented_matrice[i, k]
        i += 1

    # Back substitution for x-matrix
    x[m] = augmented_matrice[m, n] / augmented_matrice[m, m]
    for k in range(n - 2, -1, -1):
        x[k] = augmented_matrice[k, n]
        for w in range(k + 1, n):
            x[k] = x[k] - augmented_matrice[k, w] * x[w]
        x[k] = x[k] / augmented_matrice[k, k]
    return x

def calculate_spline(x: float, spline_number: int, spline_equa:np.ndarray) -> float:
    coef = spline_equa[spline_number]
    return coef[0]*x**3 + coef[1]*x**2 + coef[2]*x + coef[3]

def interpolation_lineaire(coord_1:list, coord_2:list, x_to_solve) -> float:
    x1, y1 = coord_1
    x2, y2 = coord_2
    slope = (y2 - y1) / (x2 - x1)
    return slope * (x_to_solve - x1) + y1




#------------------------------------------------------------------------------------------------------"Read JSON"------------------------------------------------------------------------------------------------------#
def load_curve(Chemin: str) -> tuple:
    """
    Fonction retournant un tupple comme suit : [Maturité, Taux].
    La maturité doit être en str et les taux en tant que float.

    Parameters
    ----------
    Chemin : str
        Folder ou sont situé les données.
    """
    Resultat = []
    with open(Chemin , "r") as file:
        data = json.load(file)
        
    for sublist in data:
        if isinstance(sublist[0], str):
            Liste = []
            for value in sublist:
                try:
                    converted_value = float(value)
                except ValueError:
                    numerateur, denominateur = (map(int, value.split('/')))
                    converted_value = numerateur/denominateur
                Liste.append(converted_value)
        else:
            Liste = [rate for rate in sublist]
        Resultat.append(Liste)
    return Resultat

def load_surface(Chemin: str) -> pd.DataFrame:
    """
    Reads a JSON file and returns its contents as a pandas DataFrame.

    Parameters
    ----------
    Chemin : str
        Path to the JSON file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the data from the JSON file.
    """
    with open(Chemin , "r") as file:
        data = json.load(file)
    if isinstance(data,list):
        df = pd.DataFrame(data[1:], columns=data[0]).set_index(data[0][0])
    if isinstance(data, dict):
        df = pd.DataFrame(data)
        df = df.set_index(df.columns[0])
    df.index.name = None
    return df