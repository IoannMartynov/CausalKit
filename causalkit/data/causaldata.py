"""
CKit class for storing DataFrame and column metadata for causal inference.
"""

import pandas as pd
from typing import Union, List, Dict, Optional, Any


class causaldata:
    """
    A class that wraps a pandas DataFrame and stores metadata about columns
    for causal inference analysis.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    target : Union[str, List[str]], optional
        Column name(s) representing the target/outcome variable(s).
    cofounders : Union[str, List[str]], optional
        Column name(s) representing the cofounders/covariates.
    treatment : Union[str, List[str]], optional
        Column name(s) representing the treatment variable(s).
    metadata : Dict[str, Any], optional
        Additional metadata about the dataset.

    Examples
    --------
    >>> from causalkit.data import generate_rct_data
    >>> from causalkit.data import causaldata
    >>> 
    >>> # Generate data
    >>> df = generate_rct_data()
    >>> 
    >>> # Create ckit object
    >>> ck = causaldata(
    ...     df=df,
    ...     target='target',
    ...     cofounders=['age', 'invited_friend'],
    ...     treatment='treatment'
    ... )
    >>> 
    >>> # Access data
    >>> ck.df.head()
    >>> 
    >>> # Access columns by role
    >>> ck.target
    >>> ck.cofounders
    >>> ck.treatment
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: Optional[Union[str, List[str]]] = None,
        cofounders: Optional[Union[str, List[str]]] = None,
        treatment: Optional[Union[str, List[str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a ckit object.
        """
        self.df = df
        self._target = self._ensure_list(target) if target is not None else []
        self._cofounders = self._ensure_list(cofounders) if cofounders is not None else []
        self._treatment = self._ensure_list(treatment) if treatment is not None else []
        self.metadata = metadata or {}

        # Validate column names
        self._validate_columns()

    def _ensure_list(self, value: Union[str, List[str]]) -> List[str]:
        """
        Ensure that the value is a list of strings.
        """
        if isinstance(value, str):
            return [value]
        return value

    def _validate_columns(self):
        """
        Validate that all specified columns exist in the DataFrame.
        """
        all_columns = set(self.df.columns)
        
        for col_list, name in [
            (self._target, "target"),
            (self._cofounders, "cofounders"),
            (self._treatment, "treatment"),
        ]:
            for col in col_list:
                if col not in all_columns:
                    raise ValueError(f"Column '{col}' specified as {name} does not exist in the DataFrame.")

    @property
    def target(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Get the target/outcome variable(s).
        
        Returns
        -------
        Union[pd.Series, pd.DataFrame]
            If a single target column is specified, returns a pandas Series.
            If multiple target columns are specified, returns a pandas DataFrame.
        """
        if not self._target:
            return None
        
        if len(self._target) == 1:
            return self.df[self._target[0]]
        
        return self.df[self._target]

    @property
    def cofounders(self) -> pd.DataFrame:
        """
        Get the cofounders/covariates.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing the cofounder columns.
        """
        if not self._cofounders:
            return None
        
        return self.df[self._cofounders]

    @property
    def treatment(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Get the treatment variable(s).
        
        Returns
        -------
        Union[pd.Series, pd.DataFrame]
            If a single treatment column is specified, returns a pandas Series.
            If multiple treatment columns are specified, returns a pandas DataFrame.
        """
        if not self._treatment:
            return None
        
        if len(self._treatment) == 1:
            return self.df[self._treatment[0]]
        
        return self.df[self._treatment]

    def __repr__(self) -> str:
        """
        String representation of the ckit object.
        """
        return (
            f"ckit(df={self.df.shape}, "
            f"target={self._target}, "
            f"cofounders={self._cofounders}, "
            f"treatment={self._treatment})"
        )