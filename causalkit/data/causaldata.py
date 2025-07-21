"""
CKit class for storing DataFrame and column metadata for causal inference.
"""

import pandas as pd
import pandas.api.types as pdtypes
from typing import Union, List, Dict, Optional, Any


class CausalData:
    """
    A class that wraps a pandas DataFrame and stores metadata about columns
    for causal inference analysis.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data. Cannot contain NaN values.
    target : Union[str, List[str]]
        Column name(s) representing the target/outcome variable(s).
    cofounders : Union[str, List[str]], optional
        Column name(s) representing the cofounders/covariates.
    treatment : Union[str, List[str]]
        Column name(s) representing the treatment variable(s).

    Examples
    --------
    >>> from causalkit.data import generate_rct_data
    >>> from causalkit.data import CausalData
    >>> 
    >>> # Generate data
    >>> df = generate_rct_data()
    >>> 
    >>> # Create ckit object
    >>> ck = CausalData(
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
        target: Union[str, List[str]],
        treatment: Union[str, List[str]],
        cofounders: Optional[Union[str, List[str]]] = None,
    ):
        """
        Initialize a ckit object.
        """
        self.df = df
        self._target = self._ensure_list(target)
        self._treatment = self._ensure_list(treatment)
        self._cofounders = self._ensure_list(cofounders) if cofounders is not None else []

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
        Validate that all specified columns exist in the DataFrame and that the DataFrame does not contain NaN values.
        Also validate that target, cofounders, and treatment columns contain only int or float values.
        """
        # Check for NaN values in the DataFrame
        if self.df.isna().any().any():
            raise ValueError("DataFrame contains NaN values, which are not allowed.")
            
        all_columns = set(self.df.columns)
        
        for col_list, name in [
            (self._target, "target"),
            (self._cofounders, "cofounders"),
            (self._treatment, "treatment"),
        ]:
            for col in col_list:
                if col not in all_columns:
                    raise ValueError(f"Column '{col}' specified as {name} does not exist in the DataFrame.")
                
                # Check if column contains only int or float values
                if not pdtypes.is_numeric_dtype(self.df[col]):
                    raise ValueError(f"Column '{col}' specified as {name} must contain only int or float values.")

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

    def get_df(
        self,
        columns: Optional[List[str]] = None,
        include_target: bool = False,
        include_cofounders: bool = False,
        include_treatment: bool = False
    ) -> pd.DataFrame:
        """
        Get a DataFrame from the causaldata object with specified columns.
        
        Parameters
        ----------
        columns : List[str], optional
            Specific column names to include in the returned DataFrame.
            If None and no other include parameters are True, returns the entire DataFrame.
        include_target : bool, default False
            Whether to include target column(s) in the returned DataFrame.
        include_cofounders : bool, default False
            Whether to include cofounder column(s) in the returned DataFrame.
        include_treatment : bool, default False
            Whether to include treatment column(s) in the returned DataFrame.
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing the specified columns.
            
        Examples
        --------
        >>> from causalkit.data import generate_rct_data
        >>> from causalkit.data import CausalData
        >>> 
        >>> # Generate data
        >>> df = generate_rct_data()
        >>> 
        >>> # Create ckit object
        >>> ck = CausalData(
        ...     df=df,
        ...     target='target',
        ...     cofounders=['age', 'invited_friend'],
        ...     treatment='treatment'
        ... )
        >>> 
        >>> # Get specific columns
        >>> ck.get_df(columns=['age', 'gender'])
        >>> 
        >>> # Get target and treatment columns
        >>> ck.get_df(include_target=True, include_treatment=True)
        >>> 
        >>> # Get all columns
        >>> ck.get_df()
        """
        # If no specific columns or includes are specified, return the entire DataFrame
        if columns is None and not any([include_target, include_cofounders, include_treatment]):
            return self.df.copy()
        
        # Start with empty list of columns to include
        cols_to_include = []
        
        # Add specific columns if provided
        if columns is not None:
            cols_to_include.extend(columns)
        
        # Add columns based on include parameters
        if include_target:
            cols_to_include.extend(self._target)
        
        if include_cofounders:
            cols_to_include.extend(self._cofounders)
        
        if include_treatment:
            cols_to_include.extend(self._treatment)
        
        # Remove duplicates while preserving order
        cols_to_include = list(dict.fromkeys(cols_to_include))
        
        # Validate that all requested columns exist
        for col in cols_to_include:
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
        
        # Return the DataFrame with selected columns
        return self.df[cols_to_include].copy()
    
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