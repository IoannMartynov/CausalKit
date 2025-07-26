"""
CKit class for storing DataFrame and column metadata for causal inference.
"""

import pandas as pd
import pandas.api.types as pdtypes
from typing import Union, List, Dict, Optional, Any


class CausalData:
    """
    A class that wraps a pandas DataFrame and stores metadata about columns
    for causal inference. The DataFrame is truncated to only include columns
    specified in treatment, cofounders, and target.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data. Cannot contain NaN values.
        Only columns specified in target, treatment, and cofounders will be stored.
    treatment : str
        Column name representing the treatment variable.
    target : str
        Column name representing the target/outcome variable.
    cofounders : Union[str, List[str]], optional
        Column name(s) representing the cofounders/covariates.

    Examples
    --------
    >>> from causalkit.data import generate_rct_data
    >>> from causalkit.data import CausalData
    >>>
    >>> # Generate data
    >>> df = generate_rct_data()
    >>>
    >>> # Create ckit object
    >>> ckit_object = CausalData(
    ...     df=df,
    ...     treatment='treatment',
    ...     target='target',
    ...     cofounders=['age', 'invited_friend']
    ... )
    >>>
    >>> # Access data
    >>> ckit_object.df.head()
    >>>
    >>> # Access columns by role
    >>> ckit_object.target
    >>> ckit_object.cofounders
    >>> ckit_object.treatment
    """

    def __init__(
            self,
            df: pd.DataFrame,
            treatment: str,
            target: str,
            cofounders: Optional[Union[str, List[str]]] = None,
    ):
        """
        Initialize a ckit object.
        """
        self._treatment = treatment
        self._target = target
        self._cofounders = self._ensure_list(cofounders) if cofounders is not None else []
        
        # Validate column names
        self._validate_columns(df)
        
        # Store only the relevant columns
        columns_to_keep = [self._target, self._treatment] + self._cofounders
        self.df = df[columns_to_keep].copy()

    def _ensure_list(self, value: Union[str, List[str]]) -> List[str]:
        """
        Ensure that the value is a list of strings.
        """
        if isinstance(value, str):
            return [value]
        return value

    def _validate_columns(self, df):
        """
        Validate that all specified columns exist in the DataFrame and that the DataFrame does not contain NaN values.
        Also validate that target, cofounders, and treatment columns contain only int or float values.
        """
        # Check for NaN values in the DataFrame
        if df.isna().any().any():
            raise ValueError("DataFrame contains NaN values, which are not allowed.")

        all_columns = set(df.columns)

        # Validate target column
        if self._target not in all_columns:
            raise ValueError(f"Column '{self._target}' specified as target does not exist in the DataFrame.")

        # Check if target column contains only int or float values
        if not pdtypes.is_numeric_dtype(df[self._target]):
            raise ValueError(f"Column '{self._target}' specified as target must contain only int or float values.")

        # Validate treatment column
        if self._treatment not in all_columns:
            raise ValueError(f"Column '{self._treatment}' specified as treatment does not exist in the DataFrame.")

        # Check if treatment column contains only int or float values
        if not pdtypes.is_numeric_dtype(df[self._treatment]):
            raise ValueError(
                f"Column '{self._treatment}' specified as treatment must contain only int or float values.")

        # Validate cofounders columns
        for col in self._cofounders:
            if col not in all_columns:
                raise ValueError(f"Column '{col}' specified as cofounders does not exist in the DataFrame.")

            # Check if column contains only int or float values
            if not pdtypes.is_numeric_dtype(df[col]):
                raise ValueError(f"Column '{col}' specified as cofounders must contain only int or float values.")

    @property
    def target(self) -> pd.Series:
        """
        Get the target/outcome variable.

        Returns
        -------
        pd.Series
            The target column as a pandas Series.
        """
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
    def treatment(self) -> pd.Series:
        """
        Get the treatment variable.

        Returns
        -------
        pd.Series
            The treatment column as a pandas Series.
        """
        return self.df[self._treatment]
        

    def get_df(
            self,
            columns: Optional[List[str]] = None,
            include_target: bool = True,
            include_cofounders: bool = True,
            include_treatment: bool = True
    ) -> pd.DataFrame:
        """
        Get a DataFrame from the causaldata object with specified columns.

        Parameters
        ----------
        columns : List[str], optional
            Specific column names to include in the returned DataFrame.
            If provided, these columns will be included in addition to any columns
            specified by the include parameters.
            If None, columns will be determined solely by the include parameters.
            If None and no include parameters are True, returns the entire DataFrame.
        include_target : bool, default True
            Whether to include target column(s) in the returned DataFrame.
        include_cofounders : bool, default True
            Whether to include cofounder column(s) in the returned DataFrame.
        include_treatment : bool, default True
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
        >>> ckit_object = CausalData(
        ...     df=df,
        ...     treatment='treatment',
        ...     target='target',
        ...     cofounders=['age', 'invited_friend']
        ... )
        >>>
        >>> # Get specific columns
        >>> ckit_object.get_df(columns=['age'])
        >>>
        >>> # Get target and treatment columns
        >>> ckit_object.get_df(include_target=True, include_treatment=True)
        >>>
        >>> # Get all columns
        >>> ckit_object.get_df()
        """
        # Start with empty list of columns to include
        cols_to_include = []

        # If specific columns are provided, add them to the list
        if columns is not None:
            cols_to_include.extend(columns)

        # If no specific columns are provided and no include parameters are True,
        # return the entire DataFrame
        if columns is None and not any([include_target, include_cofounders, include_treatment]):
            return self.df.copy()

        # Add columns based on include parameters
        if include_target:
            cols_to_include.append(self._target)

        if include_cofounders:
            cols_to_include.extend(self._cofounders)

        if include_treatment:
            cols_to_include.append(self._treatment)

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
            f"target='{self._target}', "
            f"cofounders={self._cofounders}, "
            f"treatment='{self._treatment}')"
        )