"""
CKit class for storing DataFrame and column metadata for causal inference.
"""

import pandas as pd
import pandas.api.types as pdtypes
from typing import Union, List, Optional


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
    outcome : str
        Column name representing the target/outcome variable.
    confounders : Union[str, List[str]], optional
        Column name(s) representing the cofounders/covariates.

    Examples
    --------
    >>> from causalkit.data import generate_rct_data
    >>> from causalkit.data import CausalData
    >>>
    >>> # Generate data
    >>> df = generate_rct_data()
    >>>
    >>> # Create CausalData object
    >>> causal_data = CausalData(
    ...     df=df,
    ...     treatment='treatment',
    ...     outcome='outcome',
    ...     confounders=['age', 'invited_friend']
    ... )
    >>>
    >>> # Access data
    >>> causal_data.df.head()
    >>>
    >>> # Access columns by role
    >>> causal_data.target
    >>> causal_data.cofounders
    >>> causal_data.treatment
    """

    def __init__(
            self,
            df: pd.DataFrame,
            treatment: str,
            outcome: str,
            cofounders: Optional[Union[str, List[str]]] = None,
            confounders: Optional[Union[str, List[str]]] = None,
    ):
        """
        Initialize a CausalData object.
        """
        self._treatment = treatment
        self._target = outcome
        # Accept both spellings; merge if both provided
        cof_list = self._ensure_list(cofounders) if cofounders is not None else []
        conf_list = self._ensure_list(confounders) if confounders is not None else []
        merged = []
        for v in cof_list + conf_list:
            if v not in merged:
                merged.append(v)
        self._cofounders = merged
        
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
        Also validate that outcome, cofounders, and treatment columns contain only int or float values.
        """
        # Check for NaN values in the DataFrame
        if df.isna().any().any():
            raise ValueError("DataFrame contains NaN values, which are not allowed.")

        all_columns = set(df.columns)

        # Validate outcome column
        if self._target not in all_columns:
            raise ValueError(f"Column '{self._target}' specified as outcome does not exist in the DataFrame.")

        # Check if outcome column contains only int or float values
        if not pdtypes.is_numeric_dtype(df[self._target]):
            raise ValueError(f"Column '{self._target}' specified as outcome must contain only int or float values.")

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
        Get the outcome/outcome variable.

        Returns
        -------
        pd.Series
            The outcome column as a pandas Series.
        """
        return self.df[self._target]

    # Backwards-compat alias expected by CausalEDA: expose `.outcome` as a Series
    @property
    def outcome(self) -> pd.Series:
        return self.target

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

    # Backwards-compat spelling expected by some callers: return list of names
    @property
    def confounders(self) -> Optional[List[str]]:
        return list(self._cofounders) if self._cofounders else []

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
            include_treatment: bool = True,
            include_target: bool = True,
            include_cofounders: bool = True
    ) -> pd.DataFrame:
        """
        Get a DataFrame from the CausalData object with specified columns.

        Parameters
        ----------
        columns : List[str], optional
            Specific column names to include in the returned DataFrame.
            If provided, these columns will be included in addition to any columns
            specified by the include parameters.
            If None, columns will be determined solely by the include parameters.
            If None and no include parameters are True, returns the entire DataFrame.
        include_treatment : bool, default True
            Whether to include treatment column(s) in the returned DataFrame.
        include_target : bool, default True
            Whether to include target column(s) in the returned DataFrame.
        include_cofounders : bool, default True
            Whether to include cofounder column(s) in the returned DataFrame.

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
        >>> # Create CausalData object
        >>> causal_data = CausalData(
        ...     df=df,
        ...     treatment='treatment',
        ...     outcome='outcome',
        ...     cofounders=['age', 'invited_friend']
        ... )
        >>>
        >>> # Get specific columns
        >>> causal_data.get_df(columns=['age'])
        >>>
        >>> # Get all columns
        >>> causal_data.get_df()
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
        String representation of the CausalData object.
        """
        return (
            f"CausalData(df={self.df.shape}, "
            f"treatment='{self._treatment}')"
            f"outcome='{self._target}', "
            f"cofounders={self._cofounders}, "
        )