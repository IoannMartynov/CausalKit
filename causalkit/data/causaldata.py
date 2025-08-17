"""
CKit class for storing DataFrame and column metadata for causal inference.
"""

import pandas as pd
import pandas.api.types as pdtypes
from typing import Union, List, Optional
import warnings


class CausalData:
    """
    A class that wraps a pandas DataFrame and stores metadata about columns
    for causal inference. The DataFrame is truncated to only include columns
    specified in treatment, confounders, and target.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data. Cannot contain NaN values.
        Only columns specified in target, treatment, and confounders will be stored.
    treatment : str
        Column name representing the treatment variable.
    outcome : str
        Column name representing the target/outcome variable.
    confounders : Union[str, List[str]], optional
        Column name(s) representing the confounders/covariates.

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
    >>> causal_data.confounders
    >>> causal_data.treatment
    """

    def __init__(
            self,
            df: pd.DataFrame,
            treatment: str,
            outcome: str,
            confounders: Optional[Union[str, List[str]]] = None,
    ):
        """
        Initialize a CausalData object.
        """
        self._treatment = treatment
        self._target = outcome
        # Store confounders as a list of unique names (preserve order)
        conf_list = self._ensure_list(confounders) if confounders is not None else []
        merged: List[str] = []
        for v in conf_list:
            if v not in merged:
                merged.append(v)
        self._confounders = merged
        
        # Validate column names
        self._validate_columns(df)
        
        # Store only the relevant columns
        columns_to_keep = [self._target, self._treatment] + self._confounders
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
        Also validate that outcome, confounders, and treatment columns contain only int or float values.
        Also validate that no columns are constant (have zero variance).
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

        # Check if outcome column is constant (zero variance or single value)
        if df[self._target].std() == 0 or pd.isna(df[self._target].std()):
            raise ValueError(f"Column '{self._target}' specified as outcome is constant (has zero variance), which is not allowed for causal inference.")

        # Validate treatment column
        if self._treatment not in all_columns:
            raise ValueError(f"Column '{self._treatment}' specified as treatment does not exist in the DataFrame.")

        # Check if treatment column contains only int or float values
        if not pdtypes.is_numeric_dtype(df[self._treatment]):
            raise ValueError(
                f"Column '{self._treatment}' specified as treatment must contain only int or float values.")

        # Check if treatment column is constant (zero variance or single value)
        if df[self._treatment].std() == 0 or pd.isna(df[self._treatment].std()):
            raise ValueError(f"Column '{self._treatment}' specified as treatment is constant (has zero variance), which is not allowed for causal inference.")

        # Validate confounders columns
        for col in self._confounders:
            if col not in all_columns:
                raise ValueError(f"Column '{col}' specified as confounders does not exist in the DataFrame.")

            # Check if column contains only int or float values
            if not pdtypes.is_numeric_dtype(df[col]):
                raise ValueError(f"Column '{col}' specified as confounders must contain only int or float values.")

            # Check if confounder column is constant (zero variance or single value)
            if df[col].std() == 0 or pd.isna(df[col].std()):
                raise ValueError(f"Column '{col}' specified as confounders is constant (has zero variance), which is not allowed for causal inference.")

        # Check for duplicate column values across all used columns
        self._check_duplicate_column_values(df)
        
        # Check for duplicate rows and issue warning if found
        self._check_duplicate_rows(df)

    def _check_duplicate_column_values(self, df):
        """
        Check for duplicate column values across all used columns.
        Raises ValueError if any two columns have identical values.
        """
        # Get all columns that will be used in CausalData
        columns_to_check = [self._target, self._treatment] + self._confounders
        
        # Compare each pair of columns
        for i, col1 in enumerate(columns_to_check):
            for j, col2 in enumerate(columns_to_check):
                if i < j:  # Only check each pair once
                    # Check if the two columns have identical values (using element-wise comparison)
                    # This handles cases where dtypes differ but values are the same (e.g., int vs float)
                    if (df[col1] == df[col2]).all():
                        # Determine the types of columns for better error message
                        col1_type = self._get_column_type(col1)
                        col2_type = self._get_column_type(col2)
                        raise ValueError(
                            f"Columns '{col1}' ({col1_type}) and '{col2}' ({col2_type}) have identical values, "
                            f"which is not allowed for causal inference. Only column names differ."
                        )

    def _check_duplicate_rows(self, df):
        """
        Check for duplicate rows in the DataFrame and issue a warning if found.
        Only checks the columns that will be used in CausalData.
        """
        # Get only the columns that will be used in CausalData
        columns_to_check = [self._target, self._treatment] + self._confounders
        df_subset = df[columns_to_check]
        
        # Find duplicate rows
        duplicated_mask = df_subset.duplicated()
        num_duplicates = duplicated_mask.sum()
        
        if num_duplicates > 0:
            total_rows = len(df_subset)
            unique_rows = total_rows - num_duplicates
            
            warnings.warn(
                f"Found {num_duplicates} duplicate rows out of {total_rows} total rows in the DataFrame. "
                f"This leaves {unique_rows} unique rows for analysis. "
                f"Duplicate rows may affect the quality of causal inference results. "
                f"Consider removing duplicates if they are not intentional.",
                UserWarning,
                stacklevel=3
            )

    def _get_column_type(self, column_name):
        """
        Determine the type/role of a column (treatment, outcome, or confounder).
        """
        if column_name == self._target:
            return "outcome"
        elif column_name == self._treatment:
            return "treatment"
        elif column_name in self._confounders:
            return "confounder"
        else:
            return "unknown"

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
    def confounders(self) -> List[str]:
        """List of confounder column names."""
        return list(self._confounders) if self._confounders else []

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
            include_confounders: bool = True
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
        include_confounders : bool, default True
            Whether to include confounder column(s) in the returned DataFrame.

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
        ...     confounders=['age', 'invited_friend']
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
        if columns is None and not any([include_target, include_confounders, include_treatment]):
            return self.df.copy()

        # Add columns based on include parameters
        if include_target:
            cols_to_include.append(self._target)

        if include_confounders:
            cols_to_include.extend(self._confounders)

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
            f"treatment='{self._treatment}', "
            f"outcome='{self._target}', "
            f"confounders={self._confounders})"
        )