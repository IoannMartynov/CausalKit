"""
CKit class for storing DataFrame and column metadata for causal inference.
"""

import pandas as pd
import pandas.api.types as pdtypes
from typing import Union, List, Optional
import warnings


class CausalData:
    """
    Container for causal inference datasets.

    Wraps a pandas DataFrame and stores the names of treatment, outcome, and optional confounder columns.
    The stored DataFrame is restricted to only those columns.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data. Cannot contain NaN values.
        Only columns specified in outcome, treatment, and confounders will be stored.
    treatment : str
        Column name representing the treatment variable.
    outcome : str
        Column name representing the outcome (target) variable.
    confounders : Union[str, List[str]], optional
        Column name(s) representing the confounders/covariates.

    Attributes
    ----------
    df : pd.DataFrame
        A copy of the original data restricted to [outcome, treatment] + confounders.
    treatment : str
        Name of the treatment column.
    outcome : str
        Name of the outcome (target) column.
    confounders : list[str]
        Names of the confounder columns (may be empty).

    Examples
    --------
    >>> from causalis.data import generate_rct
    >>> from causalis.data import CausalData
    >>>
    >>> # Generate data
    >>> df = generate_rct()
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
        
        # Coerce boolean columns to integers to ensure stored df is fully numeric
        for col in self.df.columns:
            if pdtypes.is_bool_dtype(self.df[col]):
                # Use int8 for compact 0/1 storage
                self.df[col] = self.df[col].astype("int8")
        
        # Final safeguard: ensure all stored columns are numeric
        for col in self.df.columns:
            if not pdtypes.is_numeric_dtype(self.df[col]):
                raise ValueError(
                    f"All columns in stored DataFrame must be numeric; column '{col}' has dtype {self.df[col].dtype}."
                )

        # Re-run duplicate-column and duplicate-row checks on the stored, normalized subset
        self._check_duplicate_column_values(self.df)
        self._check_duplicate_rows(self.df)

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

        # Check if outcome column contains numeric or boolean values
        if not (pdtypes.is_numeric_dtype(df[self._target]) or pdtypes.is_bool_dtype(df[self._target])):
            raise ValueError(f"Column '{self._target}' specified as outcome must contain only int, float, or bool values.")

        # Check if outcome column is constant (single unique value)
        if df[self._target].nunique(dropna=False) <= 1:
            raise ValueError(
                f"Column '{self._target}' specified as outcome is constant (has zero variance / single unique value), "
                f"which is not allowed for causal inference."
            )

        # Validate treatment column
        if self._treatment not in all_columns:
            raise ValueError(f"Column '{self._treatment}' specified as treatment does not exist in the DataFrame.")

        # Check if treatment column contains numeric or boolean values
        if not (pdtypes.is_numeric_dtype(df[self._treatment]) or pdtypes.is_bool_dtype(df[self._treatment])):
            raise ValueError(
                f"Column '{self._treatment}' specified as treatment must contain only int, float, or bool values.")

        # Check if treatment column is constant (single unique value)
        if df[self._treatment].nunique(dropna=False) <= 1:
            raise ValueError(
                f"Column '{self._treatment}' specified as treatment is constant (has zero variance / single unique value), "
                f"which is not allowed for causal inference."
            )

        # Validate confounders columns; drop constant ones with a warning
        kept_confounders: List[str] = []
        dropped_constants: List[str] = []
        for col in self._confounders:
            if col not in all_columns:
                raise ValueError(f"Column '{col}' specified as confounders does not exist in the DataFrame.")

            # Check if confounder column contains numeric or boolean values
            if not (pdtypes.is_numeric_dtype(df[col]) or pdtypes.is_bool_dtype(df[col])):
                raise ValueError(f"Column '{col}' specified as confounders must contain only int, float, or bool values.")

            # Check if confounder column is constant (single unique value)
            if df[col].nunique(dropna=False) <= 1:
                dropped_constants.append(col)
                continue
            kept_confounders.append(col)

        if dropped_constants:
            warnings.warn(
                "Dropping constant confounder columns (zero variance): " + ", ".join(dropped_constants),
                UserWarning,
                stacklevel=2,
            )
        # Update confounders to exclude dropped constants
        self._confounders = kept_confounders

        # Note: duplicate columns and duplicate rows are checked on the stored, normalized subset
        # after dtype coercion, to reflect the actual data used by the class.

    def _check_duplicate_column_values(self, df):
        """
        Check for duplicate column values across all used columns.
        Raises ValueError if any two columns have identical values.
        """
        # Get all columns that will be used in CausalData
        columns_to_check = [self._target, self._treatment] + self._confounders
        
        # Compare each pair of columns (post-normalization)
        for i, col1 in enumerate(columns_to_check):
            for j in range(i + 1, len(columns_to_check)):
                col2 = columns_to_check[j]
                # Use pandas.Series.equals for exact equality on stored subset (NaN not expected)
                if df[col1].equals(df[col2]):
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
        num_duplicates = int(duplicated_mask.sum())
        
        if num_duplicates > 0:
            total_rows = int(len(df_subset))
            unique_rows = total_rows - num_duplicates
            
            warnings.warn(
                f"Found {num_duplicates} duplicate rows out of {total_rows} total rows in the DataFrame. "
                f"This leaves {unique_rows} unique rows for analysis. "
                f"Duplicate rows may affect the quality of causal inference results. "
                f"Consider removing duplicates if they are not intentional.",
                UserWarning,
                stacklevel=2
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
        >>> from causalis.data import generate_rct
        >>> from causalis.data import CausalData
        >>>
        >>> # Generate data
        >>> df = generate_rct()
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

        # Validate that all requested columns exist (only needed if user passed custom columns)
        missing = [c for c in cols_to_include if c not in self.df.columns]
        if missing:
            raise ValueError(f"Column(s) {missing} do not exist in the DataFrame.")

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