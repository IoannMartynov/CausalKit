"""
Utility functions for splitting traffic data from DataFrames.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union, List


def split_traffic(
    df: pd.DataFrame,
    split_ratio: Union[float, List[float]] = 0.5,
    stratify_column: Optional[str] = None,
    random_state: Optional[int] = None
) -> Tuple[pd.DataFrame, ...]:
    """
    Split a DataFrame into multiple parts based on the specified ratio.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing traffic data.
    split_ratio : float or list of floats, default 0.5
        If float, represents the proportion of the DataFrame to include in the first split.
        If list, each value represents the proportion for each split. The values should sum to 1.
    stratify_column : str, optional
        Column name to use for stratified splitting. If provided, the splits will have
        the same proportion of values in this column.
    random_state : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    tuple of pd.DataFrame
        A tuple containing the split DataFrames. If split_ratio is a float, returns a tuple of two DataFrames.
        If split_ratio is a list, returns a tuple with length equal to len(split_ratio) + 1.
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'user_id': range(100), 'group': ['A', 'B'] * 50})
    >>> train_df, test_df = split_traffic(df, split_ratio=0.8, random_state=42)
    >>> len(train_df), len(test_df)
    (80, 20)
    
    >>> train_df, val_df, test_df = split_traffic(df, split_ratio=[0.7, 0.2], random_state=42)
    >>> len(train_df), len(val_df), len(test_df)
    (70, 20, 10)
    """
    np.random.seed(random_state)
    
    if isinstance(split_ratio, float):
        split_ratio = [split_ratio]
    
    # Validate split_ratio
    if sum(split_ratio) >= 1:
        raise ValueError("Sum of split ratios should be less than 1.")
    
    # Calculate the cumulative split points
    cum_splits = np.cumsum(split_ratio)
    
    # Create a list to store the split DataFrames
    split_dfs = []
    
    if stratify_column is not None and stratify_column in df.columns:
        # Stratified split
        unique_strata = df[stratify_column].unique()
        strata_dfs = []
        
        for stratum in unique_strata:
            stratum_df = df[df[stratify_column] == stratum].copy()
            stratum_indices = stratum_df.index.tolist()
            np.random.shuffle(stratum_indices)
            
            # Calculate split indices for this stratum
            stratum_splits = [int(len(stratum_indices) * split) for split in cum_splits]
            
            # Initialize list for this stratum's splits
            stratum_split_dfs = []
            
            # First split
            stratum_split_dfs.append(stratum_df.loc[stratum_indices[:stratum_splits[0]]])
            
            # Middle splits (if any)
            for i in range(1, len(stratum_splits)):
                stratum_split_dfs.append(
                    stratum_df.loc[stratum_indices[stratum_splits[i-1]:stratum_splits[i]]]
                )
            
            # Last split
            stratum_split_dfs.append(stratum_df.loc[stratum_indices[stratum_splits[-1]:]])
            
            strata_dfs.append(stratum_split_dfs)
        
        # Combine strata for each split
        for i in range(len(cum_splits) + 1):
            split_dfs.append(pd.concat([strata_df[i] for strata_df in strata_dfs], axis=0))
    else:
        # Random split
        indices = df.index.tolist()
        np.random.shuffle(indices)
        
        # Calculate split indices
        splits = [int(len(indices) * split) for split in cum_splits]
        
        # First split
        split_dfs.append(df.loc[indices[:splits[0]]])
        
        # Middle splits (if any)
        for i in range(1, len(splits)):
            split_dfs.append(df.loc[indices[splits[i-1]:splits[i]]])
        
        # Last split
        split_dfs.append(df.loc[indices[splits[-1]:]])
    
    return tuple(split_dfs)