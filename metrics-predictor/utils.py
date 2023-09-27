def get_standard_col_name(sorter_set, sorter_shorthand, match_score):
    """
    helper for add_custom_subset_to_cumulative_df
    create standardised col name
    """   
     
    str_match_score = str(match_score).replace('.', '_')

    # check if all sorter names are in sorter_shorthand
    if not all(sorter in sorter_shorthand.keys() for sorter in sorter_set):
        sorter_set.sort()
        sorter_component = '_'.join(sorter_set)
    else:
        shorthand_list = [sorter_shorthand[sorter] for sorter in sorter_set]
        shorthand_list.sort()
        sorter_component = '_'.join(shorthand_list)

    col_name = 'agreement_' + str_match_score + '_' + sorter_component
    # col_name = 'agreement_' + rec + '_' + str_match_score + '_' + sorter_component

    return col_name

def add_col_for_agr_subset(df_to_add_col, sorter_agr_dict, col_name):
    """
    Given a dataframe, add a column with 1 if the unit is in the sorter_agr_dict and 0 otherwise
    helper for add_custom_subset_to_cumulative_df
    :param df_to_add_col: dataframe to add column to
    :param sorter_agr_dict: dictionary of agreed units (should already be cut down using get_subset_sorter_agreement if using subset of sorters), expected format is {global_id : {sorter : unit_id}}
    :param col_name: header for column to add
    :return: df_to_add_col: dataframe with added column
    """
    # print(col_name)
    # if col_name in df_to_add_col.keys():
    #     print(f"WARNING: column {col_name} already exists in dataframe, appending")
    # else:
    #     df_to_add_col[col_name] = 0
    #     # df_to_add_col.loc[:,col_name] = 0

    for glob_unit_dict in sorter_agr_dict.values():
        for sorter, unit_id in glob_unit_dict.items():
            df_to_add_col.loc[(df_to_add_col['sorter']==sorter) & (df_to_add_col['sorter_unit_id']==unit_id), col_name] = 1

    return df_to_add_col

def print_missing_values(df, print_result=True):
    """
    Prints the number of missing values in each column of a dataframe
    Returns dict with non-zero {metric : na_count} pairs
    """
    na_dict = {}

    for col in df.columns:
        na_count = df[col].isna().sum()
        if na_count > 0:
            na_dict[col] = na_count
            if print_result:
                print(f"{col:{25}} has:  {na_count:{4}} out of {len(df)} missing values")

    if print_result:
        print(
            f"the following columns have no missing values:\n {list(df.columns[df.isna().sum()==0])}"
        )

    return na_dict


def drop_cols_with_all_nans(df, na_dict, print_result=True):
    """
    Drops columns with all nans from dataframe
    """
    num_units = len(df)

    cols_to_drop = []

    for met, na_count in na_dict.items():
        if (
            na_count == num_units and met in df.columns
        ):  # protect in case function applied twice
            cols_to_drop.append(met)
    if print_result:
        print('dropping metrics without valid values: '+', '.join(map(str, cols_to_drop)))

    return df.drop(columns=cols_to_drop)
