# def combine_ms2(cc_df: pd.DataFrame, configd: Dict) -> pd.DataFrame:
#     """combine the ms2 data for a given connected component graph of ms1 ions.
#     this is done the same way as the ms1 matching (rtree index) but uses the columns
#     defined in global MS2COLSTOMATCH

#     Args:
#         cc_df (pd.DataFrame): conncected component dataframe
#         min_reps (int, optional): Defaults to 2. minimum number of replicates required for ms2 ion to be included

#     Returns:
#         pd.DataFrame: combined ms2 dataframe
#     """

#     # ms2dfs = [pd.read_json(ms2df,orient='split') for ms2df in cc_df['MS2Info']]
#     # ms2dfs = [pd.read_json(ms2df) for ms2df in cc_df['MS2Info']]
#     MS2COLSTOMATCH = configd["MS2COLSTOMATCH"]
#     MS2ERRORCOLS = configd["MS2ERRORCOLS"]
#     FILENAMECOL = configd["FILENAMECOL"]
#     MS2COLS = configd["MS2COLS"]
#     ERRORINFO = configd["ERRORINFO"]
#     MINREPS = configd["MINREPS"]
#     ms2dfs = cc_df["MS2Info"].values.tolist()
#     ms2df = pd.concat(ms2dfs, sort=True)
#     # print(ms2df.columns)
#     if ms2df.shape[0] > 1:
#         add_error_cols(ms2df, MS2COLSTOMATCH, ERRORINFO)
#         rects = get_hyperrectangles(ms2df, MS2ERRORCOLS)
#         rtree = build_rtree(ms2df, MS2ERRORCOLS)
#         ccs = generate_connected_components(rtree, rects)
#         data = []
#         file_col = []
#         for cc in ccs:
#             if len(cc) > 1:
#                 cc_df = ms2df.iloc[list(cc)]
#                 uni_files = set(cc_df[FILENAMECOL].values)
#                 if len(uni_files) >= MINREPS:
#                     data.append(_average_data_rows(cc_df, MS2COLS))
#                     file_col.append("|".join(uni_files))
#             #     else:
#             #         data.append([None]*len(MS2COLS))
#             # else:
#             #     data.append([None]*len(MS2COLS))

#         avg_ms2 = pd.DataFrame(data, columns=MS2COLS)
#         avg_ms2[FILENAMECOL] = file_col
#     else:
#         avg_ms2 = ms2df
#     # return avg_ms2.to_json(orient='split',index=False) #note that to read back to df orient='split' must be set in pd.read_json()
#     return avg_ms2

# def _read_json(ms2json, i):
#     """helper func that can be serialized for multiproc json de-serialization"""
#     return i, pd.read_json(ms2json)

# def reduce_to_ms1(df: pd.DataFrame, configd: dict) -> pd.DataFrame:
#     """takes a dataframe w/ ms2 data in "tidy dataformat" and
#     reduces it down to a ms1 df with a ms2df object stored in MS2Info

#     Args:
#         df (pd.DataFrame): ms2 dataframe in tidy (rectangluar) format
#         FILENAMECOL (str): filename column (needed for de-replication)

#     Returns:
#         df: a ms1df which has the ms2 data in MS2Info column
#     """
#     MS1COLS = configd["MS1COLS"]
#     FILENAMECOL = configd["FILENAMECOL"]
#     gb = df.groupby(MS1COLS + [FILENAMECOL])
#     ms1_data = []
#     for gbi, ms2df in gb:
#         fname = gbi[-1]
#         ms2df = ms2df.copy()
#         ms2df[FILENAMECOL] = [fname] * len(ms2df)
#         # ms1_data.append(list(gbi)+[ms2df.to_json(orient='split',index=False)])
#         # ms1_data.append(list(gbi)+[ms2df.to_json()])
#         ms1_data.append(list(gbi) + [ms2df])
#     cols = MS1COLS + [FILENAMECOL, "MS2Info"]
#     ms1df = pd.DataFrame(ms1_data, columns=cols)

#     return ms1df
