import numpy as np

def Time_features(df):

    df['delta_time'] = df['time_step'].diff()
    df['delta_time'].fillna(0, inplace = True)
    df['delta_time'].mask(df['delta_time'] < 0, 0, inplace = True)
    df['total_time'] = df.groupby('breath_id')['time_step'].transform('max')

    return df

def Lag_features(df):

    df['u_in_lag1'] = df.groupby('breath_id')['u_in'].shift(1)
    df['u_in_lag2'] = df.groupby('breath_id')['u_in'].shift(2)
    df['u_in_lag3'] = df.groupby('breath_id')['u_in'].shift(3)
    df['u_in_lag4'] = df.groupby('breath_id')['u_in'].shift(4)
    df['u_out_lag1'] = df.groupby('breath_id')['u_out'].shift(1)
    df['u_out_lag2'] = df.groupby('breath_id')['u_out'].shift(2)
    df['u_out_lag3'] = df.groupby('breath_id')['u_out'].shift(3)
    df['u_out_lag4'] = df.groupby('breath_id')['u_out'].shift(4)
    df = df.fillna(0)

    return df

def Back_features(df):
    df['u_in_back1'] = df.groupby('breath_id')['u_in'].shift(-1)
    df['u_in_back2'] = df.groupby('breath_id')['u_in'].shift(-2)
    df['u_in_back3'] = df.groupby('breath_id')['u_in'].shift(-3)
    df['u_in_back4'] = df.groupby('breath_id')['u_in'].shift(-4)
    df = df.fillna(0)
    df['u_out_back1'] = df.groupby('breath_id')['u_out'].shift(-1)
    df['u_out_back2'] = df.groupby('breath_id')['u_out'].shift(-2)
    df['u_out_back3'] = df.groupby('breath_id')['u_out'].shift(-3)
    df['u_out_back4'] = df.groupby('breath_id')['u_out'].shift(-4)
    df = df.fillna(1)

    return df

def Global_stat_features(df):

    df['u_in_max'] = df.groupby('breath_id')['u_in'].transform('max')
    df['u_in_mean'] = df.groupby('breath_id')['u_in'].transform('mean')
    df['u_in_std'] = df.groupby('breath_id')['u_in'].transform('std')

    return df


def SlidingWindow_features(df):
    
    WindowSize = [3, 5, 7, 9] 
    ## number of time steps for each breathe
    ##df.shape[0] can be exact divided by WindowSize, int only avoids the type of n_windows be float
    for (padding_size, L) in enumerate(WindowSize):
        column_name_max = 'u_in_SW' + 'max' + str(L)
        column_name_min = 'u_in_SW' + 'min' + str(L)
        column_name_u_out = 'u_out_SW' + '_is_keyPoint_in' + str(L)
        column_name_u_in_inWindow = ['u_in']
        column_name_u_out_inWindow = ['u_in']
        for i in range(padding_size + 1):
            column_name_u_in_inWindow.append('u_in_lag' + str(i + 1))
            column_name_u_in_inWindow.append('u_in_back' + str(i + 1))
            column_name_u_out_inWindow.append('u_out_lag' + str(i + 1))
            column_name_u_out_inWindow.append('u_out_back' + str(i + 1))
        df[column_name_max] = np.max(df[column_name_u_in_inWindow].to_numpy(), axis = 1)
        df[column_name_min] = np.min(df[column_name_u_in_inWindow].to_numpy(), axis = 1)
        df[column_name_u_out] = (np.max(df[column_name_u_out_inWindow].to_numpy(), axis = 1) \
                                -np.min(df[column_name_u_out_inWindow].to_numpy(), axis = 1)).astype(int)
    return df

def Other_features(df):
    df['R_C'] = df['R'] * df['C']
    return df

def Feature_engineering(df):

    print('Constructing Time features...')
    df = Time_features(df)
    print('Done!')
    print('Constructing Lag features...')
    df = Lag_features(df)
    print('Done!')
    print('Constructing Back features...')
    df = Back_features(df)
    print('Done!')
    print('Constructing Global statistic features...')
    df= Global_stat_features(df)
    print('Done!')
    print('Constructing Sliding window features...')
    df = SlidingWindow_features(df)
    print('Done!')
    print('Constructing Other features...')
    df = Other_features(df)
    print('Done!')
    return df