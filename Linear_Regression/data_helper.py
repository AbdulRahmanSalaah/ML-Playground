def preprocessing(data, option = 1):
    '''
    Args:
        data: numpy array examples x features
        option: 1 for MinMaxScaler and 2 for StandardScaler

    Returns: preprocessed data
    '''
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    if option == 1:
        processor = MinMaxScaler()
    elif option == 2:
        processor = StandardScaler()
    else:
        return data, None # no preprocessing

    return processor.fit_transform(data), processor


def load_data(data_path, preprocessing_option = 1):
    import pandas as pd
    df = pd.read_csv(data_path)
    data = df.to_numpy()

    x = data[:, :3] # first 3 columns
    t = data[:, -1] # last column

    x, _ = preprocessing(x, preprocessing_option)  # preprocess the input data

    return df, data, x, t

