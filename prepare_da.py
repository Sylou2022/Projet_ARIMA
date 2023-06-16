def prepare_data(df, coef_t, coef_tst):
    tail = len(df)
    tail_t = int(tail * coef_t)
    tail_tst = int(tail * coef_tst)
    
    t_train = df.index[:tail_t]
    t_test = df.index[tail_t:(tail_t + tail_tst)]
    t_validation = df.index[(tail_t + tail_tst):]
    
    y_train = df['total_cases'][:tail_t]
    y_test = df['total_cases'][tail_t:(tail_t + tail_tst)]
    y_validation = df['total_cases'][(tail_t + tail_tst):]
    
    return t_train, t_test, t_validation, y_train, y_test, y_validation
