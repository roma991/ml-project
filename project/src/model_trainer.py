from sklearn.model_selection import GridSearchCV

def train_model(model, parameters, train_x, train_y, config):
    search = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        cv=config['search']['cv'],
        scoring=config['search']['scoring']
    )
    search.fit(train_x, train_y)
    return search

def get_model_parameters(config, name):
    return config['search_space']['models'][name]['parameters']

def should_train_model(config, name):
    return name in config['search_space']['models']