from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
#from xgboost import XGBClassifier,XGBRegressor
#from lightgbm import LGBMRegressor
def get_models():
    models = {
        'LogisticRegression': LogisticRegression(random_state=42),
        'RandomForestClassifier': RandomForestClassifier(random_state=42),
        'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
       # 'XGBClassifier': XGBClassifier(random_state=42),
       # 'XGBRegressor': XGBRegressor(random_state=42),
       # 'LGBMRegressor': LGBMRegressor(random_state=42),
        'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42)
    }
    return models
