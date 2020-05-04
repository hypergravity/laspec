from sklearn.decomposition import PCA,KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor


class NN:
    usepca = False
    pca = None

    xscaler = None
    yscaler = None

    xtrain = 0
    ytrain = 0
    xtest = 0
    ytest = 0
    
    regressor = None
    regression = True

    def __init__(self,
                 xtrain, ytrain, regression=True,
                 usepca=False, n_components=0.99,
                 **mlp_kwargs):
        self.xscaler = StandardScaler()
        self.yscaler = StandardScaler()

        # if PCA
        self.usepca = usepca
        self.regression = regression
        if usepca:
            self.pca = KernelPCA(n_components=n_components, kernel="rbf")
            xtrain_pca = self.pca.fit_transform(xtrain)
            print("@nn: explained variance ratio = ", self.pca.explained_variance_ratio_)
            # scale
            self.xscaler.fit(xtrain_pca)
            xtrain_scaled = self.xscaler.transform(xtrain_pca)
        else:
            # scale
            self.xscaler.fit(xtrain)
            xtrain_scaled = self.xscaler.transform(xtrain)
        
        if regression:
            self.yscaler.fit(ytrain)
            ytrain_scaled = self.yscaler.transform(ytrain)
        else:
            ytrain_scaled = ytrain
        print("xtrain.shape =", xtrain_scaled.shape, "ytrain.shape =", ytrain_scaled.shape)
        
        # NN
        _mlp_kwargs = dict(hidden_layer_sizes=(128, 12), activation="sigmoid",
                           solver="adam", learning_rate="invscaling", learning_rate_init=0.001, verbose=True)
        _mlp_kwargs.update(mlp_kwargs)
        if regression:
            self.regressor = MLPRegressor(**_mlp_kwargs)
        else:
            self.regressor = MLPClassifier(**_mlp_kwargs)

        print("@nn: fitting NN ...")
        self.regressor.fit(xtrain_scaled, ytrain_scaled)

    def transform(self, x):
        if self.usepca:
            return self.xscaler.transform(self.pca.transform(x))
        else:
            return self.xscaler.transform(x)

    def predict(self, xtest):
        if not self.regression:
            return self.regressor.predict(self.transform(xtest))
        else:
            return self.yscaler.inverse_transform(
                self.regressor.predict(self.transform(xtest)))
        
    def predict_proba(self, xtest):
        assert not self.regression
        return self.regressor.predict_proba(self.transform(xtest))
