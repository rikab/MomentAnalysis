from energyflow.archs.moment import EFN_moment, PFN_moment
import numpy as np
from sklearn.utils import shuffle

class ModelsContainer:
    def __init__(self, Phi_mapping_dim, output_dim, Phi_sizes, F_sizes, order, loss, metrics, 
                architecture_type='moment', F_acts='LeakyReLU', Phi_acts='LeakyReLU', output_act='sigmoid', category = "EFN", **kwargs):
    
        self.config = {'Phi_mapping_dim' : Phi_mapping_dim,
                    'output_dim' : output_dim, 'output_act' : output_act,
                    'Phi_sizes' : Phi_sizes, 'Phi_acts' : Phi_acts,
                    'F_sizes' : F_sizes, 'F_acts': F_acts,
                    'order' : order , 'architecture_type':architecture_type,
                    'loss': loss,'metrics': metrics, **kwargs}
        
        self.category = category
  
        self.models = []
        self.histories = []
        tmp = EFN_moment(**self.config, summary=False) if self.category == "EFN" else PFN_moment(**self.config, summary=False)
        self.num_params = tmp.count_params()
    
    def train_models(self, num_models, X_train, Y_train, validation_data, epochs, batch_size, path, callbacks = None, verbose=0, persist = False, metric_function = None):
        scores = []
        for i in range(num_models):

            if self.category == "EFN":
                z_train, p_train, Y_train = shuffle(X_train[0], X_train[1], Y_train)
                X_train = [z_train,p_train]
            elif self.category == "PFN":
                X_train, Y_train = shuffle(X_train, Y_train)
            model = EFN_moment(**self.config, summary=False) if self.category == "EFN" else PFN_moment(**self.config, summary=False)
            if callbacks == None:
                history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data = validation_data, verbose=verbose)
            else:
                history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data = validation_data, callbacks=callbacks, verbose=verbose)
            
            if persist:
                self.models.append(model)
            self.histories.append(history.history)
            model.save_weights(path+'_'+str(i) + ".keras")

            scores.append(metric_function(validation_data[1], model.predict(validation_data[0])))


        scores = np.array(scores)
        return [np.average(scores), np.std(scores)]    

    def test_meanstd(self, X_test, Y_test, metric_function):
        scores = []
        for model in self.models:
            scores.append(metric_function(Y_test, model.predict(X_test)))
        scores = np.array(scores)
        return [np.average(scores), np.std(scores)]

    def save_models(self, path):
        for i, model in enumerate(self.models):
            model.save(path+'_'+str(i) + ".keras")

    def save_model_weights(self, path):
        for i, model in enumerate(self.models):
            model.save_weights(path+'_'+str(i) + ".keras")

    def vary_parameters(self, variable, new_value):
        tmp_config = self.config
        tmp_config[variable] = new_value
        tmp_container = ModelsContainer(**tmp_config)
        return tmp_container




    

    


