import cPickle as pickle
import numpy as np
from lazyflowClassifier import LazyflowVectorwiseClassifierABC, LazyflowVectorwiseClassifierFactoryABC

import xgboost as xgb

from lazyflow.request import Request

import logging
logger = logging.getLogger(__name__)

class XgboostLazyflowClassifierFactory(LazyflowVectorwiseClassifierFactoryABC):
    """
    A factory for creating and training a XGBoost classifier.
    """
    VERSION = 1 # This is used to determine compatibility of pickled classifier factories.
                # You must bump this if any instance members are added/removed/renamed.
    
    def __init__(self, *args, **kwargs):
        """
        args, kwargs: Passed on to the classifier constructor.
        """
        self._args = args
        self._kwargs = kwargs
        self._num_threads = Request.global_thread_pool.num_workers

    def create_and_train(self, X, y, feature_names=None):
        logger.info("Training XGBoost classifier.")
        
        X = np.asarray(X, np.float32)
        y = np.asarray(y, np.uint32)
        known_classes = np.unique(y)
        
        y = y - 1 # xgboost label array must be within the range [0, num_class)

        assert X.ndim == 2
        assert len(X) == len(y)
            
        # Set classifier parameters (for reference see: https://github.com/dmlc/xgboost/blob/master/doc/parameter.md)
        param = {'bst:max_depth':2, 'bst:eta':1, 'num_class':2,'objective':'multi:softprob' }
        param['nthread'] = self._num_threads
        param['num_round'] = self._num_threads
        
        # Train XGBoost classifier
        dtrain = xgb.DMatrix(X, label=y)
        #num_round = self._num_threads
        xgboost_classifier = xgb.train(param, dtrain)#, num_round)
                
        return XgboostLazyflowClassifier( xgboost_classifier, known_classes, X.shape[1], feature_names )

    @property
    def description(self):
        return "XGBoost Classifier"

    def __eq__(self, other):
        return (    isinstance(other, type(self))
                and self._args == other._args
                and self._kwargs == other._kwargs )
    def __ne__(self, other):
        return not self.__eq__(other)

assert issubclass( XgboostLazyflowClassifierFactory, LazyflowVectorwiseClassifierFactoryABC )

class XgboostLazyflowClassifier(LazyflowVectorwiseClassifierABC):
    """
    Adapt the XGBoost classifier class to the interface lazyflow expects.
    """
    VERSION = 1 # Used for pickling compatibility

    class VersionIncompatibilityError(Exception):
        pass

    def __init__(self, xgboost_classifier, known_classes, feature_count, feature_names):
        self._xgboost_classifier = xgboost_classifier
        self._known_classes = known_classes
        self._feature_count = feature_count
        self._feature_names = feature_names

        self.VERSION = XgboostLazyflowClassifier.VERSION
    
    def predict_probabilities(self, X):
        logger.debug( 'Predicting with xgboost classifier: {}'.format( type(self._xgboost_classifier).__name__ ) )
        
        dpredict = xgb.DMatrix(X)
        
        return self._xgboost_classifier.predict(dpredict) #self._sklearn_classifier.predict_proba( np.asarray(X, dtype=np.float32) )
    
    @property
    def known_classes(self):
        return self._known_classes

    @property
    def feature_count(self):
        return self._feature_count

    @property
    def feature_names(self):
        return self._feature_names

    def serialize_hdf5(self, h5py_group):
        h5py_group['pickled_classifier'] = pickle.dumps( self )

        # This is a required field for all classifiers
        h5py_group['pickled_type'] = pickle.dumps( type(self) )

    @classmethod
    def deserialize_hdf5(cls, h5py_group):
        pickled = h5py_group['pickled_classifier'][()]
        classifier = pickle.loads( pickled )
        if not hasattr(classifier, "VERSION") or classifier.VERSION != cls.VERSION:
            raise cls.VersionIncompatibilityError("Version mismatch. Deserialized classifier version does not match this code base.")
        return classifier

assert issubclass( XgboostLazyflowClassifier, LazyflowVectorwiseClassifierABC )
