import numpy as np
import sklearn
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

# You are allowed to import any submodules of sklearn as well e.g. sklearn.svm etc
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES OTHER THAN SKLEARN WILL RESULT IN A STRAIGHT ZERO

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_predict etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################


def my_fit(Z_train):
    ################################
    #  Non Editable Region Ending  #
    ################################

    # Use this method to train your model using training CRPs
    # The first 64 columns contain the config bits
    # The next 4 columns contain the select bits for the first mux
    # The next 4 columns contain the select bits for the second mux
    # The first 64 + 4 + 4 = 72 columns constitute the challenge
    # The last column contains the response
    # --------------------------------------------#
    # --------------------------------------------#

    # Create a new training set with the one hot encoding of the config bits and the select bits for mux
    trn = np.zeros((Z_train.shape[0], 65*16))

    for i in range(Z_train.shape[0]):
        mux0 = Z_train[i, 64:68]
        mux1 = Z_train[i, 68:72]

        # covert to base 10
        m = np.array([8, 4, 2, 1])
        mux0 = np.dot(mux0, m).astype(int)
        mux1 = np.dot(mux1, m).astype(int)

        mux0 *= 65
        mux1 *= 65

        # one hot encoding and -ve encoding
        trn[i, mux0: mux0 + 64] = Z_train[i, 0:64]
        trn[i, mux1: mux1 + 64] = -Z_train[i, 0:64]

        # for the bias
        trn[i, mux0 + 64] = 1
        trn[i, mux1 + 64] = -1

    # Create the linear svm model
    model = LinearSVC(fit_intercept=False, dual=False, C=10)

    # Train the model
    model.fit(trn, Z_train[:, 72])
# --------------------------------------------#
# --------------------------------------------#

    return model					# Return the trained model


################################
# Non Editable Region Starting #
################################
def my_predict(X_tst, model):
    ################################
    #  Non Editable Region Ending  #
    ################################

    # Use this method to make predictions on test challenges
    # --------------------------------------------#
    # --------------------------------------------#

    # Create a new test set with the one hot encoding of the config bits and the select bits for mux
    tst = np.zeros((X_tst.shape[0], 65*16))

    for i in range(X_tst.shape[0]):
        mux0 = X_tst[i, 64:68]
        mux1 = X_tst[i, 68:72]

        # covert to base 10
        m = np.array([8, 4, 2, 1])
        mux0 = np.dot(mux0, m).astype(int)
        mux1 = np.dot(mux1, m).astype(int)

        mux0 *= 65
        mux1 *= 65

        # one hot encoding and -ve encoding
        tst[i, mux0: mux0 + 64] = X_tst[i, 0:64]
        tst[i, mux1: mux1 + 64] = -X_tst[i, 0:64]

        # for the bias
        tst[i, mux0 + 64] = 1
        tst[i, mux1 + 64] = -1

    # Predict the output
    pred = model.predict(tst)

    return pred
