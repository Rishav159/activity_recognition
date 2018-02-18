import numpy as np

class LLayerNeuralNetwork:
    def __init__(self, hidden_layer_sizes=[7,5,3], learning_rate=0.01, iterations=3000):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.layer_sizes = [self.X.shape[0]]+self.hidden_layer_sizes+[self.Y.shape[0]]
        self._initialize_parameters(self.layer_sizes)
        for i in range(self.iterations):
            AL, caches = self._forward_propagation(self.X)
            cost = self._compute_cost(AL)
            if (i+1)%100 == 0:
                print('Cost after '+str(i+1)+'th iteration is '+str(cost))
            grads = self._backward_propagation(AL, caches)
            self._update_parameters(grads)

    def predict(self, X):
        AL, caches = self._forward_propagation(X)
        return AL>0.5

    def _initialize_parameters(self, layer_sizes):
        self.parameters = {}
        for l in range(1,len(layer_sizes)):
            self.parameters['W'+str(l)]=np.random.randn(layer_sizes[l],layer_sizes[l-1])*0.01
            self.parameters['b'+str(l)]=np.zeros((layer_sizes[l],1))

    def _forward_propagation(self, X):
        A_prev = X
        caches = []
        L = len(self.parameters) // 2
        # Forward propagation for L-1 layers with ReLU activation function
        for l in range(L-1):
            W, b = self.parameters['W'+str(l+1)], self.parameters['b'+str(l+1)]
            Z = np.dot(W, A_prev) + b
            caches.append((A_prev,Z))
            A_prev = self._relu(Z)
        # Forward propagation for last layer with sigmoid activation function
        W, b = self.parameters['W'+str(L)], self.parameters['b'+str(L)]
        Z = np.dot(W, A_prev) + b
        caches.append((A_prev,Z))
        AL = self._sigmoid(Z)
        return AL, caches

    def _compute_cost(self, AL):
        m = self.X.shape[1]
        cost = -(np.dot(self.Y,np.log(AL).T)+np.dot(1-self.Y,np.log(1-AL).T))/m
        cost = np.squeeze(cost)
        return cost

    def _backward_propagation(self, AL, caches):
        L = len(caches)
        grads = {}
        m = self.X.shape[1]
        dAL = -np.divide(self.Y,AL)+np.divide(1-self.Y,1-AL)
        # Backward propagation for last layer with sigmoid derivative
        dZL = self._sigmoid_backward(dAL,caches[L-1][1])
        dW = np.dot(dZL,caches[L-1][0].T)/m
        db = np.sum(dZL,axis=1,keepdims=True)/m
        grads['dW'+str(L)], grads['db'+str(L)] = dW, db
        dA = np.dot(self.parameters['W'+str(L)].T, dZL)
        # Backward propagation for L-1 layers with ReLU derivative
        for l in range(L-2,-1,-1):
            dZ = self._relu_backward(dA,caches[l][1])
            dW = np.dot(dZ,caches[l][0].T)/m
            db = np.sum(dZ,axis=1,keepdims=True)/m
            grads['dW'+str(l+1)], grads['db'+str(l+1)] = dW, db
            dA = np.dot(self.parameters['W'+str(l+1)].T, dZ)
        return grads

    def _update_parameters(self, grads):
        L = len(grads) // 2
        for l in range(L):
            self.parameters['W'+str(l+1)] -= self.learning_rate * grads['dW'+str(l+1)]
            self.parameters['b'+str(l+1)] -= self.learning_rate * grads['db'+str(l+1)]

    def _relu(self, Z):
        return np.maximum(0,Z)

    def _sigmoid(self, Z):
        return 1/(1+np.exp(-Z))

    def _relu_backward(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z<=0] = 0
        return dZ

    def _sigmoid_backward(self, dA, Z):
        s = 1/(1+np.exp(-Z))
        return dA * s * (1-s)
