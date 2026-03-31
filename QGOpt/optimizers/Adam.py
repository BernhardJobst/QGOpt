import QGOpt.manifolds as m
import tensorflow as tf
import types

class RAdam(tf.keras.optimizers.Optimizer):
    """Riemannain Adam and AMSGrad optimizers. Returns a new optimizer.
    Updated from QGOpt to work with tensorflow 2.17.

    Args:
        manifold: object of the class Manifold, marks a particular manifold.
        learning_rate: real number. A learning rate. Defaults to 0.05.
        beta1: real number. An exponential decay rate for the first moment.
            Defaults to 0.9.
        beta2: real number. An exponential decay rate for the second moment.
            Defaults to 0.999.
        eps: real number. Regularization coefficient. Defaults to 1e-8.
        ams: boolean number. Use ams (AMSGrad) or not.
        name: Optional name prefix for the operations created when applying
            gradients.  Defaults to 'RAdam'.

    Notes:
        The optimizer works only with real valued tf.Variable of shape
        (..., q, p, 2), where (...) -- enumerates manifolds
        (can be either empty or any shaped),
        q and p the size of a matrix, the last index marks
        real and imag parts of a matrix
        (0 -- real part, 1 -- imag part)"""

    def __init__(self,
                 manifold,
                 learning_rate=0.05,
                 beta1=0.9,
                 beta2=0.999,
                 eps=1e-8,
                 ams=False,
                 name="RAdam",
                 **kwargs):

        super().__init__(
            learning_rate=learning_rate,
            name=name,
            **kwargs)
        self.eps = eps
        # manifold could be dict when loading model
        if isinstance(manifold, dict):
            self.manifold = m.StiefelManifold(**manifold['config'])
        else:
            self.manifold = manifold
        # make manifold keras compatible
        def get_config(self):
            return {'retraction': self._retraction,
                    'metric': self._metric}
        self.manifold.get_config = types.MethodType(get_config, self.manifold)
        
        if isinstance(beta1, (int, float)) and (beta1 < 0 or beta1 > 1):
            raise ValueError("`beta1` must be between [0, 1].")
        self.beta1 = beta1
        if isinstance(beta2, (int, float)) and (beta2 < 0 or beta2 > 1):
            raise ValueError("`beta2` must be between [0, 1].")
        self.beta2 = beta2

        self.ams = ams

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._momentums = []
        self._velocities = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="momentum"
                )
            )
            self._velocities.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="velocity"
                )
            )
        if self.ams:
            self._velocity_hats = []
            for var in var_list:
                self._velocity_hats.append(
                    self.add_variable_from_reference(
                        reference_variable=var, name="velocity_hat"
                    )
                )
    
    def update_step(self, gradient, variable, learning_rate):
        # Complex version of grad and var
        complex_var = m.real_to_complex(variable)
        complex_grad = m.real_to_complex(gradient)

        # learning rate and iter
        lr = tf.cast(learning_rate, complex_grad.dtype)
        iterations = tf.cast(self.iterations + 1, complex_grad.dtype)

        # Riemannian gradient
        rgrad = self.manifold.egrad_to_rgrad(complex_var, complex_grad)

        # Complex versions of m and v
        momentum = self._momentums[self._get_variable_index(variable)]
        v = self._velocities[self._get_variable_index(variable)]
        if self.ams:
            v_hat = self._velocity_hats[self._get_variable_index(variable)]
            v_hat_complex = m.real_to_complex(v_hat)
        momentum_complex = m.real_to_complex(momentum)
        v_complex = m.real_to_complex(v)

        # Update m, v and v_hat
        beta1 = tf.cast(self.beta1, dtype=momentum_complex.dtype)
        beta2 = tf.cast(self.beta2, dtype=momentum_complex.dtype)
        momentum_complex = beta1 * momentum_complex +\
            (1 - beta1) * rgrad
        v_complex = beta2 * v_complex +\
            (1 - beta2) * self.manifold.inner(complex_var,
                                              rgrad,
                                              rgrad)
        if self.ams:
            v_hat_complex = tf.maximum(tf.math.real(v_complex),
                                       tf.math.real(v_hat_complex))
            v_hat_complex = tf.cast(v_hat_complex, dtype=v_complex.dtype)

        # Bias correction
        lr_corr = lr * tf.math.sqrt(1 - beta2 ** iterations) /\
            (1 - beta1 ** iterations)

        # New value of var
        if self.ams:
            # Search direction
            search_dir = -lr_corr * momentum_complex /\
                (tf.sqrt(v_hat_complex) + self.eps)
            new_var, momentum_complex =\
                self.manifold.retraction_transport(complex_var,
                                                   momentum_complex,
                                                   search_dir)
        else:
            # Search direction
            search_dir = - lr_corr * momentum_complex /\
                (tf.sqrt(v_complex) + self.eps)
            new_var, momentum_complex =\
                self.manifold.retraction_transport(complex_var,
                                                   momentum_complex,
                                                   search_dir)

        # Assigning new value of momentum
        momentum.assign(m.complex_to_real(momentum_complex))
        # Assigning new value of v and v_hat
        v.assign(m.complex_to_real(v_complex))
        if self.ams:
            v_hat.assign(m.complex_to_real(v_hat_complex))

        # Update of var
        variable.assign(m.complex_to_real(new_var))
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "ams": self.ams,
            "manifold": self.manifold
        })
        return config
