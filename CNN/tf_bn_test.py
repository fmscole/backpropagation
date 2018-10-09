import tensorflow as tf
import numpy as np
import  functools
my_batchn = functools.partial(
        tf.layers.batch_normalization,
        axis=1,
        momentum=.95,
        epsilon=1e-5,
        center=True,
        scale=True,
        fused=True,
        training=True)
tx=tf.placeholder(tf.float32,[None,2,2,2],name='pos_tensor')
x=np.array(np.arange(24)).reshape((3,2,2,2))

print(x[0,0])
bx=my_batchn(tx)


init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

b=session.run(bx,feed_dict={tx:x})
print(b[0,0])