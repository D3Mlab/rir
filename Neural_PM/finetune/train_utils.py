import tensorflow as tf


def setup_tpu():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    print(resolver.master())
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)
    return strategy
