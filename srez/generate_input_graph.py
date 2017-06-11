import tensorflow as tf

sess=tf.Session()
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('checkpoint_eye/model.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('checkpoint_eye/'))
graph = tf.get_default_graph()

model_dir = "checkpoint_eye"
input_graph_name = "input_graph"
tf.train.write_graph(sess.graph.as_graph_def(), model_dir, input_graph_name)
