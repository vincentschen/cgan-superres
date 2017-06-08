import numpy as np
import os.path
import scipy.misc
import tensorflow as tf
import time
import random

FLAGS = tf.app.flags.FLAGS

def _summarize_progress(train_data, feature, label, gene_output, batch, suffix, max_samples=48):
    td = train_data

    size = [label.shape[1], label.shape[2]]

    nearest = tf.image.resize_nearest_neighbor(feature, size)
    nearest = tf.maximum(tf.minimum(nearest, 1.0), 0.0)

    bicubic = tf.image.resize_bicubic(feature, size)
    bicubic = tf.maximum(tf.minimum(bicubic, 1.0), 0.0)

    clipped = tf.maximum(tf.minimum(gene_output, 1.0), 0.0)

    image   = tf.concat([nearest, bicubic, clipped, label], 2)

    image = image[0:max_samples,:,:,:]
    image = tf.concat([image[i,:,:,:] for i in range(max_samples)], 0)
    image = td.sess.run(image)

    filename = 'batch%06d_%s.png' % (batch, suffix)
    filename = os.path.join(FLAGS.train_dir, filename)
    scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)
    print("    Saved %s" % (filename,))

def _save_checkpoint(train_data, batch):
    td = train_data

    oldname = 'checkpoint_old.txt'
    newname = 'checkpoint_new.txt'

    oldname = os.path.join(FLAGS.checkpoint_dir, oldname)
    newname = os.path.join(FLAGS.checkpoint_dir, newname)

    # Delete oldest checkpoint
    try:
        tf.gfile.Remove(oldname)
        tf.gfile.Remove(oldname + '.meta')
    except:
        pass

    # Rename old checkpoint
    try:
        tf.gfile.Rename(newname, oldname)
        tf.gfile.Rename(newname + '.meta', oldname + '.meta')
    except:
        pass

    # Generate new checkpoint
    saver = tf.train.Saver()
    saver.save(td.sess, newname)

    print("    Checkpoint saved")

def train_model(train_data):
    td = train_data

    summaries = tf.summary.merge_all()
    td.sess.run(tf.global_variables_initializer())

    lrval       = FLAGS.learning_rate_start
    start_time  = time.time()
    done  = False
    batch = 0

    assert FLAGS.learning_rate_half_life % 10 == 0

    old_graph = tf.get_default_graph()

    tf.reset_default_graph()

    saver = tf.train.import_meta_graph('checkpoint_64/model.ckpt.meta')

    class_graph = tf.get_default_graph()
    x = class_graph.get_tensor_by_name("x:0")
    y_true = class_graph.get_tensor_by_name("y_true:0")

    new_session = tf.Session(graph=class_graph) 

    saver.restore(new_session,tf.train.latest_checkpoint('checkpoint_64/'))

    # Cache test features and labels (they are small)
    test_feature, test_label = td.sess.run([td.test_features, td.test_labels])
    print("test_feature shape", test_feature.shape)

    tf.reset_default_graph()

    while not done:
        batch += 1
        gene_loss = disc_real_loss = disc_fake_loss = -1.234

        feed_dict = {td.learning_rate : lrval}

        ops_first = [td.gene_loss, td.disc_real_loss, td.disc_fake_loss]
        gene_loss, disc_real_loss, disc_fake_loss = td.sess.run(ops_first, feed_dict=feed_dict)

        gene_output = td.sess.run(td.gene_output, feed_dict=feed_dict)
        train_labels = td.sess.run(td.train_labels, feed_dict=feed_dict)

        new_feed_dict = {x:train_labels.reshape(train_labels.shape[0], train_labels.shape[1]*train_labels.shape[2]*train_labels.shape[3])}

        softmax_output = new_session.run('y_pred:0', feed_dict=new_feed_dict)
        labels_input = np.rint(softmax_output)

        new_feed_dict = {x:gene_output.reshape(gene_output.shape[0], gene_output.shape[1]*gene_output.shape[2]*gene_output.shape[3]), y_true:labels_input}

        class_loss = new_session.run('Mean:0', feed_dict=new_feed_dict)

        feed_dict = {td.learning_rate : lrval, td.class_loss: class_loss}

        ops = [td.gene_minimize, td.disc_minimize]
        hi, _ = td.sess.run(ops, feed_dict=feed_dict)
        
        if batch % 10 == 0:
            # Show we are alive
            elapsed = int(time.time() - start_time)/60
            print('Progress[%3d%%], ETA[%4dm], Batch [%4d], G_Loss[%3.3f], D_Real_Loss[%3.3f], D_Fake_Loss[%3.3f]' %
                  (int(100*elapsed/FLAGS.train_time), FLAGS.train_time - elapsed,
                   batch, gene_loss+class_loss, disc_real_loss, disc_fake_loss))

            # Finished?            
            current_progress = elapsed / FLAGS.train_time
            if current_progress >= 1.0:
                done = True
            
            # Update learning rate
            if batch % FLAGS.learning_rate_half_life == 0:
                lrval *= .5

        if batch % FLAGS.summary_period == 0:
            with old_graph.as_default():
                # Show progress with test features
                feed_dict = {td.gene_minput: test_feature}
                gene_output = td.sess.run(td.gene_moutput, feed_dict=feed_dict)
                _summarize_progress(td, test_feature, test_label, gene_output, batch, 'out')
            
        if batch % FLAGS.checkpoint_period == 0:
            with old_graph.as_default():
                # Save checkpoint
                _save_checkpoint(td, batch)

    _save_checkpoint(td, batch)
    print('Finished training!')
