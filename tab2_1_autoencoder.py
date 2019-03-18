import tensorflow as tf

def auto_encoder(X, dim=2, epoch=300, lr=0.001, batch_size=64):
    tf.reset_default_graph()

    total_batch = X.shape[0] // batch_size
    feature = X.shape[1]
    x = tf.placeholder(tf.float32, shape=(None, feature))

    with tf.variable_scope('Autoencoder'):
        with tf.variable_scope('Encoder'):
            encoder = tf.layers.dense(x, dim, activation=tf.nn.sigmoid)

        with tf.variable_scope('Decoder'):
            decoder = tf.layers.dense(encoder, feature, activation=tf.nn.sigmoid)

        with tf.variable_scope('loss'):
            y_true = x
            y_pred = decoder

            loss = tf.reduce_mean(tf.abs(y_true - y_pred))
            optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

    sess = tf.InteractiveSession()
    train_loss = 0
    sess.run(tf.global_variables_initializer())

    for e in range(epoch):

        for t in range(total_batch):
            offset = (t * batch_size) % (X.shape[0] - batch_size)

            X_ = X[offset:(offset + batch_size), :]

            feed_dict = {x: X_}
            _, loss_t = sess.run([optimizer, loss], feed_dict)

            train_loss += loss_t

        feed_dict = {x: X}
        loss_test = sess.run(loss, feed_dict)
        print("%d epoch's train loss / test loss: %f / %f" % ((e + 1, loss_t, loss_test)))

    print("Encoding complete")

    feed_dict = {x: X}
    X_encoded = sess.run(encoder, feed_dict)

    sess.close()

    return X_encoded