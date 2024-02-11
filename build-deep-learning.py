from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16
import tensorflow as tf
import json
from tensorflow.keras.models import load_model

LOGDIR = 'logs'

class FaceTracker(Model): 
    def __init__(self, eyetracker,  **kwargs): 
        super().__init__(**kwargs)
        self.model = eyetracker

    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt
    
    def train_step(self, batch, **kwargs): 
        
        x, y = batch
        
        with tf.GradientTape() as tape: 
            classes, coords = self.model(x, training=True)
            
            batch_classloss = self.closs(y[0], classes)
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
            
            total_loss = batch_localizationloss+0.5*batch_classloss
            
            grad = tape.gradient(total_loss, self.model.trainable_variables)
        
        self.opt.apply_gradients(zip(grad, self.model.trainable_variables))
        
        return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_localizationloss}
    
    def test_step(self, batch, **kwargs): 
        X, y = batch
        
        classes, coords = self.model(X, training=False)
        
        batch_classloss = self.closs(y[0], classes)
        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total_loss = batch_localizationloss+0.5*batch_classloss
        
        return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_localizationloss}
        
    def call(self, X, **kwargs): 
        return self.model(X, **kwargs)

def load_image(x): 
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding = "utf-8") as f:
        label = json.load(f)
        
    return [label['class']], label['bbox']

def localization_loss(y_true, yhat):            
    delta_coord = tf.reduce_sum(tf.square(y_true[:,:2] - yhat[:,:2]))
                  
    h_true = y_true[:,3] - y_true[:,1] 
    w_true = y_true[:,2] - y_true[:,0] 

    h_pred = yhat[:,3] - yhat[:,1] 
    w_pred = yhat[:,2] - yhat[:,0] 
    
    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true-h_pred))
    
    return delta_coord + delta_size

def get_images_and_labels(folder, shuffle):
    images = tf.data.Dataset.list_files(f'aug_data/{folder}/images/*.jpg', shuffle=False)
    images = images.map(load_image)
    images = images.map(lambda x: tf.image.resize(x, (120,120)))
    images = images.map(lambda x: x/255)

    labels = tf.data.Dataset.list_files(f'aug_data/{folder}/labels/*.json', shuffle=False)
    labels = labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

    dataset = tf.data.Dataset.zip((images, labels))
    dataset = dataset.shuffle(shuffle)
    dataset = dataset.batch(8)
    dataset = dataset.prefetch(4)

    return dataset

def build_model(): 
    input_layer = Input(shape=(120,120,3))
    
    vgg = VGG16(include_top=False)(input_layer)

    # Classification Model  
    f1 = GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048, activation='relu')(f1)
    class2 = Dense(1, activation='sigmoid')(class1)
    
    # Bounding box model
    f2 = GlobalMaxPooling2D()(vgg)
    regress1 = Dense(2048, activation='relu')(f2)
    regress2 = Dense(4, activation='sigmoid')(regress1)
    
    facetracker = Model(inputs=input_layer, outputs=[class2, regress2])
    return facetracker

def main():
    facetracker = build_model()

    train = get_images_and_labels("train", 5000)
    x, y = train.as_numpy_iterator().next()
    classes, coords = facetracker.predict(x)

    batches_per_epoch = len(train)
    lr_decay = (1./0.75 -1)/batches_per_epoch
    learning_rate = 0.0001
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, decay=lr_decay)

    classloss = tf.keras.losses.BinaryCrossentropy()
    regressloss = localization_loss

    localization_loss(y[1], coords)
    classloss(y[0], classes)
    regressloss(y[1], coords)

    model = FaceTracker(facetracker)
    model.compile(opt, classloss, regressloss)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOGDIR)
    val = get_images_and_labels("val", 1000)
    hist = model.fit(train, epochs=10, validation_data=val, callbacks=[tensorboard_callback])

    facetracker.save('facetracker.h5')

if __name__ == "__main__":
    main()